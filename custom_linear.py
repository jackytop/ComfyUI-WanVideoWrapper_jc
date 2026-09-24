import math
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from .gguf.gguf_utils import GGUFParameter, dequantize_gguf_tensor

# INT8 weights in ComfyUI's mixed precision format ("int8_tensorwise", optionally ConvRot rotated).
# comfy_kitchen ships with recent ComfyUI and provides fused INT8 matmul kernels, without it the weights are dequantized on the fly
try:
    import comfy_kitchen  # noqa: F401 registers torch.ops.comfy_kitchen
    INT8_KITCHEN_AVAILABLE = hasattr(torch.ops.comfy_kitchen, "int8_linear")
except Exception:
    INT8_KITCHEN_AVAILABLE = False

_INT8_DTYPE_CODES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_HADAMARD_CACHE = {}

def _convrot_hadamard(size, device, dtype):
    # normalized regular Hadamard matrix built from the 4x4 regular core, same construction as comfy_kitchen's ConvRot
    key = (size, str(device), dtype)
    if key not in _HADAMARD_CACHE:
        if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
            raise ValueError(f"ConvRot group size must be a power of 4, got {size}")
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32, device=device)
        h = h4
        while h.shape[0] < size:
            h = torch.kron(h, h4)
        _HADAMARD_CACHE[key] = (h / size**0.5).to(dtype)
    return _HADAMARD_CACHE[key]

def dequantize_int8_weight(qdata, scale, convrot_groupsize=0, dtype=torch.bfloat16):
    """INT8 weight [out, in] with a scalar or per output channel scale, ConvRot weights are rotated back to the original basis."""
    if INT8_KITCHEN_AVAILABLE and qdata.is_cuda and dtype in _INT8_DTYPE_CODES:
        if convrot_groupsize > 0:
            return torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(qdata, scale, convrot_groupsize, _INT8_DTYPE_CODES[dtype])
        return torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(qdata, scale, _INT8_DTYPE_CODES[dtype])
    scale = scale.to(qdata.device, torch.float32)
    weight = qdata.float() * (scale.reshape(-1, 1) if scale.numel() > 1 else scale.reshape(()))
    if convrot_groupsize > 0:
        out_f, in_f = weight.shape
        h = _convrot_hadamard(convrot_groupsize, weight.device, torch.float32)
        weight = torch.matmul(weight.view(out_f, in_f // convrot_groupsize, convrot_groupsize), h.T).view(out_f, in_f)
    return weight.to(dtype)

def parse_int8_quant_config(state_dict):
    """Collects the INT8 layers from ComfyUI's per-layer "comfy_quant" metadata, and pops the metadata from the state dict.

    Returns {layer_prefix: convrot_groupsize}, 0 meaning no rotation. The weight scales are renamed from "weight_scale"
    to "int8_scale" so they don't get mistaken for scaled fp8 scales.
    """
    import json
    int8_layers = {}
    for key in [k for k in state_dict.keys() if k.endswith(".comfy_quant")]:
        prefix = key[:-len("comfy_quant")]
        conf = json.loads(state_dict[key].cpu().numpy().tobytes())
        quant_format = conf.get("format", None)
        if quant_format in ("float8_e4m3fn", "float8_e5m2"):
            continue # handled by the scaled fp8 path
        if quant_format != "int8_tensorwise":
            raise ValueError(f"Unsupported quantization format '{quant_format}' for layer {prefix.rstrip('.')}, supported are fp8 and INT8 (int8_tensorwise, with or without ConvRot)")
        state_dict.pop(key)
        params_conf = conf.get("params", {}) if isinstance(conf.get("params", {}), dict) else {}
        convrot = conf.get("convrot", params_conf.get("convrot", False))
        groupsize = int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))) if convrot else 0
        scale = state_dict.pop(f"{prefix}weight_scale", None)
        if scale is None:
            raise ValueError(f"Missing INT8 weight scale for layer {prefix.rstrip('.')}")
        state_dict[f"{prefix}int8_scale"] = scale.float()
        int8_layers[prefix] = groupsize
    return int8_layers

@torch.library.custom_op("wanvideo::apply_lora", mutates_args=())
def apply_lora(weight: torch.Tensor, lora_diff_0: torch.Tensor, lora_diff_1: torch.Tensor, lora_diff_2: float, lora_strength: torch.Tensor) -> torch.Tensor:
    patch_diff = torch.mm(
        lora_diff_0.flatten(start_dim=1),
        lora_diff_1.flatten(start_dim=1)
    ).reshape(weight.shape)

    alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
    scale = lora_strength * alpha

    return weight + patch_diff * scale

@apply_lora.register_fake
def _(weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
    # Return weight with same metadata
    return weight.clone()

@torch.library.custom_op("wanvideo::apply_single_lora", mutates_args=())
def apply_single_lora(weight: torch.Tensor, lora_diff: torch.Tensor, lora_strength: torch.Tensor) -> torch.Tensor:
    return weight + lora_diff * lora_strength

@apply_single_lora.register_fake
def _(weight, lora_diff, lora_strength):
    # Return weight with same metadata
    return weight.clone()

@torch.library.custom_op("wanvideo::linear_forward", mutates_args=())
def linear_forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return torch.nn.functional.linear(input, weight, bias)

@linear_forward.register_fake
def _(input, weight, bias):
    # Calculate output shape: (..., out_features)
    out_features = weight.shape[0]
    output_shape = list(input.shape[:-1]) + [out_features]
    return input.new_empty(output_shape)

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_linear(model, compute_dtype, state_dict, prefix="", patches=None, scale_weights=None, compile_args=None, modules_to_not_convert=[], int8_layers=None):

    has_children = list(model.children())
    if not has_children:
        return

    allow_compile = False

    for name, module in model.named_children():
        if compile_args is not None:
            allow_compile = compile_args.get("allow_unmerged_lora_compile", False)
        module_prefix = prefix + name + "."
        module_prefix = module_prefix.replace("_orig_mod.", "")
        _replace_linear(module, compute_dtype, state_dict, module_prefix, patches, scale_weights, compile_args, modules_to_not_convert, int8_layers)

        if isinstance(module, nn.Linear) and "loras" not in module_prefix and "dual_controller" not in module_prefix and name not in modules_to_not_convert:
            weight_key = module_prefix + "weight"
            if weight_key not in state_dict:
                continue

            in_features = state_dict[weight_key].shape[1]
            out_features = state_dict[weight_key].shape[0]

            is_gguf = isinstance(state_dict[weight_key], GGUFParameter)

            scale_weight = None
            if not is_gguf and scale_weights is not None:
                scale_key = f"{module_prefix}scale_weight"
                scale_weight = scale_weights.get(scale_key)

            int8_scale_shape = None
            convrot_groupsize = 0
            if int8_layers is not None and module_prefix in int8_layers:
                int8_scale_shape = tuple(state_dict[f"{module_prefix}int8_scale"].shape)
                convrot_groupsize = int8_layers[module_prefix]

            with init_empty_weights():
                model._modules[name] = CustomLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    scale_weight=scale_weight,
                    allow_compile=allow_compile,
                    is_gguf=is_gguf,
                    int8_scale_shape=int8_scale_shape,
                    convrot_groupsize=convrot_groupsize,
                )
            model._modules[name].source_cls = type(module)
            model._modules[name].requires_grad_(False)

    return model

def set_lora_params(module, patches, module_prefix="", device=torch.device("cpu")):
    remove_lora_from_module(module)
    # Recursively set lora_diffs and lora_strengths for all CustomLinear layers
    for name, child in module.named_children():
        params = list(child.parameters())
        if params:
            device = params[0].device
        else:
            device = torch.device("cpu")
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params(child, patches, child_prefix, device)
    if isinstance(module, CustomLinear):
        key = f"diffusion_model.{module_prefix}weight"
        patch = patches.get(key, [])
        #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) == 0:
            key = key.replace("_orig_mod.", "")
            patch = patches.get(key, [])
            #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) != 0:
            lora_diffs = []
            for p in patch:
                lora_obj = p[1]
                if "head" in key:
                    continue  # For now skip LoRA for head layers
                elif hasattr(lora_obj, "weights"):
                    lora_diffs.append(lora_obj.weights)
                elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                    lora_diffs.append(lora_obj[1])
                else:
                    continue
            lora_strengths = [p[0] for p in patch]
            module.set_lora_diffs(lora_diffs, device=device)
            module.set_lora_strengths(lora_strengths, device=device)
            module._step.fill_(0)   # Initialize step for LoRA scheduling


class CustomLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        scale_weight=None,
        allow_compile=False,
        is_gguf=False,
        int8_scale_shape=None,
        convrot_groupsize=0,
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora_diffs = []
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        self.scale_weight = scale_weight
        self.lora_strengths = []
        self.allow_compile = allow_compile
        self.is_gguf = is_gguf

        # INT8: the int8 weight lives in self.weight, its scale is a parameter so that it's loaded and block swapped along with it
        self.is_int8 = int8_scale_shape is not None
        self.convrot_groupsize = convrot_groupsize
        self.int8_matmul = INT8_KITCHEN_AVAILABLE
        if self.is_int8:
            self.int8_scale = nn.Parameter(torch.empty(int8_scale_shape, dtype=torch.float32), requires_grad=False)

        if not allow_compile:
            self._apply_lora_impl = self._apply_lora_custom_op
            self._apply_single_lora_impl = self._apply_single_lora_custom_op
            self._linear_forward_impl = self._linear_forward_custom_op
        else:
            self._apply_lora_impl = self._apply_lora_direct
            self._apply_single_lora_impl = self._apply_single_lora_direct
            self._linear_forward_impl = self._linear_forward_direct


    # Direct implementations (no custom ops)
    def _apply_lora_direct(self, weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        patch_diff = torch.mm(
            lora_diff_0.flatten(start_dim=1),
            lora_diff_1.flatten(start_dim=1)
        ).reshape(weight.shape) + 0
        alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
        scale = lora_strength * alpha
        return weight + patch_diff * scale

    def _apply_single_lora_direct(self, weight, lora_diff, lora_strength):
        return weight + lora_diff * lora_strength

    def _linear_forward_direct(self, input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)

    # Custom op implementations
    def _apply_lora_custom_op(self, weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        return torch.ops.wanvideo.apply_lora(weight, lora_diff_0, lora_diff_1,
            float(lora_diff_2) if lora_diff_2 is not None else 0.0, lora_strength
        )

    def _apply_single_lora_custom_op(self, weight, lora_diff, lora_strength):
        return torch.ops.wanvideo.apply_single_lora(weight, lora_diff, lora_strength)

    def _linear_forward_custom_op(self, input, weight, bias):
        return torch.ops.wanvideo.linear_forward(input, weight, bias)

    def set_lora_diffs(self, lora_diffs, device=torch.device("cpu")):
        self.lora_diffs = []
        for i, diff in enumerate(lora_diffs):
            if len(diff) > 1:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.register_buffer(f"lora_diff_{i}_1", diff[1].to(device, self.compute_dtype))
                setattr(self, f"lora_diff_{i}_2", diff[2])
                self.lora_diffs.append((f"lora_diff_{i}_0", f"lora_diff_{i}_1", f"lora_diff_{i}_2"))
            else:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.lora_diffs.append(f"lora_diff_{i}_0")

    def set_lora_strengths(self, lora_strengths, device=torch.device("cpu")):
        self._lora_strength_tensors = []
        self._lora_strength_is_scheduled = []
        self._step = self._step.to(device)
        for i, strength in enumerate(lora_strengths):
            if isinstance(strength, list):
                tensor = torch.tensor(strength, dtype=self.compute_dtype, device=device)
                self.register_buffer(f"_lora_strength_{i}", tensor)
                self._lora_strength_is_scheduled.append(True)
            else:
                tensor = torch.tensor([strength], dtype=self.compute_dtype, device=device)
                self.register_buffer(f"_lora_strength_{i}", tensor)
                self._lora_strength_is_scheduled.append(False)

    def _get_lora_strength(self, idx):
        strength_tensor = getattr(self, f"_lora_strength_{idx}")
        if self._lora_strength_is_scheduled[idx]:
            return strength_tensor.index_select(0, self._step).squeeze(0)
        return strength_tensor[0]

    def _get_weight_with_lora(self, weight):
        """Apply LoRA using custom ops to avoid graph breaks"""
        if not hasattr(self, "lora_diff_0_0"):
            return weight

        for idx, lora_diff_names in enumerate(self.lora_diffs):
            lora_strength = self._get_lora_strength(idx)

            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_diff_2 = getattr(self, lora_diff_names[2])

                weight = self._apply_lora_impl(
                    weight, lora_diff_0, lora_diff_1,
                    float(lora_diff_2) if lora_diff_2 is not None else 0.0, lora_strength
                )
            else:
                lora_diff = getattr(self, lora_diff_names)
                weight = self._apply_single_lora_impl(weight, lora_diff, lora_strength)
        return weight

    def _prepare_weight(self, input):
        """Prepare weight tensor - handles regular, GGUF and INT8 weights"""
        if self.is_gguf:
            weight = dequantize_gguf_tensor(self.weight).to(self.compute_dtype)
        elif self.is_int8:
            weight = dequantize_int8_weight(self.weight, self.int8_scale, self.convrot_groupsize, dtype=input.dtype)
        else:
            weight = self.weight.to(input)
        return weight

    def forward(self, input):
        # INT8 matmul with dynamic per-token activation quantization, LoRAs need the dequantized weight instead
        if self.is_int8 and self.int8_matmul and input.is_cuda and input.dtype in _INT8_DTYPE_CODES and not hasattr(self, "lora_diff_0_0"):
            bias = self.bias.to(input.dtype) if self.bias is not None else None
            return torch.ops.comfy_kitchen.int8_linear(input.contiguous(), self.weight, self.int8_scale, bias,
                                                       _INT8_DTYPE_CODES[input.dtype], self.convrot_groupsize > 0, self.convrot_groupsize or 256)

        weight = self._prepare_weight(input)

        if self.bias is not None:
            bias = self.bias.to(input if not self.is_gguf else self.compute_dtype)
        else:
            bias = None

        # Only apply scale_weight for non-GGUF models
        if not self.is_gguf and self.scale_weight is not None:
            if weight.numel() < input.numel():
                weight = weight * self.scale_weight
            else:
                input = input * self.scale_weight

        weight = self._get_weight_with_lora(weight)
        out = self._linear_forward_impl(input, weight, bias)
        del weight, input, bias
        return out

def update_lora_step(module, step):
    for name, submodule in module.named_modules():
        if isinstance(submodule, CustomLinear) and hasattr(submodule, "_step"):
            submodule._step.fill_(step)

def remove_lora_from_module(module):
    for name, submodule in module.named_modules():
        if hasattr(submodule, "lora_diffs"):
            for i in range(len(submodule.lora_diffs)):
                if hasattr(submodule, f"lora_diff_{i}_0"):
                    delattr(submodule, f"lora_diff_{i}_0")
                if hasattr(submodule, f"lora_diff_{i}_1"):
                    delattr(submodule, f"lora_diff_{i}_1")
                if hasattr(submodule, f"lora_diff_{i}_2"):
                    delattr(submodule, f"lora_diff_{i}_2")
