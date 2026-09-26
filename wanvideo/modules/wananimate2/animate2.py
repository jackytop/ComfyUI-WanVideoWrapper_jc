# Wan-Animate-2: https://github.com/Wan-Video/Wan-Animate-2
# Wan2.1-I2V-14B shaped model driven directly by the pose (driving) video instead of motion extractors.
# A pose branch runs the pose video's latents through the blocks at a fixed timestep and hands every block its K/V:
# generation frame j attends every generation token plus pose frame j-1, frame 0 is the reference image slot.
import torch
import comfy.model_management as mm

from ....utils import log
from ..attention import attention
from ....custom_linear import _convrot_hadamard

_lse_fallback_warned = set()

def attention_lse(q, k, v, attention_mode="sdpa"):
    """Attention that also returns the logsumexp of the scaled scores.
    q/k/v: [B, L, H, D] -> out [B, Lq, H, D], lse [B, H, Lq] (fp32, natural log)
    """
    if "sage" in attention_mode:
        try:
            from sageattention import sageattn
            dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
            out, lse = sageattn(q.to(dtype), k.to(dtype), v.to(dtype), tensor_layout="NHD", return_lse=True)
            return out.to(q.dtype), lse.float()
        except Exception as e:
            if "sage" not in _lse_fallback_warned:
                log.warning(f"Wan-Animate-2: sageattn with logsumexp failed ({e}), falling back to sdpa for the biased attention")
                _lse_fallback_warned.add("sage")
    elif "flash" in attention_mode:
        try:
            from flash_attn import flash_attn_func
            dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            out, lse, _ = flash_attn_func(q.to(dtype), k.to(dtype), v.to(dtype), return_attn_probs=True)
            return out.to(q.dtype), lse.float()
        except Exception as e:
            if "flash" not in _lse_fallback_warned:
                log.warning(f"Wan-Animate-2: flash_attn with logsumexp failed ({e}), falling back to sdpa for the biased attention")
                _lse_fallback_warned.add("flash")
    # memory efficient sdpa kernel, available in every CUDA build (the flash one isn't on Windows)
    out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
        q.transpose(1, 2), k.to(q.dtype).transpose(1, 2), v.to(q.dtype).transpose(1, 2), None, True)[:2]
    return out.transpose(1, 2), lse[..., :q.shape[1]].float()


def animate2_attention(q, k, v, k_pose, v_pose, frames, hw, attention_mode="sdpa", log_scale=0.0, heads=None, ref_frames=0):
    """Generation branch self-attention.

    q, k, v: generation tokens [B, L, H, D], RoPE applied, tokens past frames * hw are padding
    k_pose, v_pose: pose branch tokens [B or 1, pose_frames * hw, H, D], RoPE applied
    log_scale: logit bias on the keys of the first video frame (generation frame 1), the distilled model uses -1.3 upstream
    ref_frames: extra reference frames after the reference slot, like it they have no pose frame, the video starts after them
    """
    B, L, H, D = q.shape
    valid = frames * hw
    first = 1 + ref_frames # first video frame, driven by pose frame 0
    pose_frames = max(0, min(k_pose.shape[1] // hw, frames - first))
    if k_pose.shape[0] != B:
        k_pose, v_pose = k_pose.expand(B, -1, -1, -1), v_pose.expand(B, -1, -1, -1)
    k_gen, v_gen = k[:, :valid], v[:, :valid]

    def attn(q_, k_, v_):  # "comfy" mode returns the heads flattened
        return attention(q_, k_, v_, attention_mode=attention_mode, heads=heads).reshape(q_.shape)

    if log_scale != 0.0 and frames > first:
        out = _animate2_attention_biased(q[:, :valid], k_gen, v_gen, k_pose, v_pose, pose_frames, hw, attention_mode, log_scale, first)
        if L > valid:
            out = torch.cat([out, attn(q[:, valid:], k_gen, v_gen)], dim=1)
        return out

    out = torch.empty_like(q)
    # the reference slot (and the extra references) have no pose frame
    out[:, :first * hw] = attn(q[:, :first * hw], k_gen, v_gen)
    if pose_frames > 0:
        # the generation half is the same for every frame, only the pose frame at the tail is swapped
        kbuf = k.new_empty(B, valid + hw, H, D)
        vbuf = v.new_empty(B, valid + hw, H, D)
        kbuf[:, :valid] = k_gen
        vbuf[:, :valid] = v_gen
        for j in range(pose_frames):
            f = first + j
            kbuf[:, valid:] = k_pose[:, j * hw:(j + 1) * hw]
            vbuf[:, valid:] = v_pose[:, j * hw:(j + 1) * hw]
            out[:, f * hw:(f + 1) * hw] = attn(q[:, f * hw:(f + 1) * hw], kbuf, vbuf)
        del kbuf, vbuf
    # frames past the end of the pose video, and padding
    tail = (first + pose_frames) * hw
    if tail < L:
        out[:, tail:] = attn(q[:, tail:], k_gen, v_gen)
    return out


def _animate2_attention_biased(q, k, v, k_pose, v_pose, pose_frames, hw, attention_mode, log_scale, first=1):
    # a key-group logit bias can't be expressed with the regular kernels, so the attention is split into key groups
    # (frames before the first video frame | the first video frame, biased | later frames | per-frame pose) and merged
    # by logsumexp one group at a time. The groups are views of k/v, so only the running output and one group's output
    # are alive at once.
    B, L, H, D = q.shape

    def weight(w):  # [B, H, L] -> [B, L, H, 1]
        return w.transpose(1, 2).unsqueeze(-1).to(q.dtype)

    def merge(out, lse, out_g, lse_g):  # in place into out, returns the merged lse
        new = torch.logaddexp(lse, lse_g)
        out.mul_(weight(torch.exp(lse - new))).addcmul_(out_g, weight(torch.exp(lse_g - new)))
        return new

    out = lse = None
    for start, end, bias in ((0, first, 0.0), (first, first + 1, log_scale), (first + 1, None, 0.0)):
        k_g, v_g = k[:, start * hw:None if end is None else end * hw], v[:, start * hw:None if end is None else end * hw]
        if k_g.shape[1] == 0:
            continue
        out_g, lse_g = attention_lse(q, k_g, v_g, attention_mode)
        if bias != 0.0:
            lse_g = lse_g + bias
        if out is None:
            out, lse = out_g, lse_g
        else:
            lse = merge(out, lse, out_g, lse_g)
        del out_g, lse_g

    if pose_frames > 0:
        # query frame first + j against pose frame j, batched over the frames
        n = pose_frames
        pose_range = slice(first * hw, (first + n) * hw)
        out_pose, lse_pose = attention_lse(q[:, pose_range].reshape(B * n, hw, H, D),
                                           k_pose[:, :n * hw].reshape(B * n, hw, H, D),
                                           v_pose[:, :n * hw].reshape(B * n, hw, H, D), attention_mode)
        out_pose = out_pose.reshape(B, n * hw, H, D)
        lse_pose = lse_pose.reshape(B, n, H, hw).transpose(1, 2).reshape(B, H, n * hw)
        merge(out[:, pose_range], lse[..., pose_range], out_pose, lse_pose)
    return out


def _lse_merge(out, lse, out_g, lse_g):
    """Merges a key group's attention into out in place by logsumexp, returns the merged lse ([B, H, L])"""
    new = torch.logaddexp(lse, lse_g)
    out.mul_(torch.exp(lse - new).transpose(1, 2).unsqueeze(-1).to(out.dtype))
    out.addcmul_(out_g, torch.exp(lse_g - new).transpose(1, 2).unsqueeze(-1).to(out.dtype))
    return new


@torch.compiler.disable
def animate2_attention_mem_eff(qkv, k_pose, v_pose, frames, hw, log_scale=0.0, ref_frames=0):
    """animate2_attention with the memory efficient SageAttention: qkv is a [q, k, v] list holding the only references.
    Every key group (and the pose frames) is quantized up front while the bf16 q/k/v are dropped one by one, then the
    groups run on the quantized tensors and are merged by logsumexp. Quantization, kernels and results are the ones
    sageattn gives for the same key groups."""
    from .. import sage_mem_eff as me
    q, k, v = qkv
    qkv.clear()
    B, L, H, D = q.shape
    valid = frames * hw
    first = 1 + ref_frames
    if L != valid or not me.supported(q):
        return animate2_attention(q, k, v, k_pose, v_pose, frames, hw, attention_mode="sageattn", log_scale=log_scale, heads=H, ref_frames=ref_frames)
    k, v = k.to(q.dtype), v.to(q.dtype)
    cfg = me.config(q.device)
    sm_scale = D ** -0.5
    dtype, device = q.dtype, q.device
    pose_frames = max(0, min(k_pose.shape[1] // hw, frames - first))
    if k_pose.shape[0] != B:
        k_pose, v_pose = k_pose.expand(B, -1, -1, -1), v_pose.expand(B, -1, -1, -1)

    # generation key groups: the first video frame on its own when it's biased
    if log_scale != 0.0 and frames > first:
        bounds = [(0, first, 0.0), (first, first + 1, log_scale), (first + 1, frames, 0.0)]
    else:
        bounds = [(0, frames, 0.0)]
    groups = []
    for a, b, bias in bounds:
        if b <= a:
            continue
        k_int8, k_scale, km = me.quant_k(k[:, a * hw:b * hw], cfg)
        groups.append({"k": (k_int8, k_scale), "corr": me.lse_correction(q, km, sm_scale) + bias, "range": (a, b)})
        del km
    del k
    for g in groups:
        a, b = g["range"]
        g["v"] = me.quant_v(v[:, a * hw:b * hw], cfg)
    del v

    pose = None
    if pose_frames > 0:
        # query frame first + j against pose frame j, batched over the frames
        n = pose_frames
        q_p = q[:, first * hw:(first + n) * hw].reshape(B * n, hw, H, D)
        k_p = k_pose[:, :n * hw].reshape(B * n, hw, H, D).to(dtype)
        kp_int8, kp_scale, kpm = me.quant_k(k_p, cfg)
        pose = {"k": (kp_int8, kp_scale), "corr": me.lse_correction(q_p, kpm, sm_scale), "q": me.quant_q(q_p, cfg),
                "v": me.quant_v(v_pose[:, :n * hw].reshape(B * n, hw, H, D).to(dtype), cfg), "n": n}
        del q_p, k_p, kpm
    q_int8, q_scale = me.quant_q(q, cfg)
    del q

    out = lse = None
    for g in groups:
        o = torch.empty((B, L, H, D), dtype=dtype, device=device)
        lse_g = me.kernel(q_int8, q_scale, *g["k"], *g["v"], o, cfg, sm_scale, return_lse=True) + g["corr"]
        g.clear()
        if out is None:
            out, lse = o, lse_g
        else:
            lse = _lse_merge(out, lse, o, lse_g)
        del o, lse_g
    del q_int8, q_scale, groups

    if pose is not None:
        n = pose["n"]
        o = torch.empty((B * n, hw, H, D), dtype=dtype, device=device)
        lse_p = me.kernel(*pose["q"], *pose["k"], *pose["v"], o, cfg, sm_scale, return_lse=True) + pose["corr"]
        pose_range = slice(first * hw, (first + n) * hw)
        _lse_merge(out[:, pose_range], lse[..., pose_range], o.reshape(B, n * hw, H, D),
                   lse_p.reshape(B, n, H, hw).transpose(1, 2).reshape(B, H, n * hw))
    return out


class Animate2PoseCache:
    """Pose branch block inputs, reused across the sampling steps and the cond/uncond passes.

    The pose branch depends on neither the generation latents nor the timestep, so its block inputs only need to be
    computed once per pose sequence. K/V are re-projected from the cached inputs, which is half the memory of caching
    them directly for a small cost. One slot per pose sequence (context windows get their own), least recently used
    slots are evicted when the store device runs low on memory.
    """
    INT8_GROUPSIZE = 256
    MAX_PINNED_FRACTION = 0.25  # of system RAM, pinning too much starves the OS

    def __init__(self, store_device="cpu", dtype="default"):
        self.store_device = torch.device(store_device)
        self.dtype = dtype
        self.slots = []  # most recently used last
        self.slot = None
        self._warned = False
        self._pinned_bytes = 0
        try:
            import psutil
            self._max_pinned_bytes = psutil.virtual_memory().total * self.MAX_PINNED_FRACTION
        except Exception:
            self._max_pinned_bytes = 0

    def select(self, key):
        for i, s in enumerate(self.slots):
            if s["key"].shape == key.shape and torch.equal(s["key"], key.to(s["key"].device)):
                self.slots.append(self.slots.pop(i))  # by index, list.remove would compare the key tensors
                self.slot = s
                return True
        self.slot = None
        return False

    def filled(self, num_blocks):
        return self.slot is not None and len(self.slot["blocks"]) == num_blocks

    def create(self, key, slot_bytes):
        # the cache is an optimization, skip it rather than run the store device out of memory
        slot_bytes = slot_bytes // 2 if self.dtype == "int8" else slot_bytes
        while self.slots and mm.get_free_memory(self.store_device) < slot_bytes * 1.2:
            self._free_slot(self.slots.pop(0))
        if mm.get_free_memory(self.store_device) < slot_bytes * 1.2:
            if not self._warned:
                log.warning(f"Wan-Animate-2: not enough free memory on {self.store_device} to cache the pose branch "
                            f"({slot_bytes / 1024**3:.1f} GB needed), it will be recomputed on every model call")
                self._warned = True
            self.slot = None
            return False
        self.slot = {"key": key.clone().to(self.store_device), "blocks": {}}
        self.slots.append(self.slot)
        return True

    def put(self, i, x):
        if self.slot is None:
            return
        if self.dtype == "int8":
            # per-token int8 on Hadamard rotated activations, the rotation spreads the large per-channel outliers
            shape, g = x.shape, self.INT8_GROUPSIZE
            while g > 4 and shape[-1] % g:
                g //= 4
            h = _convrot_hadamard(g, x.device, torch.float32)
            x_rot = torch.matmul(x.float().reshape(-1, shape[-1] // g, g), h).reshape(-1, shape[-1])
            scale = (x_rot.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
            q = (x_rot / scale).round_().clamp_(-128, 127).to(torch.int8)
            self.slot["blocks"][i] = (self._store(q), scale.to(self.store_device), shape, g)
        else:
            self.slot["blocks"][i] = (self._store(x), None, x.shape, 0)

    def _store(self, t):
        # pinned memory makes the per-step reads faster, up to a budget
        nbytes = t.numel() * t.element_size()
        if self.store_device.type == "cpu" and t.device.type != "cpu" and self._pinned_bytes + nbytes <= self._max_pinned_bytes:
            try:
                out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                out.copy_(t)
                self._pinned_bytes += nbytes
                return out
            except Exception:
                pass
        return t.to(self.store_device, copy=True)

    def take(self, i, device, dtype):
        q, scale, shape, g = self.slot["blocks"][i]
        x = q.to(device, non_blocking=True)
        if scale is None:
            return x.to(dtype)
        h = _convrot_hadamard(g, device, torch.float32)
        x = (x.float() * scale.to(device)).reshape(-1, shape[-1] // g, g)
        return torch.matmul(x, h.T).reshape(shape).to(dtype)

    def _free_slot(self, s):
        for q, scale, shape, g in s["blocks"].values():
            if q.is_pinned():
                self._pinned_bytes -= q.numel() * q.element_size()
        s["blocks"].clear()

    def free(self):
        for s in self.slots:
            self._free_slot(s)
        self.slots = []
        self.slot = None
