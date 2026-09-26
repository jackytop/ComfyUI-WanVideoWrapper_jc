# Memory efficient SageAttention (after KJNodes' "WanVideo Mem Eff Sage Attention Patch").
#
# sageattn() quantizes q/k to int8 and v to fp8 (or fp16) while the caller still holds the bf16 q/k/v, so at the kernel
# the original q/k/v, their quantized copies and the output are all alive. Here the caller hands over the only references
# in a list, and each bf16 tensor is freed as soon as its quantized copy exists. Same kernels, quantization and k smoothing
# as sageattn() picks for the GPU, so the results match it.
import torch

from ...utils import log

_warned = set()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg)


def _core():
    import sageattention.core as sc
    return sc


_configs = {}


def config(device):
    """Quantization and kernel sageattn() uses on this GPU, None where it isn't reimplemented here."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _configs:
        _configs[index] = _config(index)
    return _configs[index]


def _config(index):
    sc = _core()
    arch = sc.get_cuda_arch_versions()[index]
    cuda = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
    if arch in ("sm80", "sm86"):
        return {"arch": arch, "gran": "per_thread", "pv": "fp16", "accum": "fp32"}
    if arch == "sm89":
        return {"arch": arch, "gran": "per_thread", "pv": "fp8", "accum": "fp32+fp16" if cuda >= (12, 8) else "fp32+fp32"}
    if arch == "sm120":
        return {"arch": arch, "gran": "per_warp", "pv": "fp8", "accum": "fp32+fp16" if cuda >= (12, 8) else "fp32"}
    return None


def supported(q):
    try:
        ok = q.is_cuda and q.shape[-1] in (64, 128) and q.dtype in (torch.float16, torch.bfloat16) and config(q.device) is not None
    except Exception as e:
        _warn_once("import", f"sageattn_mem_eff: sageattention internals not usable ({e}), using the regular sageattn")
        return False
    if not ok:
        _warn_once("unsupported", "sageattn_mem_eff: not supported for this GPU/head size/dtype, using the regular sageattn")
    return ok


def _quant(q, k, km, cfg):
    sc = _core()
    if cfg["gran"] == "per_warp":
        return sc.per_warp_int8_cuda(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64)
    return sc.per_thread_int8_triton(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)


def quant_q(q, cfg):
    """int8 q [B, L, H, D] -> (q_int8, q_scale). The q and k quantizations are independent, k gets a one token stand-in."""
    q_int8, q_scale, _, _ = _quant(q, q[:, :1], None, cfg)
    return q_int8, q_scale


def quant_k(k, cfg):
    """int8 k [B, L, H, D] smoothed by its mean over the sequence as sageattn does -> (k_int8, k_scale, km)"""
    km = k.mean(dim=1, keepdim=True)
    _, _, k_int8, k_scale = _quant(k[:, :1], k, km, cfg)
    return k_int8, k_scale, km


def quant_v(v, cfg):
    if cfg["pv"] == "fp16":
        return v.to(torch.float16), None
    v_fp8, v_scale, _ = _core().per_channel_fp8(v, tensor_layout="NHD", scale_max=2.25 if cfg["accum"] == "fp32+fp16" else 448.0, smooth_v=False)
    return v_fp8, v_scale


def lse_correction(q, km, sm_scale):
    """The k smoothing shifts every score of a query by q . km, the kernel's logsumexp misses it: [B, H, L] fp32"""
    return torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).float() * sm_scale


def kernel(q_int8, q_scale, k_int8, k_scale, v_q, v_scale, o, cfg, sm_scale, return_lse=False):
    """Attention into o [B, L, H, D], returns the logsumexp (natural log, without the smoothing correction) or None"""
    sc = _core()
    gran = 3 if cfg["gran"] == "per_thread" else 2
    lse_flag = 1 if return_lse else 0
    if cfg["pv"] == "fp16":
        lse = sc._qattn_sm80.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v_q, o, q_scale, k_scale, 0, 0, gran, sm_scale, lse_flag)
    else:
        fn = {"fp32": sc._qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn,
              "fp32+fp32": sc._qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf,
              "fp32+fp16": sc._qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf}[cfg["accum"]]
        lse = fn(q_int8, k_int8, v_q, o, q_scale, k_scale, v_scale, 0, 0, gran, sm_scale, lse_flag)
    return lse[..., :o.shape[1]] / 1.44269504 if return_lse else None


@torch.compiler.disable
def sageattn_mem_eff(qkv):
    """qkv: [q, k, v] list of [B, L, H, D] (NHD) tensors, consumed: pass the only references so they can be freed early.
    Returns the attention output [B, L, H, D] in q's dtype."""
    q, k, v = qkv
    qkv.clear()
    k, v = k.to(q.dtype), v.to(q.dtype)
    if not supported(q):
        from .attention import sageattn_func
        return sageattn_func(q, k, v, tensor_layout="NHD").contiguous()
    cfg = config(q.device)
    shape, dtype, device = q.shape, q.dtype, q.device
    k_int8, k_scale, _ = quant_k(k, cfg)
    del k
    q_int8, q_scale = quant_q(q, cfg)
    del q
    v_q, v_scale = quant_v(v, cfg)
    del v
    o = torch.empty(shape, dtype=dtype, device=device)
    kernel(q_int8, q_scale, k_int8, k_scale, v_q, v_scale, o, cfg, shape[-1] ** -0.5)
    return o
