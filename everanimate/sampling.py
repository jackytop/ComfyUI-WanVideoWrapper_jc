import numpy as np
import torch

from comfy import model_management as mm
from comfy.utils import common_upscale

from ..nodes import WanVideoClipVisionEncode, WanVideoDecode
from ..utils import log
from ..wanvideo.schedulers import get_scheduler


def take_frames(images, start, count, loop_mode):
    length = images.shape[0]
    if length == 0:
        raise ValueError("EverAnimate needs non-empty pose and face images.")
    indices = list(range(start, start + count))
    if loop_mode == "pingpong" and length > 1:
        period = 2 * length - 2
        indices = [min(i % period, period - i % period) for i in indices]
    elif loop_mode == "loop":
        indices = [i % length for i in indices]
    else:
        indices = [min(i, length - 1) for i in indices]
    return images[indices]


def resize_images(images, width, height, crop="disabled"):
    images = images[..., :3]
    if images.shape[1:3] != (height, width):
        images = common_upscale(images.movedim(-1, 1), width, height, "lanczos", crop).movedim(1, -1)
    return images


def encode_images(vae, images, device, tiled):
    pixels = images.permute(3, 0, 1, 2).to(device=device, dtype=vae.dtype) * 2 - 1
    return vae.encode([pixels], device=device, tiled=tiled)[0].cpu()


def make_chunk_embeds(anchors, motion, pose_latents, face_images, clip_embeds, frames_per_chunk):
    # WANVAE returns model-space latents. Empty conditioning is zero in that space.
    channels, anchor_count, lat_h, lat_w = anchors.shape
    content_length = (frames_per_chunk - 1) // 4 + 1
    condition = anchors.new_zeros(channels + 4, anchor_count + content_length, lat_h, lat_w)
    condition[:4, :anchor_count] = 1
    condition[4:, :anchor_count] = anchors
    if motion is not None:
        condition[4:, anchor_count:anchor_count + 1] = motion
    face_pixels = (face_images.permute(3, 0, 1, 2).unsqueeze(0) * 2 - 1).to(dtype=anchors.dtype)
    return {
        "ref_latent": condition,
        "target_shape": (channels, anchor_count + content_length, lat_h, lat_w),
        "num_frames": frames_per_chunk + 4 * anchor_count,
        "lat_h": lat_h,
        "lat_w": lat_w,
        "clip_context": clip_embeds["clip_embeds"],
        "negative_clip_context": clip_embeds.get("negative_clip_embeds"),
        "pose_latents": pose_latents.unsqueeze(0),
        "face_pixels": face_pixels,
        "wananim_num_anchor_latents": anchor_count,
        "ref_trim": (0, anchor_count - 1),
    }


def make_scheduler(steps, shift, device, scheduler_name="euler", sigmas=None, start_step=0, end_step=-1):
    if sigmas is None:
        sigmas = torch.linspace(1, 0, steps + 1)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    else:
        sigmas = sigmas.detach().clone()
        steps = len(sigmas) - 1
    scheduler, _, _, _ = get_scheduler(scheduler_name, steps, start_step, end_step, shift, device, sigmas=sigmas)
    # Preserve the fractional timesteps in the reference FlowMatch scheduler.
    scheduler.timesteps = scheduler.sigmas[:-1] * 1000
    return {"sample_scheduler": scheduler, "timesteps": scheduler.timesteps, "start_step": start_step}


def sample_everanimate(sample, model, embeds, steps, shift, seed, scheduler, sigmas=None, **sampling_options):
    if not hasattr(model.model.diffusion_model, "pose_patch_embedding"):
        raise ValueError("EverAnimate needs a Wan2.2-Animate-14B model loaded with the jc WanVideo Model Loader.")
    vae = embeds["vae"]
    clip_vision = embeds["clip_vision"]
    tiled_vae = embeds["tiled_vae"]
    width, height = embeds["width"], embeds["height"]
    frames_per_chunk, num_chunks = embeds["frames_per_chunk"], embeds["num_chunks"]
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    if isinstance(scheduler, str) and scheduler in ("euler", "unipc", "dpm++"):
        # Match EverAnimate's schedule; explicit scheduler objects retain their own schedule.
        start_step = sampling_options.get("start_step", 0)
        denoise_strength = sampling_options.get("denoise_strength", 1.0)
        if denoise_strength < 1.0:
            if start_step != 0:
                raise ValueError("start_step must be 0 when denoise_strength is used")
            start_step = steps - int(steps * denoise_strength) - 1
        scheduler = make_scheduler(steps, shift, device, scheduler, sigmas,
                                   start_step, sampling_options.get("end_step", -1))
    rng = np.random.default_rng(seed)
    reference_latent = embeds["reference_latent"]
    anchors = reference_latent.repeat(1, 4, 1, 1)
    clip_embeds = embeds["clip_embeds"]
    clip_encoder = WanVideoClipVisionEncode()
    decoder = WanVideoDecode()
    all_frames = []
    motion = None

    try:
        for chunk_index in range(num_chunks):
            mm.throw_exception_if_processing_interrupted()
            log.info(f"EverAnimate: chunk {chunk_index + 1}/{num_chunks}")
            offset = chunk_index * (frames_per_chunk - 4)
            poses = resize_images(take_frames(embeds["pose_images"], offset, frames_per_chunk, embeds["loop_mode"]), width, height)
            faces = resize_images(take_frames(embeds["face_images"], offset, frames_per_chunk, embeds["loop_mode"]), 512, 512, "center")
            vae.to(device)
            pose_latents = encode_images(vae, poses, device, tiled_vae)
            vae.to(offload_device)
            if chunk_index > 0:
                clip_embeds = clip_encoder.process(clip_vision, current_image, strength_1=1.0, strength_2=1.0,
                                                  force_offload=True, crop="center", combine_embeds="average")[0]
            chunk_embeds = make_chunk_embeds(anchors, motion, pose_latents, faces, clip_embeds, frames_per_chunk)
            chunk_seed = (seed + chunk_index * embeds["seed_multiplier"]) % (1 << 64)
            sampled, denoised = sample(model=model, image_embeds=chunk_embeds, shift=shift, steps=steps,
                                       seed=chunk_seed, scheduler=scheduler, sigmas=sigmas, **sampling_options)
            content = sampled["samples"][:, :, 4:].clone()
            # The next chunk consumes the sampled latent directly, without an RGB round trip.
            motion = content[0, :, -1:].clone()
            frames = decoder.decode(vae, {"samples": content}, enable_vae_tiling=tiled_vae,
                                    tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128)[0]
            if chunk_index == 0 and num_chunks > 1 and embeds["anchor_mode"] == "random_plus_user":
                selected = sorted(rng.choice(np.arange(1, frames.shape[0]), size=3, replace=False).tolist())
                vae.to(device)
                anchors = torch.cat([encode_images(vae, frames[i:i + 1], device, tiled_vae) for i in selected]
                                    + [reference_latent], dim=1)
                vae.to(offload_device)
                log.info(f"EverAnimate: persistent anchors from frames {selected} and the reference image")
            all_frames.append(frames if chunk_index == 0 else frames[4:].clone())
            current_image = frames[-1:].clone()
            del chunk_embeds, sampled, pose_latents, poses, faces
    finally:
        vae.to(offload_device)
        clip_vision.model.to(offload_device)
        mm.soft_empty_cache()

    # WanVideo Decode's loop output contract uses NHWC pixels in [-1, 1].
    video = torch.cat(all_frames, dim=0).mul_(2).sub_(1)
    return {"video": video, "samples": content}, denoised
