import numpy as np
import torch

from comfy import model_management as mm
from comfy.utils import common_upscale

from ..nodes import WanVideoClipVisionEncode, WanVideoDecode
from ..nodes_sampler import WanVideoSampler
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


def make_scheduler(steps, shift, device, scheduler_name="euler", sigmas=None):
    if sigmas is None:
        sigmas = torch.linspace(1, 0, steps + 1)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    else:
        sigmas = sigmas.detach().clone()
        steps = len(sigmas) - 1
    scheduler, _, _, _ = get_scheduler(scheduler_name, steps, 0, -1, shift, device, sigmas=sigmas)
    # Preserve the fractional timesteps in the reference FlowMatch scheduler.
    scheduler.timesteps = scheduler.sigmas[:-1] * 1000
    return {"sample_scheduler": scheduler, "timesteps": scheduler.timesteps}


class WanVideoEverAnimateSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "vae": ("WANVAE",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "clip_vision": ("CLIP_VISION",),
                "reference_image": ("IMAGE",),
                "pose_images": ("IMAGE",),
                "face_images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 16}),
                "frames_per_chunk": ("INT", {"default": 77, "min": 5, "max": 10001, "step": 4}),
                "num_chunks": ("INT", {"default": 2, "min": 1, "max": 200, "tooltip": "77 frames per chunk with 4 overlapping frames gives 150 frames for two chunks."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload the diffusion model between chunks to leave room for VAE and CLIP Vision."}),
                "tiled_vae": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "scheduler": (["euler", "unipc", "dpm++"], {"default": "euler", "tooltip": "Euler with the EverAnimate sigma schedule matches the reference solver. Other solvers are experimental."}),
                "seed_multiplier": ("INT", {"default": 42, "min": 1, "max": 10000}),
                "loop_mode": (["pingpong", "loop", "hold"], {"default": "pingpong"}),
                "anchor_mode": (["random_plus_user", "user_image"], {"default": "random_plus_user", "tooltip": "Keep three frames from the first generated chunk plus the user reference as four persistent anchors."}),
                "cache_args": ("CACHEARGS",),
                "feta_args": ("FETAARGS",),
                "sigmas": ("SIGMAS", {"tooltip": "Optional complete sigma schedule including the final zero; overrides steps and shift."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "last_chunk_latent")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/EverAnimate"
    DESCRIPTION = "EverAnimate for Wan2.2-Animate-14B + EverAnimate LoRA. Keeps four anchors, propagates the last clean latent and refreshes CLIP Vision each chunk. Includes decoding and removes four overlap frames from subsequent chunks."

    def process(self, model, vae, text_embeds, clip_vision, reference_image, pose_images, face_images,
                width, height, frames_per_chunk, num_chunks, steps, cfg, shift, seed, force_offload,
                tiled_vae, scheduler="euler", seed_multiplier=42, loop_mode="pingpong",
                anchor_mode="random_plus_user", cache_args=None, feta_args=None, sigmas=None):
        if not hasattr(model.model.diffusion_model, "pose_patch_embedding"):
            raise ValueError("EverAnimate needs a Wan2.2-Animate-14B model loaded with the jc WanVideo Model Loader.")
        if vae.upsampling_factor != 8:
            raise ValueError("EverAnimate needs the Wan2.1 VAE (16 latent channels, spatial scale 8).")
        width, height = width // 16 * 16, height // 16 * 16
        frames_per_chunk = max(5, (frames_per_chunk - 1) // 4 * 4 + 1)
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        reference = resize_images(reference_image[:1], width, height)
        sampler = WanVideoSampler()
        clip_encoder = WanVideoClipVisionEncode()
        decoder = WanVideoDecode()
        schedule = make_scheduler(steps, shift, device, scheduler, sigmas)
        steps = len(schedule["timesteps"])
        rng = np.random.default_rng(seed)
        all_frames = []
        motion = None
        current_image = reference

        try:
            vae.to(device)
            reference_latent = encode_images(vae, reference, device, tiled_vae)
            anchors = reference_latent.repeat(1, 4, 1, 1)
            vae.to(offload_device)

            for chunk_index in range(num_chunks):
                mm.throw_exception_if_processing_interrupted()
                log.info(f"EverAnimate: chunk {chunk_index + 1}/{num_chunks}")
                offset = chunk_index * (frames_per_chunk - 4)
                poses = resize_images(take_frames(pose_images, offset, frames_per_chunk, loop_mode), width, height)
                faces = resize_images(take_frames(face_images, offset, frames_per_chunk, loop_mode), 512, 512, "center")
                vae.to(device)
                pose_latents = encode_images(vae, poses, device, tiled_vae)
                vae.to(offload_device)
                clip_embeds = clip_encoder.process(clip_vision, current_image, strength_1=1.0, strength_2=1.0,
                                                  force_offload=True, crop="center", combine_embeds="average")[0]
                embeds = make_chunk_embeds(anchors, motion, pose_latents, faces, clip_embeds, frames_per_chunk)
                chunk_seed = (seed + chunk_index * seed_multiplier) % (1 << 64)
                sampled, _ = sampler.process(model=model, image_embeds=embeds, text_embeds=text_embeds,
                                             shift=shift, steps=steps, cfg=cfg, seed=chunk_seed, scheduler=schedule,
                                             riflex_freq_index=0, force_offload=force_offload, rope_function="comfy_chunked",
                                             cache_args=cache_args, feta_args=feta_args)
                content = sampled["samples"][:, :, 4:].clone()
                # Copy the one-slot memory so it doesn't retain the entire previous chunk.
                motion = content[0, :, -1:].clone()
                frames = decoder.decode(vae, {"samples": content}, enable_vae_tiling=tiled_vae,
                                        tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128)[0]
                if chunk_index == 0 and num_chunks > 1 and anchor_mode == "random_plus_user":
                    selected = sorted(rng.choice(np.arange(1, frames.shape[0]), size=3, replace=False).tolist())
                    vae.to(device)
                    anchors = torch.cat([encode_images(vae, frames[i:i + 1], device, tiled_vae) for i in selected]
                                        + [reference_latent], dim=1)
                    vae.to(offload_device)
                    log.info(f"EverAnimate: persistent anchors from frames {selected} and the reference image")
                all_frames.append(frames if chunk_index == 0 else frames[4:].clone())
                current_image = frames[-1:].clone()
                del embeds, sampled, pose_latents, poses, faces, clip_embeds
        finally:
            vae.to(offload_device)
            clip_vision.model.to(offload_device)
            mm.soft_empty_cache()

        return torch.cat(all_frames, dim=0), {"samples": content}


NODE_CLASS_MAPPINGS = {"WanVideoEverAnimateSampler": WanVideoEverAnimateSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"WanVideoEverAnimateSampler": "WanVideo EverAnimate Sampler"}
