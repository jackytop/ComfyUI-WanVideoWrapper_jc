from comfy import model_management as mm

from ..nodes import WanVideoClipVisionEncode, WanVideoDecode
from ..nodes_sampler import WanVideoSampler
from .sampling import encode_images, resize_images


class WanVideoEverAnimateEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("WANVAE",),
                "clip_vision": ("CLIP_VISION",),
                "reference_image": ("IMAGE",),
                "pose_images": ("IMAGE",),
                "face_images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 16}),
                "frames_per_chunk": ("INT", {"default": 77, "min": 5, "max": 10001, "step": 4}),
                "num_chunks": ("INT", {"default": 2, "min": 1, "max": 200, "tooltip": "Automatically generate and join this many chunks in WanVideo Sampler. Two 77-frame chunks produce 150 frames."}),
            },
            "optional": {
                "loop_mode": (["pingpong", "loop", "hold"], {"default": "pingpong"}),
                "anchor_mode": (["random_plus_user", "user_image"], {"default": "random_plus_user", "tooltip": "Keep three frames from the first generated chunk plus the user reference as four persistent anchors."}),
                "seed_multiplier": ("INT", {"default": 42, "min": 1, "max": 10000, "tooltip": "Add this to the sampler seed for each new chunk. The sampler seed also controls anchor selection."}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use VAE tiling for reference encoding and all chunk encoding/decoding inside WanVideo Sampler."}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Pose conditioning strength for every chunk. 1.0 matches the original behavior; 0 disables the pose adapter contribution."}),
                "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Face conditioning strength for every chunk. 1.0 matches the original behavior; 0 disables the face adapter contribution."}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/EverAnimate"
    DESCRIPTION = "EverAnimate conditioning for Wan2.2-Animate-14B + EverAnimate LoRA. Connect to WanVideo Sampler (jc), then WanVideo Decode (jc). The sampler handles all chunks, persistent anchors, motion memory and CLIP Vision refresh. Recommended: Euler, 20 steps, CFG 1, shift 5."

    def process(self, vae, clip_vision, reference_image, pose_images, face_images, width, height,
                frames_per_chunk, num_chunks, loop_mode="pingpong", anchor_mode="random_plus_user",
                seed_multiplier=42, tiled_vae=False, pose_strength=1.0, face_strength=1.0):
        if vae.upsampling_factor != 8:
            raise ValueError("EverAnimate needs the Wan2.1 VAE (16 latent channels, spatial scale 8).")
        width, height = width // 16 * 16, height // 16 * 16
        frames_per_chunk = max(5, (frames_per_chunk - 1) // 4 * 4 + 1)
        reference = resize_images(reference_image[:1], width, height)
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        try:
            vae.to(device)
            reference_latent = encode_images(vae, reference, device, tiled_vae)
        finally:
            vae.to(offload_device)
        clip_embeds = WanVideoClipVisionEncode().process(
            clip_vision, reference, strength_1=1.0, strength_2=1.0,
            force_offload=True, crop="center", combine_embeds="average")[0]
        return ({"everanimate": {
            "vae": vae, "clip_vision": clip_vision, "reference_latent": reference_latent,
            "clip_embeds": clip_embeds, "pose_images": pose_images, "face_images": face_images,
            "width": width, "height": height, "frames_per_chunk": frames_per_chunk, "num_chunks": num_chunks,
            "loop_mode": loop_mode, "anchor_mode": anchor_mode,
            "seed_multiplier": seed_multiplier, "tiled_vae": tiled_vae,
            "pose_strength": pose_strength, "face_strength": face_strength,
        }},)


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
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Pose conditioning strength for every chunk. 1.0 matches the original behavior; 0 disables the pose adapter contribution."}),
                "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Face conditioning strength for every chunk. 1.0 matches the original behavior; 0 disables the face adapter contribution."}),
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
                anchor_mode="random_plus_user", cache_args=None, feta_args=None, sigmas=None,
                pose_strength=1.0, face_strength=1.0):
        embeds = WanVideoEverAnimateEmbeds().process(
            vae, clip_vision, reference_image, pose_images, face_images, width, height,
            frames_per_chunk, num_chunks, loop_mode, anchor_mode, seed_multiplier, tiled_vae,
            pose_strength=pose_strength, face_strength=face_strength)[0]
        result, _ = WanVideoSampler().process(
            model=model, image_embeds=embeds, text_embeds=text_embeds, shift=shift, steps=steps, cfg=cfg,
            seed=seed, scheduler=scheduler, riflex_freq_index=0, force_offload=force_offload,
            rope_function="comfy_chunked", cache_args=cache_args, feta_args=feta_args, sigmas=sigmas)
        images = WanVideoDecode().decode(vae, result, tiled_vae, 272, 272, 144, 128)[0]
        return images, {"samples": result["samples"]}


NODE_CLASS_MAPPINGS = {
    "WanVideoEverAnimateEmbeds": WanVideoEverAnimateEmbeds,
    "WanVideoEverAnimateSampler": WanVideoEverAnimateSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoEverAnimateEmbeds": "WanVideo EverAnimate Embeds",
    "WanVideoEverAnimateSampler": "WanVideo EverAnimate Sampler",
}
