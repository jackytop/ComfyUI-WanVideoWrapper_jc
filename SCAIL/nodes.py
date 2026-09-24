import math
import torch
import torch.nn.functional as F
from ..utils import log
import comfy.model_management as mm
from comfy.utils import common_upscale

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

# SCAIL-2 was trained on these identity colors, in this order
SCAIL2_PALETTE = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0)]

def scail2_mask_to_latent(mask_rgb):
    """Colored mask frames [T, 3, H, W] in [0, 1] -> SCAIL-2's 28 channel binary latent [28, T_lat, H/8, W/8].

    7 binary color channels (white, red, green, blue, yellow, magenta, cyan) thresholded at 225/255, area downsampled 8x,
    and 4 frames stacked per latent frame with the first frame repeated, like the VAE's temporal compression.
    """
    R, G, B = [(mask_rgb[:, i:i + 1].float() > 225.0 / 255.0).float() for i in range(3)]
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    binary = torch.cat([R * G * B, R * nG * nB, nR * G * nB, nR * nG * B, R * G * nB, R * nG * B, nR * G * B], dim=1)
    T, _, H, W = binary.shape
    for _ in range(3):
        H, W = (H + 1) // 2, (W + 1) // 2
    binary = F.interpolate(binary, size=(H, W), mode="area")
    padded = torch.cat([binary[:1].repeat(4, 1, 1, 1), binary[1:]], dim=0)
    return padded.view((T - 1) // 4 + 1, 28, H, W).permute(1, 0, 2, 3)

def scail2_resize(images, width, height):
    """IMAGE [B, H, W, C] -> [B, 3, height, width] in [0, 1], scaled to cover and center cropped like upstream"""
    return common_upscale(images[..., :3].movedim(-1, 1), width, height, "bicubic", "center").clamp(0, 1)

def scail2_half(images):
    return F.interpolate(images, scale_factor=0.5, mode="bilinear", align_corners=False)

class WanVideoSCAIL2Embeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "width": ("INT", {"default": 512, "min": 64, "max": 8096, "step": 32, "tooltip": "Output width, divisible by 32"}),
                    "height": ("INT", {"default": 896, "min": 64, "max": 8096, "step": 32, "tooltip": "Output height, divisible by 32"}),
                    "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to generate, the pose video and its mask are trimmed or ping-pong padded to this length"}),
                    "replacement_mode": ("BOOLEAN", {"default": False, "tooltip": "False: Animation Mode, animates the reference character with the driving video. True: Replacement Mode, replaces the masked character in the driving video with the reference"}),
                    "segment_len": ("INT", {"default": 81, "min": 5, "max": 10000, "step": 4, "tooltip": "Frames per generation segment. Longer videos are generated segment after segment, each continuing from the end of the previous one, as upstream does"}),
                    "segment_overlap": ("INT", {"default": 5, "min": 1, "max": 1000, "step": 4, "tooltip": "Frames of the previous segment used as clean history for the next one, upstream uses 5"}),
                    "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scales the pose tokens"}),
                    "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "force_offload": ("BOOLEAN", {"default": True}),
                },
                "optional": {
                    "ref_image": ("IMAGE", {"tooltip": "Reference image. Extra images in the batch are used as additional references (another view, a close-up, a clean background), each with its own mask"}),
                    "ref_mask": ("IMAGE", {"tooltip": "Colored reference mask, one per ref_image. Animation Mode: white background with the character in its identity color. Replacement Mode: black background"}),
                    "pose_video": ("IMAGE", {"tooltip": "Driving video: the source video itself (end-to-end) or a rendered pose video"}),
                    "pose_video_mask": ("IMAGE", {"tooltip": "Colored per-frame mask of the driving video. Animation Mode: black background with each character in its identity color. Replacement Mode: white background"}),
                    "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded reference image"}),
                    "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Embeds for SCAIL-2 (https://github.com/zai-org/SCAIL-2), use with a SCAIL-2 model and the regular WanVideo Sampler. The masks are colored: see WanVideo SCAIL2 Colored Mask"

    def process(self, vae, width, height, num_frames, replacement_mode, segment_len, segment_overlap, pose_strength, pose_start_percent, pose_end_percent,
                force_offload, ref_image=None, ref_mask=None, pose_video=None, pose_video_mask=None, clip_embeds=None, tiled_vae=False):
        from ..utils import tensor_pingpong_pad
        if segment_overlap >= segment_len:
            raise ValueError(f"segment_overlap ({segment_overlap}) must be smaller than segment_len ({segment_len})")
        W, H = (width // 32) * 32, (height // 32) * 32
        lat_h, lat_w = H // vae.upsampling_factor, W // vae.upsampling_factor
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        segment_len = ((segment_len - 1) // 4) * 4 + 1
        looping = num_frames > segment_len

        mm.soft_empty_cache()
        vae.to(device)

        # references: the additional ones first, the main reference last, each encoded as its own single frame
        refs = scail2_resize(ref_image, W, H) if ref_image is not None else torch.zeros(1, 3, H, W)
        refs = torch.cat([refs[1:], refs[:1]])
        ref_latent = torch.cat([vae.encode([(r.unsqueeze(1) * 2 - 1).to(device, vae.dtype)], device, tiled=tiled_vae)[0] for r in refs], dim=1).to(offload_device)
        ref_latent = torch.cat([ref_latent, torch.ones_like(ref_latent[:4])])

        if ref_mask is None:
            log.warning(f"SCAIL-2: no ref_mask, using a plain {'black' if replacement_mode else 'white'} one. The masks matter, without them animation tends to behave like replacement")
            ref_masks = torch.full((1, 3, H, W), 0.0 if replacement_mode else 1.0)
        else:
            ref_masks = scail2_resize(ref_mask, W, H)
        ref_masks = ref_masks[[min(i, ref_masks.shape[0] - 1) for i in range(refs.shape[0])]]
        ref_masks = torch.cat([ref_masks[1:], ref_masks[:1]])
        ref_mask_latent = torch.cat([scail2_mask_to_latent(m[None]) for m in ref_masks], dim=1)

        # driving video and its mask, at half resolution like upstream
        pose_half = mask_half = None
        if pose_video is not None:
            pose_half = scail2_half(scail2_resize(pose_video, W, H))
            if pose_video_mask is None:
                log.warning(f"SCAIL-2: no pose_video_mask, using a plain {'white' if replacement_mode else 'black'} one")
                mask_half = torch.full_like(pose_half[:1], 1.0 if replacement_mode else 0.0).repeat(pose_half.shape[0], 1, 1, 1)
            else:
                mask_half = scail2_half(scail2_resize(pose_video_mask, W, H))
                if mask_half.shape[0] < pose_half.shape[0]:
                    mask_half = torch.cat([mask_half, mask_half[-1:].repeat(pose_half.shape[0] - mask_half.shape[0], 1, 1, 1)])
                mask_half = mask_half[:pose_half.shape[0]]
            if pose_half.shape[0] < num_frames:
                log.info(f"SCAIL-2: ping-pong padding the pose video from {pose_half.shape[0]} to {num_frames} frames")
                pose_half = tensor_pingpong_pad(pose_half.movedim(0, 1), num_frames).movedim(1, 0)
                mask_half = tensor_pingpong_pad(mask_half.movedim(0, 1), num_frames).movedim(1, 0)
            pose_half, mask_half = pose_half[:num_frames], mask_half[:num_frames]

        scail2 = {
            "ref_latent": ref_latent,
            "ref_mask": ref_mask_latent,
            "replace": replacement_mode,
            "pose_strength": pose_strength,
            "start_percent": pose_start_percent,
            "end_percent": pose_end_percent,
            "looping": looping,
        }
        window_latents = ((segment_len if looping else num_frames) - 1) // 4 + 1
        if not looping:
            if pose_half is not None:
                scail2["pose_latent"] = vae.encode([(pose_half.movedim(0, 1) * 2 - 1).to(device, vae.dtype)], device, tiled=tiled_vae)[0].to(offload_device)
                scail2["driving_mask"] = scail2_mask_to_latent(mask_half)
        else:
            log.info(f"SCAIL-2: {num_frames} frames in segments of {segment_len} with {segment_overlap} frames of overlap")
            scail2.update({"pose_pixels": pose_half, "mask_pixels": mask_half, "num_frames": num_frames,
                           "segment_len": segment_len, "segment_overlap": segment_overlap})

        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()

        image_embeds = {
            "target_shape": (16, window_latents, lat_h, lat_w),
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "max_seq_len": math.ceil(lat_h * lat_w / 4 * window_latents),
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "vae": vae,
            "tiled_vae": tiled_vae,
            "scail2": scail2,
        }
        return (image_embeds,)

class WanVideoSCAIL2ColoredMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "replacement_mode": ("BOOLEAN", {"default": False, "tooltip": "Animation Mode: driving mask on black, reference mask on white. Replacement Mode: driving mask on white, reference mask on black"}),
                    "identity": ("INT", {"default": 0, "min": 0, "max": len(SCAIL2_PALETTE) - 1, "tooltip": "Identity color: 0 blue, 1 red, 2 green, 3 magenta, 4 cyan, 5 yellow. Use the same one for a character in the driving and the reference mask"}),
                },
                "optional": {
                    "pose_video_mask": ("MASK", {"tooltip": "Character mask for every driving video frame, e.g. from SAM2/SAM3"}),
                    "ref_mask": ("MASK", {"tooltip": "Character mask of the reference image"}),
                    "prev_pose_video_mask": ("IMAGE", {"tooltip": "Colored driving mask of other characters to draw this one on top of"}),
                    "prev_ref_mask": ("IMAGE", {"tooltip": "Colored reference mask of other characters to draw this one on top of"}),
                }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("pose_video_mask", "ref_mask",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Turns plain masks into the colored masks SCAIL-2 expects. Chain several for multiple characters, one identity color each"

    def process(self, replacement_mode, identity, pose_video_mask=None, ref_mask=None, prev_pose_video_mask=None, prev_ref_mask=None):
        color = torch.tensor(SCAIL2_PALETTE[identity]).view(1, 1, 1, 3)

        def render(mask, background, prev):
            mask = mask if mask.ndim == 3 else mask.unsqueeze(0)
            if prev is None:
                prev = torch.full((*mask.shape, 3), background)
            elif prev.shape[1:3] != mask.shape[1:3]:
                raise ValueError(f"mask size {tuple(mask.shape[1:])} doesn't match the previous colored mask {tuple(prev.shape[1:3])}")
            prev = prev[[min(i, prev.shape[0] - 1) for i in range(mask.shape[0])]]
            return torch.where((mask > 0.5).unsqueeze(-1), color, prev[..., :3].float().cpu())

        pose_out = render(pose_video_mask.cpu(), 1.0 if replacement_mode else 0.0, prev_pose_video_mask) if pose_video_mask is not None else prev_pose_video_mask
        ref_out = render(ref_mask.cpu(), 0.0 if replacement_mode else 1.0, prev_ref_mask) if ref_mask is not None else prev_ref_mask
        if pose_out is None:
            pose_out = torch.zeros(1, 64, 64, 3)
        if ref_out is None:
            ref_out = torch.full((1, 64, 64, 3), 0.0 if replacement_mode else 1.0)
        return (pose_out, ref_out)

class WanVideoAddSCAILReferenceEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "vae": ("WANVAE", {"tooltip": "VAE model"}),
                    "ref_image": ("IMAGE",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the reference embedding"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the embedding application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the embedding application"}),
                },
                "optional": {
                    "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, vae, ref_image, strength, start_percent, end_percent, clip_embeds=None):
        updated = dict(embeds)

        vae.to(device)
        ref_image_in = (ref_image[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
        ref_latent = vae.encode([ref_image_in], device, tiled=False)[0]
        log.info(f"SCAIL ref_latent shape: {ref_latent.shape}")

        ref_mask = torch.ones_like(ref_latent[:4])
        ref_latent = torch.cat([ref_latent, ref_mask], dim=0)
        vae.to(offload_device)

        updated.setdefault("scail_embeds", {})
        updated["scail_embeds"]["ref_latent_pos"] = ref_latent * strength
        updated["scail_embeds"]["ref_latent_neg"] = torch.zeros_like(ref_latent)
        updated["scail_embeds"]["ref_start_percent"] = start_percent
        updated["scail_embeds"]["ref_end_percent"] = end_percent
        updated["clip_context"] = clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None

        return (updated,)

class WanVideoAddSCAILPoseEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "vae": ("WANVAE", {"tooltip": "VAE model"}),
                    "pose_images": ("IMAGE", {"tooltip": "Pose images for the entire video"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the pose control"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the pose control application"}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the pose control application"}),
                },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, vae, pose_images, strength, start_percent=0.0, end_percent=1.0):
        updated = dict(embeds)

        vae.to(device)
        pose_images_in = (pose_images[..., :3].permute(3, 0, 1, 2) * 2 - 1).to(device, vae.dtype)
        pose_latent = vae.encode([pose_images_in], device, tiled=False)[0]
        pose_mask = torch.ones_like(pose_latent[:4])
        pose_latent = torch.cat([pose_latent, pose_mask], dim=0)
        log.info(f"SCAIL pose_latent shape: {pose_latent.shape}")

        vae.to(offload_device)

        updated.setdefault("scail_embeds", {})
        updated["scail_embeds"]["pose_latent"] = pose_latent
        updated["scail_embeds"]["pose_strength"] = strength
        updated["scail_embeds"]["pose_start_percent"] = start_percent
        updated["scail_embeds"]["pose_end_percent"] = end_percent

        return (updated,)


NODE_CLASS_MAPPINGS = {
    "WanVideoAddSCAILPoseEmbeds": WanVideoAddSCAILPoseEmbeds,
    "WanVideoAddSCAILReferenceEmbeds": WanVideoAddSCAILReferenceEmbeds,
    "WanVideoSCAIL2Embeds": WanVideoSCAIL2Embeds,
    "WanVideoSCAIL2ColoredMask": WanVideoSCAIL2ColoredMask,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAddSCAILReferenceEmbeds": "WanVideo Add SCAIL Reference Embeds",
    "WanVideoAddSCAILPoseEmbeds": "WanVideo Add SCAIL Pose Embeds",
    "WanVideoSCAIL2Embeds": "WanVideo SCAIL2 Embeds (SCAIL-2)",
    "WanVideoSCAIL2ColoredMask": "WanVideo SCAIL2 Colored Mask",
    }