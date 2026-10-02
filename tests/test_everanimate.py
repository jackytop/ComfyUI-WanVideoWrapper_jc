"""Run with ComfyUI's Python; set COMFYUI_PATH for a checkout outside custom_nodes."""
import copy
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

PACK = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_PATH", PACK.parent.parent))
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import server

server.PromptServer.instance = types.SimpleNamespace(last_node_id=None, client_id=None)

package = types.ModuleType("wanvideo_everanimate_test")
package.__path__ = [str(PACK)]
sys.modules[package.__name__] = package
from wanvideo_everanimate_test.everanimate import nodes as ea_nodes
from wanvideo_everanimate_test.everanimate import sampling as ea
from wanvideo_everanimate_test.nodes_sampler import WanVideoSampler
from wanvideo_everanimate_test.wanvideo.modules.model import WanModel


class FakeVAE:
    dtype = torch.float32
    upsampling_factor = 8

    def to(self, device):
        self.device = device

    def encode(self, pixels, device, tiled=False):
        video = pixels[0]
        values = video.mean(dim=(0, 2, 3))[::4]
        return values[None, None, :, None, None].expand(1, 16, -1, video.shape[2] // 8, video.shape[3] // 8).clone()


class EverAnimateTests(unittest.TestCase):
    def test_example_widget_order_and_types(self):
        classes = {"WanVideoEverAnimateSampler_jc": ea_nodes.WanVideoEverAnimateSampler,
                   "WanVideoEverAnimateEmbeds_jc": ea_nodes.WanVideoEverAnimateEmbeds,
                   "WanVideoSampler_jc": WanVideoSampler, "WanVideoDecode_jc": ea.WanVideoDecode}
        for filename in ("wanvideo_EverAnimate_example_01.json", "wanvideo_EverAnimate_embeds_example_02.json"):
            workflow = json.loads((PACK / "example_workflows" / filename).read_text(encoding="utf-8"))
            for node in workflow["nodes"]:
                if node["type"] not in classes:
                    continue
                with self.subTest(workflow=filename, node=node["type"]):
                    values = iter(node["widgets_values"])
                    schema = classes[node["type"]].INPUT_TYPES()
                    for group in ("required", "optional"):
                        for name, spec in schema.get(group, {}).items():
                            kind = spec[0]
                            if isinstance(kind, list):
                                self.assertIn(next(values), kind)
                            elif kind in ("INT", "FLOAT", "BOOLEAN"):
                                value = next(values)
                                self.assertIsInstance(value, {"INT": int, "FLOAT": (int, float), "BOOLEAN": bool}[kind], name)
                                if name == "seed":
                                    self.assertEqual(next(values), "fixed")
                    self.assertEqual(list(values), [])

    def test_reference_schedule_and_euler_update(self):
        config = ea.make_scheduler(4, 5.0, torch.device("cpu"))
        expected = torch.tensor([1.0, 15 / 16, 5 / 6, 5 / 8, 0.0])
        torch.testing.assert_close(config["sample_scheduler"].sigmas, expected)
        torch.testing.assert_close(config["timesteps"], expected[:-1] * 1000)
        for _ in range(2):
            scheduler = copy.deepcopy(config["sample_scheduler"])
            sample = torch.ones(1, 16, 7, 2, 2)
            for timestep in config["timesteps"]:
                sample = scheduler.step(torch.full_like(sample, 2), timestep, sample)[0]
            torch.testing.assert_close(sample, -torch.ones_like(sample))
        self.assertIsNone(config["sample_scheduler"].step_index)

    def test_custom_sigmas_preserved(self):
        sigmas = torch.tensor([1.0, 0.52345, 0.1, 0.0])
        config = ea.make_scheduler(20, 5, torch.device("cpu"), sigmas=sigmas)
        torch.testing.assert_close(config["sample_scheduler"].sigmas, sigmas)
        torch.testing.assert_close(config["timesteps"], sigmas[:-1] * 1000)
        self.assertNotEqual(config["sample_scheduler"].sigmas.data_ptr(), sigmas.data_ptr())

    def test_reference_schedule_respects_step_range(self):
        config = ea.make_scheduler(4, 5, torch.device("cpu"), start_step=1, end_step=3)
        torch.testing.assert_close(config["sample_scheduler"].sigmas, torch.tensor([15 / 16, 5 / 6, 5 / 8]))
        torch.testing.assert_close(config["timesteps"], torch.tensor([15 / 16, 5 / 6]) * 1000)
        self.assertEqual(config["start_step"], 1)

    def test_loop_decode_does_not_change_cached_video(self):
        video = torch.linspace(-1, 1, 18).reshape(2, 1, 3, 3)
        original = video.clone()
        decoder = ea.WanVideoDecode()
        first = decoder.decode(None, {"video": video}, False, 272, 272, 144, 128)[0]
        second = decoder.decode(None, {"video": video}, False, 272, 272, 144, 128)[0]
        torch.testing.assert_close(first, (original + 1) / 2)
        torch.testing.assert_close(second, first)
        torch.testing.assert_close(video, original)

    def test_embeds_prepare_reference_without_sampling(self):
        reference = torch.full((1, 64, 64, 3), 0.25)
        vae = FakeVAE()
        with patch.object(ea.WanVideoClipVisionEncode, "process", return_value=({"clip_embeds": None},)), \
             patch.object(WanVideoSampler, "process") as sampler:
            output, = ea_nodes.WanVideoEverAnimateEmbeds().process(
                vae, object(), reference, reference, reference, 64, 64, 9, 3)
        sampler.assert_not_called()
        self.assertEqual(ea_nodes.WanVideoEverAnimateEmbeds.RETURN_TYPES, ("WANVIDIMAGE_EMBEDS",))
        self.assertEqual(output["everanimate"]["reference_latent"].shape, (16, 1, 8, 8))
        self.assertTrue(torch.all(output["everanimate"]["reference_latent"] == -0.5))
        self.assertEqual(str(vae.device), "cpu")
        self.assertTrue(torch.all(reference == 0.25))

    def test_optional_solvers_accept_reference_schedule(self):
        for name in ("unipc", "dpm++"):
            with self.subTest(scheduler=name):
                config = ea.make_scheduler(4, 5.0, torch.device("cpu"), name)
                scheduler = config["sample_scheduler"]
                sample = torch.ones(1, 16, 7, 2, 2)
                for timestep in config["timesteps"]:
                    sample = scheduler.step(torch.full_like(sample, 2), timestep, sample)[0]
                self.assertTrue(torch.isfinite(sample).all())
                self.assertEqual(sample.shape, (1, 16, 7, 2, 2))

    def test_control_frame_alignment_and_padding(self):
        images = torch.arange(4).view(4, 1, 1, 1)
        self.assertEqual(ea.take_frames(images, 2, 9, "pingpong").flatten().tolist(), [2, 3, 2, 1, 0, 1, 2, 3, 2])
        self.assertEqual(ea.take_frames(images, 3, 6, "loop").flatten().tolist(), [3, 0, 1, 2, 3, 0])
        self.assertEqual(ea.take_frames(images, 3, 6, "hold").flatten().tolist(), [3] * 6)
        self.assertEqual(ea.take_frames(images[:1], 100, 9, "pingpong").shape[0], 9)

    def test_condition_mask_motion_and_model_space_zeros(self):
        anchors = torch.arange(4.0)[None, :, None, None].expand(16, 4, 2, 2).clone()
        pose = torch.ones(16, 3, 2, 2)
        face = torch.zeros(9, 8, 8, 3)
        embeds = ea.make_chunk_embeds(anchors, None, pose, face, {"clip_embeds": torch.ones(1, 257, 1280)}, 9)
        condition = embeds["ref_latent"]
        self.assertEqual(embeds["target_shape"], (16, 7, 2, 2))
        self.assertEqual(embeds["wananim_num_anchor_latents"], 4)
        self.assertEqual(embeds["ref_trim"], (0, 3))
        self.assertTrue(torch.all(condition[:4, :4] == 1))
        self.assertTrue(torch.all(condition[:, 4:] == 0))
        torch.testing.assert_close(condition[4:, :4], anchors)
        motion = torch.full((16, 1, 2, 2), 7.0)
        second = ea.make_chunk_embeds(anchors, motion, pose, face, {"clip_embeds": None}, 9)["ref_latent"]
        torch.testing.assert_close(second[4:, 4:5], motion)
        self.assertTrue(torch.all(second[:4, 4:] == 0))
        self.assertTrue(torch.all(second[4:, 5:] == 0))

    def test_actual_pose_adapter_skips_four_anchors(self):
        holder = types.SimpleNamespace(pose_patch_embedding=torch.nn.Identity())
        video = [torch.zeros(1, 16, 7, 2, 2)]
        pose = torch.ones(1, 16, 3, 2, 2)
        result = WanModel.wananimate_pose_embedding(holder, video, pose, num_anchor_latents=4)
        self.assertTrue(torch.all(result[0][:, :, :4] == 0))
        self.assertTrue(torch.all(result[0][:, :, 4:] == 1))

    def test_actual_face_adapter_skips_four_anchors(self):
        class Motion(torch.nn.Module):
            def forward(self, pixels):
                return pixels.mean(dim=(2, 3))

        class Face(torch.nn.Module):
            dtype = torch.float32

            def forward(self, vectors):
                return vectors[:, ::4].unsqueeze(2)

        holder = types.SimpleNamespace(motion_encoder=Motion(), face_encoder=Face(), main_device="cpu", offload_device="cpu")
        result = WanModel.wananimate_face_embedding(holder, torch.ones(1, 3, 9, 8, 8), num_anchor_latents=4)
        self.assertEqual(result.shape, (1, 7, 1, 3))
        self.assertTrue(torch.all(result[:, :4] == 0))
        self.assertTrue(torch.all(result[:, 4:] == 1))

    def test_three_chunks_keep_memory_and_refresh_clip(self):
        calls, clip_images = [], []
        vae = FakeVAE()
        model = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace(pose_patch_embedding=True)))
        clip = types.SimpleNamespace(model=types.SimpleNamespace(to=lambda device: None))
        reference = torch.full((1, 64, 64, 3), 0.25)
        pose = torch.arange(20.0)[:, None, None, None].expand(20, 64, 64, 3) / 20
        face = pose.clone()
        original_process = WanVideoSampler.process
        original_decode = ea.WanVideoDecode.decode
        cache = {"cache_type": "test"}

        def clip_encode(*args, **kwargs):
            clip_images.append(args[1].clone())
            return ({"clip_embeds": torch.zeros(1, 257, 1280)},)

        def sample(**kwargs):
            if "everanimate" in kwargs["image_embeds"]:
                return original_process(WanVideoSampler(), **kwargs)
            calls.append(copy.deepcopy(kwargs["image_embeds"]))
            self.assertNotIn("everanimate", kwargs["image_embeds"])
            self.assertEqual(kwargs["seed"], 11 + (len(calls) - 1) * 42)
            self.assertEqual(kwargs["steps"], 2)
            self.assertEqual(kwargs["cfg"], 1.0)
            self.assertEqual(kwargs["rope_function"], "comfy_chunked")
            self.assertIs(kwargs["cache_args"], cache)
            torch.testing.assert_close(kwargs["scheduler"]["timesteps"], torch.tensor([1000.0, 5000 / 6]))
            samples = torch.full((1, 16, 7, 8, 8), float(len(calls)))
            samples[:, :, :4] = -99  # These slots must never be decoded or propagated as motion.
            return {"samples": samples, "has_ref": True, "ref_trim": (0, 3)}, {}

        def decode(vae, samples, **kwargs):
            self.assertEqual(samples["samples"].shape, (1, 16, 3, 8, 8))
            level = samples["samples"].mean().item() / 10
            frames = torch.linspace(level, level + 0.08, 9)[:, None, None, None].expand(9, 64, 64, 3).clone()
            return (frames,)

        with patch.object(WanVideoSampler, "process", side_effect=sample), \
             patch.object(ea.WanVideoClipVisionEncode, "process", side_effect=clip_encode), \
             patch.object(ea.WanVideoDecode, "decode", side_effect=decode):
            embeds, = ea_nodes.WanVideoEverAnimateEmbeds().process(
                vae, clip, reference, pose, face, 64, 64, 9, 3)
            initial_reference = embeds["everanimate"]["reference_latent"].clone()
            result, _ = original_process(
                WanVideoSampler(), model, embeds, 5.0, 2, 1.0, 11, "euler", 0,
                text_embeds={}, force_offload=True, rope_function="comfy_chunked", cache_args=cache)
        frames = original_decode(ea.WanVideoDecode(), vae, result, False, 272, 272, 144, 128)[0]
        self.assertEqual(frames.shape, (19, 64, 64, 3))
        torch.testing.assert_close(clip_images[0], reference)
        torch.testing.assert_close(clip_images[1], torch.full_like(reference, 0.18))
        torch.testing.assert_close(clip_images[2], torch.full_like(reference, 0.28))
        torch.testing.assert_close(calls[1]["ref_latent"][4:, :4], calls[2]["ref_latent"][4:, :4])
        self.assertTrue(torch.all(calls[1]["ref_latent"][4:, 4:5] == 1))
        self.assertTrue(torch.all(calls[2]["ref_latent"][4:, 4:5] == 2))
        self.assertTrue(torch.all(calls[2]["ref_latent"][4:, 3:4] == -0.5))
        self.assertAlmostEqual(calls[1]["pose_latents"][0, 0, 0, 0, 0].item(), -0.5)
        self.assertTrue(torch.all(result["samples"] == 3))
        self.assertTrue(torch.all(reference == 0.25))
        torch.testing.assert_close(embeds["everanimate"]["reference_latent"], initial_reference)
        self.assertAlmostEqual(frames[9, 0, 0, 0].item(), 0.24)
        self.assertAlmostEqual(frames[14, 0, 0, 0].item(), 0.34)

    def test_user_anchors_and_explicit_scheduler_are_retained(self):
        reference = torch.full((1, 64, 64, 3), 0.25)
        vae = FakeVAE()
        clip = types.SimpleNamespace(model=types.SimpleNamespace(to=lambda device: None))
        model = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace(pose_patch_embedding=True)))
        explicit = ea.make_scheduler(3, 2, torch.device("cpu"))
        conditions = []

        def sample(**kwargs):
            self.assertIs(kwargs["scheduler"], explicit)
            self.assertEqual(kwargs["cfg"], 2.5)
            self.assertEqual(kwargs["seed"], 5 + len(conditions) * 7)
            conditions.append(kwargs["image_embeds"]["ref_latent"][4:, :4].clone())
            return {"samples": torch.ones(1, 16, 7, 8, 8)}, {"samples": None}

        with patch.object(ea.WanVideoClipVisionEncode, "process", return_value=({"clip_embeds": None},)), \
             patch.object(ea.WanVideoDecode, "decode", return_value=(reference.repeat(9, 1, 1, 1),)):
            embeds, = ea_nodes.WanVideoEverAnimateEmbeds().process(
                vae, clip, reference, reference, reference, 64, 64, 9, 2,
                anchor_mode="user_image", seed_multiplier=7)
            result, _ = ea.sample_everanimate(sample, model, embeds["everanimate"], 3, 5, 5, explicit, cfg=2.5)
        self.assertEqual(result["video"].shape[0], 14)
        for condition in conditions:
            self.assertTrue(torch.all(condition == -0.5))

    def test_legacy_sampler_uses_the_same_embeds_path(self):
        reference = torch.full((1, 64, 64, 3), 0.25)
        vae = FakeVAE()
        raw_video = torch.full((9, 64, 64, 3), -0.6)
        last = torch.ones(1, 16, 3, 8, 8)
        with patch.object(ea.WanVideoClipVisionEncode, "process", return_value=({"clip_embeds": None},)), \
             patch.object(WanVideoSampler, "process", return_value=({"video": raw_video, "samples": last}, {})) as sampler:
            frames, latent = ea_nodes.WanVideoEverAnimateSampler().process(
                object(), vae, {}, object(), reference, reference, reference,
                64, 64, 9, 1, 2, 1.0, 5.0, 11, True, False)
        self.assertIn("everanimate", sampler.call_args.kwargs["image_embeds"])
        self.assertTrue(torch.allclose(frames, torch.full_like(frames, 0.2)))
        self.assertIs(latent["samples"], last)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
