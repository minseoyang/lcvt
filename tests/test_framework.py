"""Checks for both restored LoD branches and progressive inference."""

import csv
import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from lcvt import LCvT, LCvTConfig
from lcvt.data import parent_labels, read_manifest, validate_splits
from lcvt.experiment import classification_loss, load_model
from lcvt.patches import gather_tokens, source_fine_indices, spatial_fine_indices


def tiny_config(**overrides):
    values = dict(image_size=64, num_classes=(3, 5), channels=(8, 8, 8),
                  depths=(1, 1, 1), heads=(1, 1, 1), head_dims=(8, 8, 8),
                  branch_dim=24, branch_heads=3, branch_depth=2, patch_sizes=(2, 1),
                  dropout=0.0, embedding_dropout=0.0)
    return LCvTConfig(**dict(values, **overrides))


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_batch_one_cpu_training_reaches_every_head(self):
        model = LCvT(tiny_config()).train()
        outputs = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(outputs["lod1"]["fine"].shape, (1, 3))
        self.assertEqual(outputs["lod2"]["coarse"].shape, (1, 5))
        loss = classification_loss(outputs, torch.tensor([1]), torch.tensor([4]), [1, 2, 3, 4])
        loss.backward()
        for branch in model.branches:
            self.assertTrue(branch.selection)
            self.assertIsNotNone(branch.fine_position.grad)
            self.assertTrue(torch.isfinite(branch.fine_position.grad).all())
            for block in (branch.coarse_blocks[0], branch.fine_blocks[0]):
                self.assertIsNotNone(block.qkv.weight.grad)
                self.assertTrue(torch.isfinite(block.qkv.weight.grad).all())
        self.assertTrue(torch.isfinite(loss))

    def test_both_restored_selection_paths_run_one_fine_pass(self):
        model = LCvT(tiny_config()).eval()
        counts = [[], []]
        hooks = [branch.fine_blocks[0].register_forward_pre_hook(
            lambda module, inputs, i=i: counts[i].append(inputs[0].shape[1])
        ) for i, branch in enumerate(model.branches)]
        model(torch.randn(1, 3, 64, 64))
        for hook in hooks:
            hook.remove()
        self.assertEqual(counts, [[11], [5]])
        self.assertTrue(model.config.lod1_selection)
        self.assertTrue(model.branches[1].selection)

    def test_requested_lod_outputs_match_joint_training_outputs(self):
        model = LCvT(tiny_config()).eval()
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = model(images)
        for lod in (1, 2):
            for granularity in ("coarse", "fine"):
                torch.testing.assert_close(model.predict(images, lod, granularity), output[f"lod{lod}"][granularity])
        self.assertNotIn("early_exit", inspect.signature(model.predict).parameters)
        self.assertNotIn("exit_threshold", model.config.to_dict())

    def test_source_helper_and_gather_for_both_lods(self):
        indices = torch.tensor([[0, 3]])
        self.assertEqual(source_fine_indices(indices, 8).tolist(), [[0, 6, 1, 7, 8, 14, 9, 15]])
        tokens = torch.arange(18).reshape(1, 6, 3)
        torch.testing.assert_close(gather_tokens(tokens, torch.tensor([[4, 1]])), tokens[:, [4, 1]])
        for first, second in ((False,False),(True,False),(False,True),(True,True)):
            model = LCvT(tiny_config(lod1_selection=first, lod2_selection=second)).eval()
            for lod in (1, 2):
                self.assertTrue(torch.isfinite(model.predict(torch.randn(1, 3, 64, 64), lod)).all())

    def test_requested_lod_skips_unrelated_stages_and_branch(self):
        model = LCvT(tiny_config()).eval()
        image = torch.randn(1, 3, 64, 64)
        calls = [0] * 5
        def record(index):
            def hook(module, inputs):
                calls[index] += 1
            return hook
        hooks = [module.register_forward_pre_hook(record(index)) for index, module in enumerate([
            *model.stages, model.branches[0].coarse_blocks[0], model.branches[1].coarse_blocks[0]])]
        model.predict(image, 1)
        self.assertEqual(calls, [1, 0, 0, 1, 0])
        calls[:] = [0] * 5
        model.predict(image, 2)
        self.assertEqual(calls, [1, 1, 1, 0, 1])
        for hook in hooks:
            hook.remove()
        self.assertEqual(set(model(image, lods=(1,))), {"lod1"})
        self.assertEqual(set(model(image, lods=(2,))), {"lod2"})

    def test_high_confidence_does_not_exit_before_fine(self):
        model = LCvT(tiny_config()).eval()
        with torch.no_grad():
            for branch in model.branches:
                branch.head[1].weight.zero_()
                branch.head[1].bias.zero_()
                branch.head[1].bias[0] = 100.0
        calls = []
        hooks = [branch.fine_blocks[0].register_forward_pre_hook(
            lambda module, inputs, i=i: calls.append(i)
        ) for i, branch in enumerate(model.branches)]
        output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(calls, [0, 1])
        for branch in output.values():
            self.assertEqual(branch["coarse"].softmax(-1).max().item(), 1.0)
        for hook in hooks:
            hook.remove()

    def test_cache_reuses_features_across_lod_changes(self):
        model = LCvT(tiny_config()).eval()
        image = torch.randn(1, 3, 64, 64)
        expected = model(image)
        calls = [0] * 3
        hooks = []
        for index, stage in enumerate(model.stages):
            def record(module, inputs, i=index):
                calls[i] += 1
            hooks.append(stage.register_forward_pre_hook(record))
        cache = model.prepare_cache(image)
        self.assertEqual(calls, [0, 0, 0])
        image.zero_()
        torch.testing.assert_close(model.predict_from_cache(cache, 1, "coarse"), expected["lod1"]["coarse"])
        torch.testing.assert_close(model.predict_from_cache(cache, 1), expected["lod1"]["fine"])
        self.assertEqual(calls, [1, 0, 0])
        torch.testing.assert_close(model.predict_from_cache(cache, 2), expected["lod2"]["fine"])
        self.assertEqual(calls, [1, 1, 1])
        model.predict_from_cache(cache, 1)
        model.predict_from_cache(cache, 2)
        self.assertEqual(calls, [1, 1, 1])
        with self.assertRaises(ValueError):
            LCvT(tiny_config()).eval().predict_from_cache(cache, 1)
        for hook in hooks:
            hook.remove()

    def test_spatial_mapping_covers_the_corresponding_fine_cells(self):
        children = spatial_fine_indices(torch.tensor([[0, 3]]), 2, 8)
        expected = [row * 8 + col for row in range(4) for col in range(4)]
        expected += [row * 8 + col for row in range(4, 8) for col in range(4, 8)]
        self.assertEqual(children.tolist(), [expected])
        all_children = spatial_fine_indices(torch.tensor([[0, 1, 2, 3]]), 2, 8)
        self.assertEqual(sorted(all_children[0].tolist()), list(range(64)))
        for alpha in (0.0, 0.5, 1.0):
            model = LCvT(tiny_config(patch_mapping="spatial", alpha=alpha)).eval()
            for lod in (1, 2):
                self.assertTrue(torch.isfinite(model.predict(torch.randn(1, 3, 64, 64), lod)).all())

    def test_checkpoint_roundtrip(self):
        model = LCvT(tiny_config()).eval()
        inputs = torch.randn(1, 3, 64, 64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            torch.save({"format":"lcvt-framework-v2", "model_config":model.config.to_dict(),
                        "model_state":model.state_dict()}, path)
            loaded, _ = load_model(path, "cpu")
            torch.testing.assert_close(model.predict(inputs, 2), loaded.predict(inputs, 2))

    def test_labels_splits_and_path_validation(self):
        torch.testing.assert_close(parent_labels(torch.tensor([0, 2, 1]), [1, 0, 2], (3, 3)), torch.tensor([1, 2, 0]))
        with self.assertRaises(ValueError):
            validate_splits([("a.jpg", 0, 0)], [("a.jpg", 0, 0)])
        with self.assertRaises(ValueError):
            validate_splits([("a.jpg", 0, 0)], [("b.jpg", 1, 0)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            for rows in ([('../private.jpg', 0, 0)], [('a.jpg', 0, 0), ('b.jpg', 1, 0)]):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["path", "lod1", "lod2"])
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    read_manifest(path, (3, 5))

    def test_bad_geometry_is_rejected_and_inference_requires_eval(self):
        with self.assertRaises(ValueError):
            LCvTConfig(image_size=224)
        with self.assertRaises(RuntimeError):
            LCvT(tiny_config()).predict(torch.randn(1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
