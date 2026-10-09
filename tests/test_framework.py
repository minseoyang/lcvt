"""Checks for the retained experiment computation and portable data/weights."""

import csv
import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from lcvt import LCvT, LCvTConfig
from lcvt.data import parent_labels, read_manifest, validate_splits
from lcvt.experiment import classification_loss, load_model
from lcvt.patches import gather_tokens, source_fine_indices


def tiny_config(**overrides):
    values = dict(image_size=64, num_classes=(3, 5), channels=(8, 8, 8),
                  depths=(1, 1, 1), heads=(1, 1, 1), head_dims=(8, 8, 8),
                  branch_dim=24, branch_heads=3, branch_depth=1, patch_sizes=(2, 1),
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
            for block in (branch.coarse_blocks[0], branch.fine_blocks[0]):
                self.assertIsNotNone(block.qkv.weight.grad)
                self.assertTrue(torch.isfinite(block.qkv.weight.grad).all())
        self.assertTrue(torch.isfinite(loss))

    def test_original_default_token_counts_and_lod2_repeat_are_retained(self):
        model = LCvT(tiny_config()).eval()
        counts = [[], []]
        hooks = [branch.fine_blocks[0].register_forward_pre_hook(
            lambda module, inputs, i=i: counts[i].append(inputs[0].shape[1])
        ) for i, branch in enumerate(model.branches)]
        model(torch.randn(1, 3, 64, 64))
        for hook in hooks:
            hook.remove()
        self.assertEqual(counts, [[65], [17, 17]])
        self.assertFalse(model.config.lod1_selection)
        self.assertFalse(model.branches[1].selection)

    def test_inference_selects_the_requested_output_without_adaptive_paths(self):
        model = LCvT(tiny_config()).eval()
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            output = model(images)
        for lod in (1, 2):
            for granularity in ("coarse", "fine"):
                torch.testing.assert_close(model.predict(images, lod, granularity), output[f"lod{lod}"][granularity])
        self.assertNotIn("early_exit", inspect.signature(model.predict).parameters)
        self.assertNotIn("exit_threshold", model.config.to_dict())

    def test_optional_source_lod1_helper_and_gather(self):
        indices = torch.tensor([[0, 3]])
        self.assertEqual(source_fine_indices(indices, 8).tolist(), [[0, 6, 1, 7, 8, 14, 9, 15]])
        tokens = torch.arange(18).reshape(1, 6, 3)
        torch.testing.assert_close(gather_tokens(tokens, torch.tensor([[4, 1]])), tokens[:, [4, 1]])
        model = LCvT(tiny_config(lod1_selection=True)).eval()
        self.assertTrue(torch.isfinite(model.predict(torch.randn(1, 3, 64, 64), 1)).all())

    def test_checkpoint_roundtrip(self):
        model = LCvT(tiny_config()).eval()
        inputs = torch.randn(1, 3, 64, 64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            torch.save({"format":"lcvt-experiment-v1", "model_config":model.config.to_dict(),
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
