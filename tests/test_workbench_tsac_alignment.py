"""Check the optional workbench T-SAC layout against the original networks."""

import unittest

import torch

from satellite_rl.agents.sac_transformer import PolicyNet, ValueNet
from satellite_rl.agents.workbench_networks import Encoder, MultiHeadNetwork, model_description


def tsac_config():
    return {"algorithm": "sac", "network": {
        "kind": "transformer", "transformer_layout": "tsac_13_tokens",
        "embedding_dim": 128, "num_heads": 8, "num_layers": 2,
        "feedforward_dim": 512, "activation": "relu", "dropout": 0.1,
    }}


def copy_original_weights(original, workbench, head_names):
    old = original.state_dict()
    mapped = {}
    for name, value in old.items():
        if name.startswith("encoder."):
            target = "encoder.net." + name[len("encoder."):]
        elif name.startswith(("geo_embed.", "rate_embed.", "power_embed.",
                              "slot_embed.", "pos_embed")):
            target = "encoder." + name
        else:
            target = "heads." + head_names[name.split(".", 1)[0]] + "." + name.split(".", 1)[1]
        mapped[target] = value
    workbench.load_state_dict(mapped, strict=True)


class TsacAlignmentTests(unittest.TestCase):
    def test_actor_probability_and_critic_value_parity(self):
        torch.manual_seed(51001)
        config = tsac_config()
        inputs = torch.randn(3, 204)
        for source_type, names in (
            (PolicyNet, {"freq_head": "0", "slots_head": "1", "power_head": "2"}),
            (ValueNet, {"q_freq": "0", "q_slots": "1", "q_power": "2"}),
        ):
            with self.subTest(source=source_type.__name__):
                source = source_type(204, 128, 100, 10, 10).eval()
                target = MultiHeadNetwork(config, 204, [100, 10, 10]).eval()
                copy_original_weights(source, target, names)
                self.assertEqual(sum(p.numel() for p in source.parameters()),
                                 sum(p.numel() for p in target.parameters()))
                with torch.no_grad():
                    actual = target(inputs)
                    expected = source(inputs)
                for output, reference in zip(actual, expected):
                    if source_type is PolicyNet:
                        output = output.softmax(-1)
                    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)

    def test_slot_tokens_use_ten_rows_with_twenty_features(self):
        config = tsac_config()
        encoder = Encoder(config, 204).eval()
        state = torch.arange(204, dtype=torch.float32).unsqueeze(0)
        seen = []
        hook = encoder.slot_embed.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
        try:
            with torch.no_grad():
                output = encoder(state)
        finally:
            hook.remove()
        self.assertEqual(tuple(output.shape), (1, 128))
        self.assertEqual(tuple(seen[0].shape), (1, 10, 20))
        torch.testing.assert_close(seen[0][0, 0], torch.tensor(
            list(range(4, 14)) + list(range(104, 114)), dtype=torch.float32))

    def test_description_and_shape_guard(self):
        description = model_description(tsac_config(), 204, [100, 10, 10])
        self.assertIn("tsac_13_tokens", description["variant"])
        layers = description["branches"]["策略"]
        token = next(layer for layer in layers if layer["type"] == "TokenAssembly")
        pool = next(layer for layer in layers if layer["type"] == "FirstTokenPool")
        self.assertEqual(token["shape"], [13, 128])
        self.assertEqual(pool["input_shape"], [13, 128])
        self.assertEqual(pool["shape"], [128])
        attention = next(layer for layer in layers if layer["type"] == "MultiheadAttention")
        self.assertEqual(attention["input_shape"], [13, 128])
        self.assertEqual(attention["shape"], [13, 128])
        with self.assertRaisesRegex(ValueError, "204"):
            Encoder(tsac_config(), 205)
        with self.assertRaisesRegex(ValueError, "204"):
            Encoder(tsac_config(), 204)(torch.zeros(1, 205))

    def test_other_valid_transformer_width(self):
        config = tsac_config()
        config["network"].update(embedding_dim=32, num_heads=4,
                                 num_layers=1, feedforward_dim=64, dropout=0)
        network = MultiHeadNetwork(config, 204, [100, 10, 10]).eval()
        with torch.no_grad():
            outputs = network(torch.zeros(2, 204))
        self.assertEqual(tuple(output.shape for output in outputs),
                         (torch.Size([2, 100]), torch.Size([2, 10]), torch.Size([2, 10])))

    def test_scalar_layout_default_remains_available(self):
        config = tsac_config()
        del config["network"]["transformer_layout"]
        config["network"].update(embedding_dim=16, num_heads=4,
                                 num_layers=1, feedforward_dim=32, dropout=0)
        encoder = Encoder(config, 6).eval()
        self.assertEqual(tuple(encoder(torch.ones(2, 6)).shape), (2, 16))
        self.assertEqual(tuple(encoder.position.shape), (1, 6, 16))


if __name__ == "__main__":
    unittest.main()
