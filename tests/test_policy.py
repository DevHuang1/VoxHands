from __future__ import annotations

import unittest

import numpy as np

from voxhands.policy import CanonicalDinnerPolicy, MLPPolicy, Policy
from voxhands.policy.deterministic import CANONICAL_DINNER_STEPS


class DeterministicPolicyTests(unittest.TestCase):

    def test_returns_first_action_when_fresh(self) -> None:
        policy = CanonicalDinnerPolicy()
        action = policy({})
        expected = CANONICAL_DINNER_STEPS[0]
        self.assertEqual(action.arm, expected["arm"])
        self.assertEqual(action.object_id, expected["object_id"])
        self.assertEqual(action.target_id, expected["target_id"])
        self.assertEqual(action.duration_ms, expected["duration_ms"])

    def test_advances_through_full_sequence(self) -> None:
        policy = CanonicalDinnerPolicy()
        for i, expected in enumerate(CANONICAL_DINNER_STEPS):
            action = policy({})
            self.assertEqual(action.arm, expected["arm"], f"step {i} arm")
            self.assertEqual(action.object_id, expected["object_id"], f"step {i} object_id")
            self.assertEqual(action.target_id, expected["target_id"], f"step {i} target_id")
        action = policy({})
        first = CANONICAL_DINNER_STEPS[0]
        self.assertEqual(action.arm, first["arm"])
        self.assertEqual(action.object_id, first["object_id"])

    def test_can_be_pinned_to_step(self) -> None:
        policy = CanonicalDinnerPolicy()
        policy({})
        action = policy({"step": 5})
        expected = CANONICAL_DINNER_STEPS[5]
        self.assertEqual(action.arm, expected["arm"])
        self.assertEqual(action.object_id, expected["object_id"])

    def test_returns_action_not_plan(self) -> None:
        policy = CanonicalDinnerPolicy()
        result = policy({})
        self.assertFalse(isinstance(result, type) and issubclass(result, Exception))
        self.assertEqual(result.status, "queued")


class MLPPolicyTests(unittest.TestCase):

    def test_forward_shape(self) -> None:
        net = MLPPolicy(input_dim=16, hidden_dims=[32, 16], output_dim=8)
        x = np.random.default_rng(0).standard_normal(16)
        out = net.forward(x)
        self.assertEqual(out.shape, (8,))
        self.assertEqual(out.dtype, np.float64)

    def test_one_backward_step_no_nan(self) -> None:
        net = MLPPolicy(input_dim=16, hidden_dims=[32, 16], output_dim=8)
        x = np.random.default_rng(1).standard_normal(16)
        target = np.random.default_rng(2).standard_normal(8)
        loss = net.backward(x, target)
        self.assertFalse(np.isnan(loss), "loss is NaN")
        for p in net.params():
            self.assertFalse(np.any(np.isnan(p)), "parameter contains NaN")

    def test_train_step_reduces_loss(self) -> None:
        net = MLPPolicy(input_dim=8, hidden_dims=[16], output_dim=4, learning_rate=0.01)
        x = np.array([1.0, 0.0, 0.5, -0.5, 0.3, -0.3, 0.1, 0.2])
        target = np.array([1.0, 0.0, 0.0, 0.0])
        loss1 = net.backward(x, target)
        loss2 = net.backward(x, target)
        self.assertLess(loss2, loss1)

    def test_save_and_load(self) -> None:
        import tempfile
        import os

        net1 = MLPPolicy(input_dim=8, hidden_dims=[16], output_dim=4)
        x = np.array([0.1] * 8)
        out1 = net1.forward(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.json")
            net1.save(path)
            net2 = MLPPolicy(input_dim=8, hidden_dims=[16], output_dim=4)
            net2.load(path)
            out2 = net2.forward(x)
        np.testing.assert_array_almost_equal(out1, out2)

    def test_train_with_demos(self) -> None:
        net = MLPPolicy(input_dim=4, hidden_dims=[8], output_dim=2, learning_rate=0.01)
        demos = [{"state": {"a": 1.0, "b": 0.5}, "target": np.array([1.0, 0.0])} for _ in range(5)]
        net.train(demos)
        for p in net.params():
            self.assertFalse(np.any(np.isnan(p)))

    def test_is_subclass_of_policy(self) -> None:
        self.assertTrue(issubclass(MLPPolicy, Policy))

    def test_call_returns_action(self) -> None:
        net = MLPPolicy(input_dim=8, hidden_dims=[16], output_dim=2)
        action = net({"x": 0.5, "y": 0.3})
        self.assertEqual(action.status, "queued")
        self.assertIn(action.arm, ("left", "right"))


class CanonicalSequenceTests(unittest.TestCase):

    def test_sequence_length(self) -> None:
        self.assertEqual(len(CANONICAL_DINNER_STEPS), 9)

    def test_step_names(self) -> None:
        names = [s["action"] for s in CANONICAL_DINNER_STEPS]
        self.assertEqual(names[0], "open_drawer")
        self.assertEqual(names[-1], "check_final_state")


if __name__ == "__main__":
    unittest.main()
