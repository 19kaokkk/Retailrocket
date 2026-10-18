"""Focused checks of USER gradients, clipping/noise and privacy accounting."""
import unittest

import numpy as np
import pandas as pd
import torch
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer

from run_ml5_privacy import (Config, FEATURES, UserData, make_model, poisson_users,
    schedule, train_steps, user_gradients, accountant_epsilon)


class PrivacyMechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_sessions_averaged_before_user_clipping(self):
        torch.manual_seed(7)
        model = make_model(3)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.4, -0.2, 0.1]]))
            model.bias.fill_(-0.3)
        x = torch.rand(7, 3)
        y = torch.tensor([0., 1., 0., 1., 1., 0., 0.])
        owner = torch.tensor([0, 0, 0, 1, 2, 2, 2])
        counts = torch.tensor([3, 1, 3])
        actual = user_gradients(model, x, y, owner, counts, 100.)
        expected = []
        for user in range(3):
            selection = owner == user
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(x[selection]).flatten(), y[selection], pos_weight=torch.tensor(100.))
            weight, bias = torch.autograd.grad(loss, tuple(model.parameters()))
            expected.append(torch.cat((weight.flatten(), bias)))
        torch.testing.assert_close(actual, torch.stack(expected))

    def test_opacus_clips_joint_user_weight_and_bias(self):
        model = make_model(2)
        base = torch.optim.SGD(model.parameters(), lr=1)
        opt = DPOptimizer(base, noise_multiplier=0, max_grad_norm=1,
            expected_batch_size=2)
        gradients = torch.tensor([[3., 0., 4.], [0., 0., 0.5]])
        model.weight.grad_sample = gradients[:, :2].unsqueeze(1)
        model.bias.grad_sample = gradients[:, 2:]
        opt.pre_step()
        # Joint norm 5, not separate norms for weight and bias.
        torch.testing.assert_close(model.weight.grad, torch.tensor([[0.3, 0.]]))
        torch.testing.assert_close(model.bias.grad, torch.tensor([0.65]))

    def test_empty_poisson_batch_still_noised_and_accounted(self):
        model = make_model(2)
        account = RDPAccountant()
        base = torch.optim.SGD(model.parameters(), lr=1)
        opt = DPOptimizer(base, noise_multiplier=1, max_grad_norm=2,
            expected_batch_size=4, generator=torch.Generator().manual_seed(9))
        opt.attach_step_hook(account.get_optimizer_hook_fn(sample_rate=0.1))
        model.weight.grad_sample = torch.empty(0, 1, 2)
        model.bias.grad_sample = torch.empty(0, 1)
        opt.pre_step()
        self.assertEqual(sum(h[2] for h in account.history), 1)
        self.assertGreater(float(model.weight.grad.abs().sum()), 0)

    def test_expected_batch_denominator_not_actual_count(self):
        model = make_model(1)
        opt = DPOptimizer(torch.optim.SGD(model.parameters(), lr=1),
            noise_multiplier=0, max_grad_norm=100, expected_batch_size=10)
        model.weight.grad_sample = torch.ones(2, 1, 1)
        model.bias.grad_sample = torch.zeros(2, 1)
        opt.pre_step()
        self.assertAlmostEqual(model.weight.grad.item(), 0.2)

    def test_sampler_is_user_poisson_and_handles_q_one(self):
        rng = np.random.default_rng(32)
        draws = [poisson_users(rng, 20, 0.2) for _ in range(10000)]
        self.assertTrue(all(len(d) == len(set(d)) for d in draws))
        self.assertAlmostEqual(np.mean([len(d) for d in draws]), 4, delta=0.08)
        rate = np.bincount(np.concatenate(draws), minlength=20) / len(draws)
        self.assertTrue(np.all(np.abs(rate - 0.2) < 0.015))
        np.testing.assert_array_equal(np.sort(poisson_users(rng, 20, 1)), np.arange(20))

    def test_client_accountant_persists_across_rounds_and_prox_is_post_noise(self):
        frame = pd.DataFrame({feature: np.ones(6) for feature in FEATURES})
        frame['visitorid'] = [1, 1, 2, 3, 3, 3]
        frame['label_has_purchase'] = [0, 1, 0, 1, 0, 0]
        data = UserData(frame)
        self.assertEqual(data.n_users, 3)
        np.testing.assert_array_equal(data.counts, [2, 1, 3])
        cfg = Config(deterministic=True, clip=10, mu=0.1)
        spec = schedule(data.n_users, 2, 1)
        model = make_model(len(FEATURES))
        account = RDPAccountant()
        anchor = [p.detach().clone() for p in model.parameters()]
        train_steps(model, data, spec, cfg, 'dp', 1, account, 'first', anchor)
        first = accountant_epsilon(account, 1e-7)
        train_steps(model, data, spec, cfg, 'dp', 1, account, 'second', anchor)
        self.assertEqual(sum(h[2] for h in account.history), spec['steps'] * 2)
        self.assertGreater(accountant_epsilon(account, 1e-7), first)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_proximal_term_after_noise_matches_parameter_update(self):
        frame = pd.DataFrame({feature: np.ones(4) for feature in FEATURES})
        frame['visitorid'] = [1, 1, 2, 2]
        frame['label_has_purchase'] = [0, 1, 0, 0]
        data = UserData(frame)
        cfg = Config(deterministic=True, clip=10, mu=0.3, lr=0.1)
        model = make_model(len(FEATURES))
        with torch.no_grad():
            model.weight.fill_(0.2)
            model.bias.fill_(0.4)
        before = [p.detach().clone() for p in model.parameters()]
        anchor = [torch.zeros_like(p) for p in model.parameters()]
        twin = make_model(len(FEATURES))
        twin.load_state_dict(model.state_dict())
        spec = schedule(data.n_users, data.n_users, 1)
        train_steps(model, data, spec, cfg, 'dp', 1, RDPAccountant(), 'same-noise', anchor)
        train_steps(twin, data, spec, cfg, 'dp', 1, RDPAccountant(), 'same-noise')
        for actual, nonprox, initial in zip(model.parameters(), twin.parameters(), before):
            torch.testing.assert_close(actual, nonprox - cfg.lr * cfg.mu * initial)

    def test_noise_scale_is_sigma_times_clip_divided_by_expected_batch(self):
        model = make_model(1)
        generator = torch.Generator().manual_seed(17)
        twin_generator = torch.Generator().manual_seed(17)
        opt = DPOptimizer(torch.optim.SGD(model.parameters(), lr=1),
            noise_multiplier=2, max_grad_norm=3, expected_batch_size=10, generator=generator)
        model.weight.grad_sample = torch.zeros(1, 1, 1)
        model.bias.grad_sample = torch.zeros(1, 1)
        opt.pre_step()
        expected_w = torch.normal(0., 6., size=model.weight.shape, generator=twin_generator) / 10
        expected_b = torch.normal(0., 6., size=model.bias.shape, generator=twin_generator) / 10
        torch.testing.assert_close(model.weight.grad, expected_w)
        torch.testing.assert_close(model.bias.grad, expected_b)

    def test_absent_user_keeps_public_slot_and_has_zero_gradient(self):
        frame = pd.DataFrame({feature: np.ones(3) for feature in FEATURES})
        frame['visitorid'] = [1, 1, 2]
        frame['label_has_purchase'] = [0, 1, 0]
        roster = [1, 2, 3]
        original = UserData(frame, roster)
        neighboring = UserData(frame[frame.visitorid != 1], roster)
        self.assertEqual(original.n_users, neighboring.n_users)
        self.assertEqual(schedule(original.n_users, 2, 3), schedule(neighboring.n_users, 2, 3))
        model = make_model(len(FEATURES))
        gradients = user_gradients(model, *neighboring.batch(np.array([0, 1, 2])), 100.)
        torch.testing.assert_close(gradients[[0, 2]], torch.zeros(2, len(FEATURES) + 1))
        self.assertGreater(float(gradients[1].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
