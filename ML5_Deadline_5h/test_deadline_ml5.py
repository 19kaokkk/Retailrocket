"""Regression tests for the deadline-only adapter, not a privacy certificate."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_deadline_ml5 as deadline
from recommendation_model import MultiBehaviorSASRec


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = MultiBehaviorSASRec(20, hidden_units=4, maxlen=4, num_blocks=1,
                                        num_heads=1, dropout_rate=0)
        self.arrays = (torch.tensor([[1,2,3,4],[0,2,3,1],[0,0,1,2]]),
            torch.tensor([[1,1,2,1],[0,1,3,1],[0,0,1,1]]),
            torch.tensor([[2,3,4,5],[0,3,1,6],[0,0,2,3]]),
            torch.tensor([[6,7,8,9],[0,7,8,9],[0,0,8,9]]))
        arrays = self.arrays
        class Data:
            def batch(self, rows, rng, maxlen, device):
                return tuple(a[rows].to(device) for a in arrays)
        self.data = Data()
        self.config = dict(learning_rate=.05, maxlen=4, mu=.01)
        self.spec = dict(q=1., steps_per_round=1, expected_batch_users=7.)

    def check_case(self, absent, proximal):
        a, b = copy.deepcopy(self.model), copy.deepcopy(self.model)
        roster = dict(rows=np.arange(3), absent=np.array(absent))
        anchor = {n:p.detach().clone()+.1 for n,p in a.named_parameters()} if proximal else None
        deadline.ORIGINAL_TRAIN_LOCAL(a,self.data,roster,np.arange(3),self.spec,
            self.config,'no_dp',0,None,'cpu',anchor)
        deadline.batched_no_dp(b,self.data,roster,np.arange(3),self.spec,
            self.config,'cpu',anchor)
        for left,right in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)

    def test_expected_denominator_and_per_user_mean(self):
        self.check_case([False]*3,False)

    def test_absent_slots_keep_denominator_and_proximal(self):
        self.check_case([False,True,False],True)

    def test_all_absent_updates_only_proximal(self):
        self.check_case([True]*3,True)

    def test_private_modes_delegate_unchanged(self):
        for mode in ('clip_only','dp'):
            with patch.object(deadline,'ORIGINAL_TRAIN_LOCAL') as original:
                deadline.selected_train_local(self.model,self.data,{},[],{}, {},
                    mode,2.,None,'cpu')
                original.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
