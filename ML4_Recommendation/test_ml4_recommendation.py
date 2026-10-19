"""Focused mechanism and real-data checks; no claims of statistical efficacy."""
import copy
import io
import json
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from prepare_recommendation_data import DataOnlyUnpickler, client_bucket, pad, validate_zip_names
from recommendation_model import MultiBehaviorSASRec, per_user_loss, proximal_penalty
from recommend import recommend_topk
from run_ml4_recommendation import (RecommendationData, aggregate_states, evaluate,
    make_optimizer, metrics_from_ranks, ranks_from_scores, seed_all, train_pass, validate_config)

BASE = Path(__file__).resolve().parent


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_all(42)
        self.model = MultiBehaviorSASRec(12, hidden_units=8,maxlen=4,num_blocks=1,dropout_rate=0)
        self.seq = torch.tensor([[0,1,2,3],[0,0,4,5]])
        self.beh = torch.tensor([[0,1,2,1],[0,0,1,3]])
        self.pos = torch.tensor([[0,2,3,4],[0,0,5,6]])
        self.neg = torch.tensor([[0,7,8,9],[0,0,8,9]])

    def test_causal_mask_blocks_future(self):
        self.model.eval()
        original = self.model(self.seq,self.beh).detach()
        changed = self.seq.clone(); changed[:, -1] = 10
        new = self.model(changed,self.beh).detach()
        torch.testing.assert_close(original[:, :-1],new[:, :-1],rtol=0,atol=0)

    def test_padding_hidden_states_are_zero(self):
        result = self.model(self.seq,self.beh)
        self.assertEqual(torch.count_nonzero(result[self.seq == 0]).item(),0)

    def test_behavior_embeddings_affect_predictions(self):
        original = self.model(self.seq,self.beh)
        changed = self.beh.clone(); changed[self.seq != 0] = 1
        self.assertFalse(torch.equal(original,self.model(self.seq,changed)))

    def test_per_user_normalization_not_token_weighting(self):
        together = per_user_loss(self.model,self.seq,self.beh,self.pos,self.neg)
        separate = torch.cat([per_user_loss(self.model,self.seq[i:i+1],self.beh[i:i+1],
            self.pos[i:i+1],self.neg[i:i+1]) for i in range(2)])
        torch.testing.assert_close(together,separate)
        self.assertEqual(together.shape,(2,))
        self.assertTrue(torch.isfinite(together).all())

    def test_proximal_gradient_includes_bias(self):
        anchor = {n:p.detach().clone() for n,p in self.model.named_parameters()}
        name, parameter = next((n,p) for n,p in self.model.named_parameters() if n.endswith('bias'))
        with torch.no_grad(): parameter.add_(0.5)
        proximal_penalty(self.model,anchor,0.2).backward()
        torch.testing.assert_close(parameter.grad,torch.full_like(parameter,0.1))
        self.assertEqual(float(proximal_penalty(self.model,anchor,0).detach()),0)

    def test_same_initialization_across_algorithms(self):
        seed_all(99)
        first = MultiBehaviorSASRec(12,hidden_units=8,maxlen=4,num_blocks=1)
        seed_all(99)
        second = MultiBehaviorSASRec(12,hidden_units=8,maxlen=4,num_blocks=1)
        for key,value in first.state_dict().items():
            torch.testing.assert_close(value,second.state_dict()[key],rtol=0,atol=0)

    def test_recommendations_mask_seen_items(self):
        catalog = np.arange(100,112)
        result = recommend_topk(self.model,catalog,[{'itemid':100,'event':'view'},
            {'itemid':101,'event':'addtocart'}],k=4)
        self.assertEqual(len(result),4)
        self.assertTrue(all(r['itemid'] not in [100,101] for r in result))
        self.assertEqual(len(set(r['itemid'] for r in result)),4)

    def test_recommendation_rejects_unknown_items(self):
        with self.assertRaises(ValueError):
            recommend_topk(self.model,np.arange(100,112),[{'itemid':1,'event':'view'}])


class ProtocolTests(unittest.TestCase):
    def test_weighted_aggregation(self):
        states = [({'weight':torch.tensor([1.,3.])},1),({'weight':torch.tensor([5.,7.])},3)]
        torch.testing.assert_close(aggregate_states(iter(states))['weight'],torch.tensor([4.,6.]))

    def test_invalid_aggregation_weights_fail(self):
        with self.assertRaises(ValueError):
            aggregate_states([({'w':torch.tensor([1.])},0)])

    def test_rank_ties_follow_shared_priorities(self):
        scores = np.array([[1.,1.,0.],[0.,1.,0.]])
        priority = np.array([[2,1,0],[2,0,1]])
        np.testing.assert_array_equal(ranks_from_scores(scores),[1,2])
        np.testing.assert_array_equal(ranks_from_scores(scores,priority),[2,3])

    def test_metrics_one_positive(self):
        m = metrics_from_ranks(np.array([1,10,11]))
        self.assertEqual(m['hr10'],2/3)
        self.assertEqual(m['recall10'],m['hr10'])
        self.assertAlmostEqual(m['ndcg10'],(1 + 1/np.log2(11))/3)

    def test_nonfinite_predictions_fail(self):
        with self.assertRaises(ValueError):
            ranks_from_scores([[0,np.nan]])

    def test_stable_visitor_grouping(self):
        self.assertEqual(client_bucket(123),client_bucket(np.int64(123)))
        self.assertTrue(all(0 <= client_bucket(i) < 50 for i in range(100)))

    def test_archive_traversal_rejected(self):
        for name in ['../escape.csv','C:/escape.csv','/escape.csv','safe/../escape.csv']:
            with self.assertRaises(ValueError):
                validate_zip_names([zipfile.ZipInfo(name)])

    def test_pickle_global_execution_rejected(self):
        with self.assertRaises(Exception):
            DataOnlyUnpickler(io.BytesIO(b'cos\nsystem\n.')).load()

    def test_subset_cannot_be_labelled_full(self):
        config = json.loads((BASE / 'configs/full.json').read_text())
        config['max_clients'] = 3
        with self.assertRaises(ValueError): validate_config(config)

    def test_padding_keeps_recent_events(self):
        np.testing.assert_array_equal(pad([1,2,3,4],3),[2,3,4])
        np.testing.assert_array_equal(pad([2],3),[0,0,2])


@unittest.skipUnless((BASE / 'data/data_manifest.json').exists(),'Prepared ML3 data not present')
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = RecommendationData(BASE / 'data')
        cls.config = json.loads((BASE / 'configs/smoke.json').read_text())
        cls.rows,cls.evaluation = cls.data.subset(cls.config)

    def test_full_ml3_alignment_assertions(self):
        self.assertTrue(all(self.data.manifest['checks'].values()))
        self.assertEqual(len(self.data.eval_users),172513)
        self.assertEqual(self.data.num_items,114800)

    def test_negative_samples_exclude_entire_training_history(self):
        _,_,positive,negative = self.data.batch(self.rows[:16],np.random.default_rng(42),50,'cpu')
        for i,row in enumerate(self.rows[:16]):
            start,end = self.data.history_offsets[row:row+2]
            self.assertFalse(np.isin(negative[i][positive[i] > 0].numpy(),self.data.history_items[start:end]).any())

    def test_users_are_unique_and_client_disjoint(self):
        self.assertEqual(len(self.rows),len(np.unique(self.data.user_ids[self.rows])))
        self.assertEqual(len(np.unique(self.data.client_ids[self.rows])),3)

    def test_single_client_fedavg_equals_centralized_for_one_pass(self):
        config = dict(self.config,hidden_units=8,num_blocks=1,dropout_rate=0,maxlen=8)
        seed_all(44)
        first = MultiBehaviorSASRec(self.data.num_items,hidden_units=8,num_blocks=1,dropout_rate=0,maxlen=8)
        second = copy.deepcopy(first)
        rows = self.rows[:16]
        a = train_pass(first,make_optimizer(first,config),self.data,rows,np.random.default_rng(7),config,'cpu')
        b = train_pass(second,make_optimizer(second,config),self.data,rows,np.random.default_rng(7),config,'cpu')
        state = aggregate_states([(second.state_dict(),len(rows))])
        for name,value in first.state_dict().items():
            torch.testing.assert_close(value,state[name],rtol=0,atol=0)
        self.assertEqual(a['optimizer_steps'],b['optimizer_steps'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
