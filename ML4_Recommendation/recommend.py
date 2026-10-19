"""Produce full-catalog Top-K from an explicitly supplied observed history.

Scores are ranking logits, not calibrated purchase probabilities. A smoke
checkpoint only demonstrates execution and is not a validated recommender.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recommendation_model import MultiBehaviorSASRec, BEHAVIOR2ID
from run_ml4_recommendation import MODEL_KEYS


@torch.inference_mode()
def recommend_topk(model, catalog, history, k=10, exclude_seen=True, device='cpu'):
    if k < 1 or not history:
        raise ValueError('Provide a positive k and a nonempty chronological history')
    catalog = np.asarray(catalog)
    lookup = {int(item):idx + 1 for idx,item in enumerate(catalog)}
    item_ids, behavior_ids = [], []
    for event in history:
        item, behavior = int(event['itemid']), event['event']
        if item not in lookup:
            raise ValueError(f'Item {item} is outside the training catalog; no cold-start model is implemented')
        if behavior not in BEHAVIOR2ID:
            raise ValueError(f'Unknown behavior: {behavior}')
        item_ids.append(lookup[item]); behavior_ids.append(BEHAVIOR2ID[behavior])
    length = model.maxlen
    seq = np.zeros((1,length),dtype=np.int64)
    beh = np.zeros_like(seq)
    seq[0,-min(length,len(item_ids)):] = item_ids[-length:]
    beh[0,-min(length,len(behavior_ids)):] = behavior_ids[-length:]
    model.eval()
    last = model(torch.from_numpy(seq).to(device),torch.from_numpy(beh).to(device))[:, -1, :]
    scores = (last @ model.item_emb.weight[1:].T).flatten()
    if exclude_seen:
        seen = torch.tensor(sorted(set(item_ids)),device=device,dtype=torch.long) - 1
        scores[seen] = -torch.inf
    count = min(k,int(torch.isfinite(scores).sum()))
    if count == 0:
        return []
    values,indices = torch.topk(scores,count)
    return [dict(itemid=int(catalog[idx]),score=float(score))
        for idx,score in zip(indices.cpu().tolist(),values.cpu().tolist())]


def load_recommender(checkpoint, data_dir, device='cpu'):
    saved = torch.load(checkpoint,map_location=device,weights_only=True)
    config = saved['config']
    catalog = np.load(Path(data_dir) / 'catalog.npy',allow_pickle=False)
    model = MultiBehaviorSASRec(len(catalog),**{key:config[key] for key in MODEL_KEYS}).to(device)
    model.load_state_dict(saved['model'])
    return model,catalog,config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--history',type=Path,required=True,help='JSON list of itemid/event objects, oldest first')
    parser.add_argument('--k',type=int,default=10)
    args = parser.parse_args()
    model,catalog,config = load_recommender(args.checkpoint,args.data_dir)
    history = json.loads(args.history.read_text(encoding='utf-8'))
    print(json.dumps(dict(scope=config['scope'], privacy='NOT DP',
        recommendations=recommend_topk(model,catalog,history,args.k)),indent=2))


if __name__ == '__main__':
    main()
