"""Verify saved scores/checksums without changing training or accounting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE/'results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    experiment = json.loads((OUT/'experiment.json').read_text(encoding='utf-8'))
    cohort = pd.read_csv(OUT/'cohort_metadata.csv')
    assert len(cohort)==cohort.visitorid.nunique()==500
    assert cohort.client_id.nunique()==10 and (cohort.groupby('client_id').size()==50).all()
    assert cohort.train_events.min()>=10
    users = np.load(OUT/'evaluated_users.npy',allow_pickle=False)
    assert len(users)==experiment['evaluation_users']
    assert np.isin(users,cohort.visitorid.to_numpy()).all()
    records=[]
    for path in OUT.glob('runs/**/metrics.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        assert r['fingerprint']==experiment['fingerprint']
        assert r['scope']=='active_user_subset_time_bounded'
        assert r['final_round']==3 and r['roster_users']==500 and r['clients']==10
        for name,expected in r['artifact_sha256'].items():
            assert digest(path.parent/name)==expected, path.parent/name
        ranks=np.load(path.with_name('test_ranks.npy'),allow_pickle=False)
        assert len(ranks)==len(users) and np.all((ranks>=1)&(ranks<=101))
        hit=ranks<=10
        actual=dict(hr10=float(hit.mean()),recall10=float(hit.mean()),
                    ndcg10=float(np.where(hit,1/np.log2(ranks+1),0).mean()),
                    mrr10=float(np.where(hit,1/ranks,0).mean()),mean_rank=float(ranks.mean()))
        for key,value in actual.items():
            assert abs(value-r['test'][key])<1e-12,(path,key)
        history=pd.read_csv(path.with_name('round_history.csv'))
        assert history['round'].tolist()==[1,2,3]
        if r['mode']=='dp':
            assert np.isfinite(r['epsilon_upper_bound'])
            assert r['epsilon_upper_bound']<=r['target_epsilon']+.01 or any(
                e['reserved_steps']>e['planned_steps'] for e in r['ledger'])
            for e in r['ledger']:
                assert e['committed_steps']==e['planned_steps']
                assert e['reserved_steps']>=e['committed_steps']
                assert e['expected_batch_users']==e['q']*e['roster_users']
        records.append(dict(algorithm=r['algorithm'],mode=r['mode'],
            target_epsilon=r['target_epsilon'],replicate=r['replicate'],**actual))
    frame=pd.DataFrame(records)
    complete=[int(rep) for rep,g in frame.groupby('replicate') if len(g)==18] if len(frame) else []
    summary_path=OUT/'results_summary.csv'
    if complete and summary_path.exists():
        summary=pd.read_csv(summary_path)
        included=frame[frame.replicate.isin(complete)]
        for row in summary.itertuples():
            selected=included[(included.algorithm==row.algorithm)&(included['mode']==row.mode)]
            if row.mode=='dp':
                selected=selected[selected.target_epsilon==row.target_epsilon]
            assert len(selected)==row.replicates==len(complete)
            for metric in ('hr10','ndcg10','mrr10'):
                assert abs(selected[metric].mean()-getattr(row,'test_'+metric+'_mean'))<1e-12
                if len(selected)>1:
                    assert abs(selected[metric].std(ddof=1)-getattr(row,'test_'+metric+'_std'))<1e-12
                else:
                    assert pd.isna(getattr(row,'test_'+metric+'_std'))
    result=dict(verified_runs=len(records),planned_runs=54,complete_replicates=complete,
                all_planned_complete=len(records)==54,rank_metrics_recomputed=True,
                checksums_verified=True,cohort_verified=True,
                privacy_scope='conditional mechanism only; no end-to-end certification')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    verify()
