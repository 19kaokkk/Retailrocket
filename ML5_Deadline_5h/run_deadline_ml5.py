"""Matched ML5 subset experiment with incremental outputs and a time guard.

Shared data and DP primitives remain read-only in ../ML5_Recommendation.
Accounting is conditional on fixed roster/catalog/grouping, NOT end-to-end DP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / 'ML5_Recommendation'
sys.path.insert(0, str(SHARED))
import numpy as np
import pandas as pd
import torch
import opacus
import run_ml5_recommendation as shared
from privacy_mechanism import fresh_randomness, poisson_users, apply_proximal_gradient
from recommendation_model import per_user_loss
from run_ml4_recommendation import RecommendationData, save_json, seed_all
from prepare_recommendation_data import sha256

TZ = timezone(timedelta(hours=7))
ALGORITHMS = ['Centralized', 'FedAvg', 'FedProx']
ORIGINAL_TRAIN_LOCAL = shared.train_local


def now():
    return datetime.now(TZ)


def batched_no_dp(model, data, roster, slots, spec, config, device, anchor=None):
    """Same Poisson sum/qN objective; only no-DP gradients can be batched."""
    model.train()
    sampling_rng, _ = fresh_randomness(device)
    negative_rng = np.random.default_rng()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    for _ in range(spec['steps_per_round']):
        selected = slots[poisson_users(sampling_rng, len(slots), spec['q'])]
        usable = selected[(roster['rows'][selected] >= 0) & (~roster['absent'][selected])]
        optimizer.zero_grad(set_to_none=True)
        if len(usable):
            batch = data.batch(roster['rows'][usable], negative_rng, config['maxlen'], device)
            loss = per_user_loss(model, *batch).sum() / spec['expected_batch_users']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite no-DP loss')
            loss.backward()
        else:
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
        apply_proximal_gradient(model, anchor, config['mu'] if anchor is not None else 0)
        optimizer.step()
        with torch.no_grad():
            model.item_emb.weight[0].zero_()
            model.beh_emb.weight[0].zero_()


def selected_train_local(model, data, roster, slots, spec, config, mode, sigma,
                         accountant, device, anchor=None):
    if mode == 'no_dp':
        return batched_no_dp(model, data, roster, slots, spec, config, device, anchor)
    return ORIGINAL_TRAIN_LOCAL(model, data, roster, slots, spec, config, mode,
                                sigma, accountant, device, anchor)


def prepare(config, output):
    data = RecommendationData(SHARED / 'data')
    for name, digest in data.manifest['files'].items():
        if sha256(SHARED / 'data' / name) != digest:
            raise ValueError('Shared data changed: ' + name)
    train_rows = np.array(data.trainable_rows)
    lengths = np.count_nonzero(data.train_items[train_rows], axis=1)
    eligible = train_rows[lengths >= config['selection_min_train_events']]
    available = np.unique(data.client_ids[eligible])
    enough = [int(c) for c in available if np.sum(data.client_ids[eligible] == c)
              >= config['selection_users_per_client']]
    if len(enough) < config['selection_clients']:
        raise ValueError('Not enough clients for the predeclared cohort')
    rng = np.random.default_rng(config['subset_seed'])
    clients = np.sort(rng.choice(enough, config['selection_clients'], replace=False))
    selected = np.sort(np.concatenate([rng.choice(eligible[data.client_ids[eligible] == c],
        config['selection_users_per_client'], replace=False) for c in clients]))
    users = np.array(data.user_ids[selected], dtype=np.int64)
    groups = np.array(data.client_ids[selected], dtype=np.int16)
    eval_rows = np.flatnonzero(np.isin(data.eval_users, users))
    if not len(eval_rows):
        raise ValueError('Empty evaluation cohort')
    roster = dict(users=users, rows=selected, groups=groups,
                  absent=np.zeros(len(users), dtype=bool), eval_rows=eval_rows,
                  provenance='training-only active-user filter and random balanced client sample; NOT a public/DP-derived roster')
    model_schedules = shared.schedules_for(roster, config)
    code_files = sorted(p for p in SHARED.glob('*.py') if not p.name.startswith('test_'))
    dependency_hashes = {str(p): sha256(p) for p in code_files}
    payload = dict(config=config, data=data.manifest['files'],
        shared_code=dependency_hashes, runner_sha256=sha256(Path(__file__)),
        roster_sha256=hashlib.sha256(users.tobytes()+groups.tobytes()).hexdigest(),
        evaluation_sha256=hashlib.sha256(np.array(data.eval_users[eval_rows]).tobytes()).hexdigest())
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    experiment_file = output / 'experiment.json'
    if experiment_file.exists():
        if json.loads(experiment_file.read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
            raise ValueError('Changed code/data/config: use a NEW output directory')
    else:
        save_json(experiment_file, dict(config=config, fingerprint=fingerprint,
            roster_users=len(users), active_clients=len(clients), evaluation_users=len(eval_rows),
            schedules=model_schedules, selection_training_only=True, eligible_training_users=len(eligible),
            roster_provenance=roster['provenance'], selected_clients=clients.tolist(),
            python=platform.python_version(), torch=str(torch.__version__), opacus=opacus.__version__,
            device='cpu', shared_code_sha256=dependency_hashes,
            no_dp_optimization='batched sum of per-user mean losses / expected Poisson batch',
            dp_implementation='unchanged streaming user gradients and Opacus DPOptimizer',
            assumptions=shared.DISCLAIMERS, full_dataset=False))
        np.save(output / 'fixed_training_users.npy', users)
        np.save(output / 'evaluated_users.npy', np.array(data.eval_users[eval_rows]))
        pd.DataFrame(dict(visitorid=users, client_id=groups, train_row=selected,
            train_events=np.count_nonzero(data.train_items[selected], axis=1))).to_csv(
                output / 'cohort_metadata.csv', index=False)
    return data, roster, model_schedules, fingerprint


def raw_table(results):
    return pd.DataFrame([{**{k:v for k,v in r.items() if k not in
        ('validation', 'test', 'ledger', 'artifact_sha256')},
        **{'val_'+k:v for k,v in r['validation'].items()},
        **{'test_'+k:v for k,v in r['test'].items()}} for r in results])


def save_incremental(results, output, config, state, final=False):
    frame = raw_table(results)
    frame.to_csv(output / 'results_per_run_all_completed.csv', index=False)
    matrix_size = len(ALGORITHMS) * (2+len(config['epsilon_targets']))
    complete = [rep for rep, group in frame.groupby('replicate') if len(group) == matrix_size] if len(frame) else []
    save_json(output / 'progress.json', dict(**state, completed_runs=len(results),
        planned_runs=matrix_size*config['replicates'], complete_replicates=complete,
        target_replicates=config['replicates'], last_updated=now().isoformat(),
        all_planned_complete=len(results)==matrix_size*config['replicates'],
        final=final, scope=config['scope'], whole_package_dp_certified=False))
    if not complete:
        return
    selected = [r for r in results if r['replicate'] in complete]
    summary, joint = shared.summarize(selected, output, config)
    # Joint accounting includes ALL reservations in this output, not just complete matrices.
    if final or len(results) % matrix_size == 0:
        write_outputs(summary, selected, joint, config, output, complete, state)


def write_outputs(summary, results, joint, config, output, complete, state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'Centralized':'#275580', 'FedAvg':'#ce842a', 'FedProx':'#2c8556'}
    for metric in ('hr10', 'ndcg10', 'mrr10'):
        fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
        for alg in ALGORITHMS:
            group = summary[(summary.algorithm==alg) & (summary['mode']=='dp')].sort_values('epsilon_upper_bound')
            baseline = summary[(summary.algorithm==alg) & (summary['mode']=='no_dp')].iloc[0]
            errors = group['test_'+metric+'_std'] if len(complete)>1 else None
            ax.errorbar(group.epsilon_upper_bound, group['test_'+metric+'_mean'], yerr=errors,
                        marker='o', capsize=3, color=colors[alg], label=alg)
            ax.axhline(baseline['test_'+metric+'_mean'], ls='--', color=colors[alg], alpha=.45)
        ax.set(xlabel='Accounted epsilon upper bound', ylabel=metric.upper().replace('10','@10'),
               title=f'Active-user subset | {len(complete)} replicates | dashed: matched no-DP')
        ax.grid(alpha=.2); ax.legend()
        fig.savefig(output / ('privacy_utility_'+metric+'.png'), dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
    comparison = []
    for alg in ALGORITHMS:
        group = summary[(summary.algorithm==alg) & (summary['mode']=='dp')].sort_values('target_epsilon')
        ax.plot(group.target_epsilon, group.sigma, marker='o', color=colors[alg], label=alg)
    ax.set(xlabel='Nominal epsilon target', ylabel='Noise multiplier sigma', title='Fixed subset schedules; read sigma with q and steps')
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output/'noise_levels.png', dpi=160); plt.close(fig)
    for eps in config['epsilon_targets']:
        group = summary[(summary['mode']=='dp') & (summary.target_epsilon==eps)].set_index('algorithm')
        comparison.append(dict(target_epsilon=eps, ndcg_fedavg=group.loc['FedAvg','test_ndcg10_mean'],
            ndcg_fedprox=group.loc['FedProx','test_ndcg10_mean'],
            delta_prox_minus_avg=group.loc['FedProx','test_ndcg10_mean']-group.loc['FedAvg','test_ndcg10_mean']))
    pd.DataFrame(comparison).to_csv(output/'fedprox_vs_fedavg.csv', index=False)
    first = results[0]
    table_rows = []
    def num(v, digits=6):
        return 'NA' if pd.isna(v) else f'{v:.{digits}f}'
    for r in summary.itertuples():
        table_rows.append(f'| {r.algorithm} | {r.mode} | {num(r.target_epsilon,0)} | {num(r.sigma,4)} | '
            f'{num(r.epsilon_upper_bound,4)} | {num(r.test_hr10_mean)} | {num(r.test_ndcg10_mean)} | '
            f'{num(r.test_ndcg10_std)} | {num(r.test_mrr10_mean)} | {num(r.ndcg10_retention_pct,2)} |')
    report = f'''# 4.3.6 ML5 Privacy Utility Trade off

## Phạm vi thực nghiệm

Thực nghiệm giới hạn thời gian trên một tập con cố định gồm {first['roster_users']} người dùng,
{first['clients']} client và {first['evaluation_users']} người dùng đánh giá. Đây KHÔNG phải kết quả full.
Chỉ chọn người dùng có ít nhất {config['selection_min_train_events']} sự kiện trong cửa sổ train;
lấy ngẫu nhiên {config['selection_users_per_client']} người dùng/client ở {config['selection_clients']} client,
seed chọn mẫu {config['subset_seed']}. Không chọn mẫu dựa vào điểm test hoặc mức cải thiện của DP.
Mọi phương án dùng chung cohort, ứng viên và kiến trúc. Chỉ các replicate đã hoàn thành toàn bộ ma trận
được đưa vào bảng: {complete}; kế hoạch là {config['replicates']} replicate. Kết quả từng lượt hoàn tất
khác vẫn được lưu ở results_per_run_all_completed.csv, không lược bỏ theo chất lượng.

## 4.3.6.1 DP-SGD

MB-SASRec dùng hidden 50, 2 block, 1 head, cửa sổ tối đa 50 vị trí và dropout 0.5,
danh mục train 114.800 sản phẩm. Train dùng view, addtocart, transaction; test dự đoán next-view,
1 positive và 100 negatives cố định. Giữ cách tách theo từng người dùng như benchmark ML3/ML4,
không phải chia thời gian toàn hệ thống. Khởi tạo ngẫu nhiên, không warm-start bằng mô hình không DP.

Mỗi lượt chạy {config['rounds']} vòng, SGD lr={config['learning_rate']}; batch mục tiêu Centralized
{config['central_batch_users']} và FL {config['local_batch_users']}. Poisson theo slot người dùng;
gradient của toàn bộ tham số theo mỗi người được gộp rồi cắt norm C={config['clip_norm']}.
Opacus DPOptimizer cộng Gaussian std sigma*C vào tổng đã cắt, chia qN kỳ vọng, không chia batch thực.
Batch rỗng vẫn cộng nhiễu và được accountant tính. Tăng tốc theo batch chỉ áp dụng cho no_dp;
clip_only và dp giữ nguyên cơ chế gradient riêng từng người.

## 4.3.6.2 Các mức độ nhiễu

Epsilon mục tiêu {config['epsilon_targets']}, delta={config['delta']:g}. Sigma được hiệu chỉnh bằng
RDP accountant theo q và số bước định trước, sau đó tính lại ngân sách upper bound từng client.
Không gọi sigma là epsilon; no_dp/clip_only không có chứng nhận DP hữu hạn, không phải epsilon=0.

| Thuật toán | Chế độ | epsilon mục tiêu | sigma | epsilon upper bound | HR@10 TB | NDCG@10 TB | SD NDCG@10 | MRR@10 TB | Giữ NDCG (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(table_rows)}

## 4.3.6.3 Privacy Utility Trade off

HR@10 bằng Recall@10 trong giao thức một positive. NDCG@10 và MRR@10 được đánh giá trên 101 ứng viên,
không phải toàn danh mục. Trung bình/SD mẫu tính qua {len(complete)} replicate hoàn chỉnh; nếu chỉ một
replicate thì SD không xác định. Đối chứng no_dp và clip_only được train lại trên cùng tập con và lịch SGD,
không lấy số ML4 full/Adam làm baseline. Tỷ lệ giữ lại là NDCG_DP/NDCG_noDP của cùng thuật toán;
NA nếu mẫu số bằng 0. Đường cong là thống kê mô tả, không bảo đảm đơn điệu, không lựa epsilon bằng test.
Checkpoint cuối theo lịch cố định được đánh giá, không chọn best bằng validation/test.

## 4.3.6.4 FedProx cộng DP-SGD

FedAvg và FedProx dùng cùng cơ chế; proximal mu={config['mu']} được thêm sau clipping/nhiễu/chia qN,
chỉ dựa vào mô hình đầu vòng và trạng thái hiện tại. Server tổng hợp theo số slot cố định;
accountant không reset qua các vòng. Chênh lệch theo epsilon được lưu ở fedprox_vs_fedavg.csv.
Không suy diễn một chênh lệch dương nhỏ thành bằng chứng FedProx luôn tốt hơn, và không ghép tùy tiện
replicate cùng số thành kiểm định cặp vì các luồng nhiễu/lấy mẫu độc lập.

## 4.3.6.5 Phạm vi trả lời RQ5

Các số liệu trên mô tả đánh đổi riêng tư và chất lượng CHỈ trong nhóm khách hàng có lịch sử train đủ dài
và cấu hình giới hạn này. Không thể coi đây là kết quả của 115.150 người dùng full, so trực tiếp với
ML4 full, khẳng định điểm epsilon tối ưu hay hứa chất lượng cao khi triển khai. Phân tích quản trị cần
đọc kèm quy mô và các giới hạn này, kể cả khi hiệu năng DP thấp hoặc xu hướng có dao động.

Accountant chỉ áp dụng cho cơ chế huấn luyện có điều kiện theo roster/catalog/grouping cố định.
Việc lọc active users, chọn cohort, catalog, tiền xử lý, metric, ranks và metadata CHƯA DP.
PRNG chưa được kiểm toán mật mã. Không có chứng nhận user-level DP đầu cuối hoặc client-level/local DP.
Nếu phát hành chung mọi mô hình DP trong thư mục này, ngân sách cộng gộp bảo thủ là
{joint['joint_model_mechanism_epsilon_upper_bound']:.6f} tại delta={config['delta']:g},
gồm {joint['dp_runs_accounted']} lượt DP có reservation. Những lần chạy ở thư mục khác phải cộng thêm.
Xóa hoặc đổi thư mục kết quả không đặt lại ngân sách đã tiêu trên cùng dữ liệu.

Nguồn: kết quả thực nghiệm đã lưu; Opacus DPOptimizer https://opacus.ai/api/optim/optimizers.html
'''
    (output/'ML5_Subset_Report.md').write_text(report, encoding='utf-8')
    with zipfile.ZipFile(output/'ML5_Subset_Results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in output.iterdir():
            if p.is_file() and p.suffix in ('.csv','.json','.md','.png','.npy'):
                archive.write(p, 'results/'+p.name)
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ('.py','.json','.md','.ipynb'):
                archive.write(p, p.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--confirm-full-stopped', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=HERE/'results')
    parser.add_argument('--hours', type=float, default=4.0)
    args = parser.parse_args(argv)
    if not args.prepare_only and not args.confirm_full_stopped:
        parser.error('Stop the old full run first, then pass --confirm-full-stopped')
    if not 0 < args.hours <= 4.25:
        parser.error('Time budget must be in (0, 4.25] hours')
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    lock_handle = open(output/'runner.lock', 'a+b')
    import msvcrt
    lock_handle.seek(0); lock_handle.write(b'1'); lock_handle.flush(); lock_handle.seek(0)
    try:
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit('Another subset runner is already using this directory')
    config = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    shared.validate_config(config)
    torch.set_num_threads(config['threads'])
    data, roster, specs, fingerprint = prepare(config, output)
    print(f'SUBSET ONLY | train={len(roster["users"])} | eval={len(roster["eval_rows"])} | clients={len(specs["FedAvg"])}', flush=True)
    print('NOT full. NOT end-to-end DP. No unchanged original full results are reused.', flush=True)
    if args.prepare_only:
        print('PREPARED: training not started', flush=True)
        return
    start = time.perf_counter()
    hard_stop = now()+timedelta(hours=args.hours)
    state = dict(pid=os.getpid(), started_at=now().isoformat(), time_guard_at=hard_stop.isoformat(),
                 status='running', current_run=None)
    shared.train_local = selected_train_local
    results = []
    modes = [('no_dp',None), ('clip_only',None)] + [('dp',float(e)) for e in config['epsilon_targets']]
    try:
        for replicate in range(1, config['replicates']+1):
            for mode, target in modes:
                for alg in ALGORITHMS:
                    label = f'epsilon_{target:g}' if mode=='dp' else mode
                    cached = output/'runs'/alg/label/f'replicate_{replicate}'/'metrics.json'
                    # Leave enough time for a conservative run estimate and exporting outputs.
                    if not cached.exists() and now()+timedelta(minutes=7) >= hard_stop:
                        state['status']='time_guard_reached'
                        save_incremental(results,output,config,state,final=True)
                        print('TIME GUARD: retained all completed runs; only complete replicate matrices summarized', flush=True)
                        return
                    sigma = shared.calibrate_noise(target,config['delta'],
                        max(v['q'] for v in specs[alg].values()),
                        max(v['planned_steps'] for v in specs[alg].values())) if mode=='dp' else 0.
                    state['current_run']=dict(algorithm=alg, mode=mode, target_epsilon=target, replicate=replicate)
                    save_incremental(results,output,config,state)
                    result = shared.train_run(alg,mode,target,replicate,sigma,data,roster,
                        specs[alg],config,output,torch.device('cpu'),fingerprint)
                    results.append(result)
                    save_incremental(results,output,config,state)
                    print(f'PROGRESS {len(results)}/54 | elapsed={(time.perf_counter()-start)/60:.1f} min', flush=True)
        state['status']='complete'; state['current_run']=None
        save_incremental(results,output,config,state,final=True)
        print('COMPLETE: 54 matched runs. Results: '+str(output), flush=True)
    except BaseException as error:
        state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        state['error']=type(error).__name__+': '+str(error)
        save_incremental(results,output,config,state,final=True)
        raise
    finally:
        shared.train_local = ORIGINAL_TRAIN_LOCAL
        lock_handle.close()


if __name__ == '__main__':
    main()
