"""Centralized/FedAvg/FedProx on one MB-SASRec next-view benchmark."""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from recommendation_model import MultiBehaviorSASRec, per_user_loss, proximal_penalty

ALGORITHMS = ['Centralized', 'FedAvg', 'FedProx']
MODEL_KEYS = ['hidden_units', 'maxlen', 'num_blocks', 'num_heads', 'dropout_rate']


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RecommendationData:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'data_manifest.json').read_text(encoding='utf-8'))
        self.num_items = self.manifest['num_items']
        for name in ['user_ids', 'train_items', 'train_behaviors', 'trainable_rows',
                     'history_items', 'history_offsets', 'eval_users', 'client_ids',
                     'valid_items', 'valid_behaviors', 'valid_candidates', 'valid_priorities',
                     'test_items', 'test_behaviors', 'test_candidates', 'test_priorities', 'catalog']:
            setattr(self, name, np.load(self.folder / f'{name}.npy', mmap_mode='r', allow_pickle=False))

    def subset(self, config):
        rng = np.random.default_rng(config['subset_seed'])
        rows = np.array(self.trainable_rows)
        clients = np.unique(self.client_ids[rows])
        if config['max_clients']:
            clients = np.sort(rng.choice(clients, min(len(clients), config['max_clients']), replace=False))
        selected = []
        for client in clients:
            group = rows[self.client_ids[rows] == client]
            if config['users_per_client']:
                group = rng.choice(group, min(len(group), config['users_per_client']), replace=False)
            selected.append(np.sort(group))
        rows = np.sort(np.concatenate(selected))
        evaluation = np.arange(len(self.eval_users))
        if config['max_eval_users']:
            # Keep the smoke evaluation relevant to the users actually trained.
            pool = evaluation[np.isin(self.eval_users, self.user_ids[rows])]
            if len(pool) == 0:
                raise ValueError('No evaluation users in the selected training subset')
            evaluation = np.sort(rng.choice(pool, min(len(pool), config['max_eval_users']), replace=False))
        return rows, evaluation

    def batch(self, rows, rng, maxlen, device):
        windows = np.array(self.train_items[rows, -(maxlen + 1):], dtype=np.int64)
        behaviors = np.array(self.train_behaviors[rows, -(maxlen + 1):], dtype=np.int64)
        positive = windows[:, 1:]
        negative = np.zeros_like(positive)
        for idx, row in enumerate(rows):
            mask = positive[idx] != 0
            start, end = self.history_offsets[row:row + 2]
            forbidden = self.history_items[start:end]
            if len(np.unique(forbidden)) >= self.num_items:
                raise ValueError('Cannot sample a negative outside the training history')
            samples = rng.integers(1, self.num_items + 1, size=int(mask.sum()))
            bad = np.isin(samples, forbidden)
            while bad.any():
                samples[bad] = rng.integers(1, self.num_items + 1, size=int(bad.sum()))
                bad = np.isin(samples, forbidden)
            negative[idx, mask] = samples
        return tuple(torch.from_numpy(np.array(a, copy=True)).to(device) for a in
            [windows[:, :-1], behaviors[:, :-1], positive, negative])


def ranks_from_scores(scores, priorities=None):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError('Candidate scores must be a finite two-dimensional matrix')
    better = scores[:, 1:] > scores[:, [0]]
    if priorities is not None:
        priority = np.asarray(priorities)
        if priority.shape != scores.shape:
            raise ValueError('Invalid priority matrix shape')
        better |= (scores[:, 1:] == scores[:, [0]]) & (priority[:, 1:] < priority[:, [0]])
    return (1 + better.sum(axis=1)).astype(np.int32)


def metrics_from_ranks(ranks):
    ranks = np.asarray(ranks)
    hit = ranks <= 10
    return dict(hr10=float(hit.mean()), recall10=float(hit.mean()),
        ndcg10=float(np.where(hit, 1 / np.log2(ranks + 1), 0).mean()),
        mrr10=float(np.where(hit, 1 / ranks, 0).mean()),
        mean_rank=float(ranks.mean()), evaluated_users=len(ranks))


@torch.inference_mode()
def evaluate(model, data, rows, split, config, device):
    model.eval()
    ranks = []
    for offset in range(0, len(rows), config['eval_batch_size']):
        chunk = rows[offset:offset + config['eval_batch_size']]
        length = config['maxlen']
        arrays = [getattr(data, f'{split}_{name}')[chunk] for name in ['items','behaviors','candidates']]
        arrays[0], arrays[1] = arrays[0][:, -length:], arrays[1][:, -length:]
        tensors = [torch.from_numpy(np.array(a, dtype=np.int64, copy=True)).to(device) for a in arrays]
        scores = model.candidate_scores(*tensors).cpu().numpy()
        # Match ML3: strict score comparison on validation; shared priorities on test.
        priority = None if split == 'valid' else data.test_priorities[chunk]
        ranks.append(ranks_from_scores(scores, priority))
    ranks = np.concatenate(ranks)
    return metrics_from_ranks(ranks), ranks


def make_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
        betas=(0.9, 0.98), weight_decay=0)


def train_pass(model, optimizer, data, rows, rng, config, device, anchor=None):
    model.train()
    losses, examples, updates, gradient_norms = 0.0, 0, 0, []
    shuffled = rng.permutation(rows)
    for start in range(0, len(shuffled), config['batch_size']):
        batch_rows = shuffled[start:start + config['batch_size']]
        batch = data.batch(batch_rows, rng, config['maxlen'], device)
        optimizer.zero_grad(set_to_none=True)
        user_losses = per_user_loss(model, *batch)
        objective = user_losses.mean()
        if anchor is not None:
            objective = objective + proximal_penalty(model, anchor, config['mu'])
        if not torch.isfinite(objective):
            raise FloatingPointError('Non-finite training loss')
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            model.item_emb.weight[0].zero_()
            model.beh_emb.weight[0].zero_()
        losses += float(user_losses.detach().sum())
        gradient_norms.append(float(norm))
        examples += len(batch_rows)
        updates += 1
    return dict(user_loss_sum=losses, user_visits=examples, optimizer_steps=updates,
        max_preclip_gradient_norm=max(gradient_norms, default=0.0))


def aggregate_states(weighted_states):
    """Streaming weighted mean; all clients must have one compatible float state."""
    aggregate, total = None, 0
    for state, users in weighted_states:
        if users <= 0:
            raise ValueError('Client aggregation weights must be positive')
        if aggregate is None:
            aggregate = {name: value.detach().clone().mul_(users) for name, value in state.items()}
        else:
            if state.keys() != aggregate.keys():
                raise ValueError('Incompatible client states')
            for name, value in state.items():
                if value.shape != aggregate[name].shape:
                    raise ValueError('Incompatible client parameter shape')
                aggregate[name].add_(value.detach(), alpha=users)
        total += users
    if aggregate is None:
        raise ValueError('No clients to aggregate')
    for value in aggregate.values():
        value.div_(total)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite aggregated parameter')
    return aggregate


def config_fingerprint(config, manifest):
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['run_ml4_recommendation.py','recommendation_model.py']}
    payload = dict(config=config, data=manifest['files'], code=code_hashes)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def train_experiment(algorithm, seed, data, train_rows, eval_rows, config, output, device, fingerprint):
    run_dir = output / 'runs' / algorithm / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / 'metrics.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('fingerprint') != fingerprint:
            raise ValueError(f'Refusing to mix a changed configuration with old results: {run_dir}')
        for name in ['best_checkpoint.pt','test_ranks.npy','history.csv']:
            if not (run_dir / name).exists():
                raise FileNotFoundError(f'Incomplete cached result: {run_dir / name}')
            expected = result['artifact_sha256'][name]
            if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Cached result failed its checksum: {run_dir / name}')
        print(f'{algorithm} seed {seed}: verified cached run', flush=True)
        return result
    seed_all(seed)
    model = MultiBehaviorSASRec(data.num_items, **{k:config[k] for k in MODEL_KEYS}).to(device)
    optimizer = make_optimizer(model, config) if algorithm == 'Centralized' else None
    local = copy.deepcopy(model) if algorithm != 'Centralized' else None
    clients = {int(c): train_rows[data.client_ids[train_rows] == c] for c in np.unique(data.client_ids[train_rows])}
    best, best_round, start = -math.inf, None, time.perf_counter()
    histories, total_steps, total_visits = [], 0, 0
    rounds_completed = 0
    # Restart-safe round checkpoints include optimizer and PRNG state. The round
    # schedule itself is derived from (seed, round), so NumPy need not be pickled.
    latest = run_dir / 'latest_checkpoint.pt'
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=True)
        if checkpoint.get('fingerprint') != fingerprint:
            raise ValueError(f'Changed configuration cannot resume {run_dir}')
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type == 'cuda' and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
        histories = checkpoint['history']
        best, best_round = checkpoint['best'], checkpoint['best_round']
        total_steps, total_visits = checkpoint['steps'], checkpoint['visits']
        rounds_completed = len(histories)
        print(f'Resuming {algorithm} seed {seed} at round {rounds_completed + 1}', flush=True)
    for round_index in range(rounds_completed, config['rounds']):
        round_start = time.perf_counter()
        rng = np.random.default_rng(np.random.SeedSequence([seed, round_index, 1729]))
        round_stats = []
        if algorithm == 'Centralized':
            for _ in range(config['local_epochs']):
                round_stats.append(train_pass(model, optimizer, data, train_rows, rng, config, device))
        else:
            anchor = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            ordered_clients = rng.permutation(sorted(clients))

            def local_states():
                for client in ordered_clients:
                    local.load_state_dict(model.state_dict())
                    local_optimizer = make_optimizer(local, config)
                    local_rng = np.random.default_rng(np.random.SeedSequence([seed, round_index, int(client), 2718]))
                    for _ in range(config['local_epochs']):
                        round_stats.append(train_pass(local, local_optimizer, data, clients[int(client)],
                            local_rng, config, device, anchor if algorithm == 'FedProx' else None))
                    yield local.state_dict(), len(clients[int(client)])

            model.load_state_dict(aggregate_states(local_states()))
        validation, _ = evaluate(model, data, eval_rows, 'valid', config, device)
        total_steps += sum(s['optimizer_steps'] for s in round_stats)
        total_visits += sum(s['user_visits'] for s in round_stats)
        row = dict(round=round_index + 1, train_user_loss=sum(s['user_loss_sum'] for s in round_stats) /
            sum(s['user_visits'] for s in round_stats), optimizer_steps=total_steps,
            user_visits=total_visits, elapsed_round_seconds=time.perf_counter() - round_start,
            **{f'valid_{k}':v for k,v in validation.items()})
        histories.append(row)
        if validation['ndcg10'] > best:
            best, best_round = validation['ndcg10'], round_index + 1
            save_checkpoint(run_dir / 'best_checkpoint.pt',dict(model=model.state_dict(), seed=seed, algorithm=algorithm,
                round=best_round, config=config, fingerprint=fingerprint))
        save_checkpoint(latest,dict(model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            history=histories, best=best, best_round=best_round, steps=total_steps, visits=total_visits,
            fingerprint=fingerprint))
        pd.DataFrame(histories).to_csv(run_dir / 'history.csv', index=False)
        print(f'{algorithm} seed={seed} round={round_index + 1}/{config["rounds"]} '
            f'loss={row["train_user_loss"]:.5f} valid HR@10={validation["hr10"]:.5f} '
            f'NDCG@10={validation["ndcg10"]:.5f} ({row["elapsed_round_seconds"]:.1f}s)', flush=True)
    best_checkpoint = torch.load(run_dir / 'best_checkpoint.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint['model'])
    test_metrics, test_ranks = evaluate(model, data, eval_rows, 'test', config, device)
    np.save(run_dir / 'test_ranks.npy', test_ranks)
    result = dict(algorithm=algorithm, seed=seed, scope=config['scope'], best_round=best_round,
        best_valid_ndcg10=best, training_users=len(train_rows), clients=len(clients),
        total_optimizer_steps=total_steps, total_user_visits=total_visits,
        test_metrics=test_metrics, elapsed_this_invocation_seconds=time.perf_counter() - start,
        elapsed_saved_rounds_seconds=sum(h['elapsed_round_seconds'] for h in histories),
        fingerprint=fingerprint,device=str(device),torch=str(torch.__version__),
        python=platform.python_version(),artifact_sha256={name:hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
            for name in ['best_checkpoint.pt','test_ranks.npy','history.csv']})
    save_json(result_path, result)
    return result


def render_outputs(results, output, config, data, train_rows, eval_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame([{**{k:v for k,v in r.items() if k != 'test_metrics'},
        **{f'test_{k}':v for k,v in r['test_metrics'].items()}} for r in results])
    frame.to_csv(output / 'results_per_run.csv', index=False)
    summary = frame.groupby('algorithm', sort=False).agg(
        seeds=('seed','count'), hr10_mean=('test_hr10','mean'), hr10_std=('test_hr10','std'),
        ndcg10_mean=('test_ndcg10','mean'), ndcg10_std=('test_ndcg10','std'),
        mrr10_mean=('test_mrr10','mean'), mrr10_std=('test_mrr10','std')).reset_index()
    summary.to_csv(output / 'results_summary.csv', index=False)
    title = 'SMOKE TEST ONLY' if config['scope'] != 'full_dataset' else 'Full Dataset Experiment'
    colors = ['#24745f','#c04e45','#536cb0']
    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    for axis, metric, label in zip(axes, ['hr10','ndcg10'], ['HR@10','NDCG@10']):
        axis.bar(summary.algorithm, summary[f'{metric}_mean'],
            yerr=summary[f'{metric}_std'].fillna(0), capsize=4, color=colors)
        axis.set_ylabel(label)
        axis.set_ylim(0, min(1, max(0.2,float(summary[f'{metric}_mean'].max()) * 1.25)))
        axis.grid(axis='y', alpha=.2)
    fig.suptitle(f'{title} | MB-SASRec | 101 candidates')
    fig.savefig(output / 'model_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for index, algorithm in enumerate(ALGORITHMS):
        histories = pd.concat([pd.read_csv(output / 'runs' / algorithm / f'seed_{r["seed"]}' / 'history.csv')
            for r in results if r['algorithm'] == algorithm])
        means = histories.groupby('round').mean(numeric_only=True)
        axes[0].plot(means.index, means.train_user_loss, label=algorithm, color=colors[index])
        axes[1].plot(means.index, means.valid_ndcg10, label=algorithm, color=colors[index])
    for axis, label in zip(axes,['Training User Loss','Validation NDCG@10']):
        axis.set_xlabel('Round / Centralized pass')
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(alpha=.2)
    fig.suptitle(title)
    fig.savefig(output / 'training_curves.png',dpi=160)
    plt.close(fig)
    membership = pd.DataFrame(dict(visitorid=data.user_ids[train_rows],client_id=data.client_ids[train_rows]))
    membership.to_csv(output / 'training_client_membership.csv.gz', index=False, compression='gzip')
    client_counts = membership.groupby('client_id').size()
    fig, axis = plt.subplots(figsize=(10,3.5), constrained_layout=True)
    axis.bar([str(c) for c in client_counts.index], client_counts.values, color='#24745f')
    axis.set_xlabel('Client ID'); axis.set_ylabel('Training users')
    axis.set_title(f'{title} | {len(client_counts)} active clients')
    if len(client_counts) > 15:
        axis.tick_params(axis='x',labelsize=7,rotation=90)
    fig.savefig(output / 'client_distribution.png',dpi=160)
    plt.close(fig)
    np.save(output / 'evaluated_users.npy', data.eval_users[eval_rows])
    rows = '\n'.join(f'| {r.algorithm} | {r.seeds} | {r.hr10_mean:.6f} | {r.ndcg10_mean:.6f} |'
        for r in summary.itertuples())
    caution = ('**CHỈ CHẠY THỬ KỸ THUẬT: không dùng bảng này làm kết quả cuối hoặc so sánh số trực tiếp với ML3.**'
        if config['scope'] != 'full_dataset' else 'Kết quả trên đầy đủ bộ dữ liệu; diễn giải theo cấu hình và số seed được công bố.')
    report = f'''# ML4 Gợi ý sản phẩm với Federated Learning

{caution}

## 4.3.5.1. Thiết lập client

MB-SASRec (add) dùng view/addtocart/transaction; {len(train_rows):,} khách hàng train,
{len(client_counts)} client hoạt động, {len(eval_rows):,} khách hàng đánh giá.
Mỗi khách hàng thuộc đúng một client theo hash ổn định. Vòng lặp FL là mô phỏng
trên một máy, không kiểm chứng mạng, thiết bị phân tán hay khả năng cách ly dữ liệu.
Chia leave-last-two-views-out theo khách hàng; không phải mốc thời gian toàn cục.
Test xếp hạng một sản phẩm view mục tiêu giữa 100 sản phẩm lấy mẫu, không phải toàn catalog.

## 4.3.5.2. Centralized Training

Centralized dùng cùng MB-SASRec, loss trung bình theo khách hàng, dữ liệu và ngân sách
lượt đi qua khách hàng như các phương án FL. Adam giữ trạng thái qua các round.

## 4.3.5.3. FedAvg

Mọi client được chọn trong thí nghiệm đều tham gia mỗi round. Client bắt đầu từ cùng
tham số toàn cục, Adam cục bộ khởi tạo lại mỗi client/round; server trung bình tham số
theo số khách hàng đủ điều kiện train. Không tổng hợp theo số phiên hoặc số sự kiện.

## 4.3.5.4. FedProx

Giống FedAvg, thêm μ/2 nhân bình phương khoảng cách tới tham số đầu round trên mọi
tham số huấn luyện, kể cả embedding và bias. μ = {config['mu']}. Đây không phải DP-SGD.

## 4.3.5.5. So sánh kết quả

| Thuật toán | Số seed | HR@10 trung bình | NDCG@10 trung bình |
| --- | --- | --- | --- |
{rows}

Độ lệch chuẩn mẫu, chỉ số từng seed, số bước tối ưu và lượt khách hàng được lưu trong CSV.
Một seed không có độ lệch chuẩn mẫu xác định; thanh sai số 0 khi đó không chứng minh ổn định.
Validation chỉ dùng chọn checkpoint theo NDCG@10; test chỉ được tính sau chọn checkpoint.
Không dùng kết quả test chọn μ, kiến trúc hoặc số round.

## 4.3.5.6. Phạm vi trả lời RQ4

{'Chạy thử xác nhận đường đi dữ liệu, huấn luyện, tổng hợp và đánh giá hoạt động; chưa trả lời RQ4 bằng thực nghiệm đầy đủ.' if config['scope'] != 'full_dataset' else 'Các kết quả trên cho phép so sánh theo cấu hình đã chạy; không suy ra FL luôn tốt hơn Centralized hoặc FedProx luôn tốt hơn FedAvg.'}
Client cân bằng về số lượng không chứng minh phân phối hành vi đồng nhất. Chưa có
kiểm định non-IID có kiểm soát hay đo chi phí truyền thông thực tế. HR@10 bằng Recall@10
chỉ vì mỗi hàng có đúng một positive. Các kết quả dự đoán mua trước đây là bài toán khác.

## Chuẩn bị cho ML5

Một hàng train hiện đại diện một khách hàng với cửa sổ hành vi giới hạn; loss được chuẩn
hóa theo khách hàng. Điều này hỗ trợ thiết kế thí nghiệm tiếp theo nhưng không tự tạo
user-level DP. Cần triển khai lấy mẫu, cắt gradient theo đơn vị đã chọn, Gaussian noise,
accounting, adjacency và chính sách phát hành; catalog, roster, preprocessing và metric
hiện không riêng tư. Không dùng checkpoint huấn luyện không DP làm khởi tạo công khai
rồi tuyên bố toàn bộ dữ liệu đã được bảo vệ bởi DP.

## Nguồn phương pháp

- FedAvg: https://proceedings.mlr.press/v54/mcmahan17a.html
- FedProx: https://arxiv.org/abs/1812.06127
- Kiến trúc MB-SASRec: code ML3 do nhóm cung cấp (Big (1).py).
'''
    (output / 'ML4_Recommendation_Report.md').write_text(report,encoding='utf-8')
    return summary


def validate_config(config):
    for name in ['rounds','local_epochs','batch_size','eval_batch_size','hidden_units','maxlen','num_blocks','num_heads','threads']:
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    for name in ['max_clients','users_per_client','max_eval_users']:
        if not isinstance(config[name], int) or config[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    for name in ['learning_rate','gradient_clip']:
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not math.isfinite(config['mu']) or config['mu'] < 0:
        raise ValueError('mu must be finite and nonnegative')
    if not 0 <= config['dropout_rate'] < 1 or config['hidden_units'] % config['num_heads']:
        raise ValueError('Invalid model dimensions/dropout')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Provide at least one unique seed')
    if config['scope'] == 'full_dataset' and any(config[k] for k in ['max_clients','users_per_client','max_eval_users']):
        raise ValueError('A subset cannot be labelled full_dataset')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument('--data-dir',type=Path,default=base / 'data')
    parser.add_argument('--profile',choices=['smoke','full'],default='smoke')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args(argv)
    config_file = args.config or base / 'configs' / f'{args.profile}.json'
    config = json.loads(config_file.read_text(encoding='utf-8'))
    validate_config(config)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available in the active kernel')
    device = torch.device('cuda' if args.device == 'cuda' or
        (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
    data = RecommendationData(args.data_dir)
    from prepare_recommendation_data import sha256
    for name, expected in data.manifest['files'].items():
        if sha256(args.data_dir / name) != expected:
            raise ValueError(f'Prepared data failed its checksum: {name}')
    if config['maxlen'] > data.manifest['maxlen']:
        raise ValueError('Model maxlen exceeds the prepared sequence length')
    train_rows, eval_rows = data.subset(config)
    fingerprint = config_fingerprint(config, data.manifest)
    output = args.output_dir or base / f'results_{args.profile}'
    output.mkdir(parents=True, exist_ok=True)
    experiment = output / 'experiment.json'
    if experiment.exists() and json.loads(experiment.read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
        raise ValueError('Use a different output folder for a changed experiment')
    if not experiment.exists():
        save_json(experiment,dict(config=config, fingerprint=fingerprint, device=str(device),
            python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,
            training_users=len(train_rows),evaluation_users=len(eval_rows),
            active_clients=len(np.unique(data.client_ids[train_rows])), privacy='NOT DP'))
    print(f'Scope={config["scope"]}; device={device}; train users={len(train_rows):,}; '
        f'eval users={len(eval_rows):,}; model=MB-SASRec (add)', flush=True)
    if device.type == 'cpu' and config['scope'] == 'full_dataset':
        print('Full training on CPU can take substantial time; completed rounds are resumable.',flush=True)
    results = [train_experiment(algorithm, seed, data, train_rows, eval_rows, config, output, device, fingerprint)
        for seed in config['seeds'] for algorithm in ALGORITHMS]
    summary = render_outputs(results,output,config,data,train_rows,eval_rows)
    print(summary.to_string(index=False),flush=True)
    print(f'Outputs: {output.resolve()}',flush=True)
    return output


if __name__ == '__main__':
    main()
