"""Build multi-behavior sequences while checking the uploaded ML3 benchmark."""
import argparse
import hashlib
import io
import json
import pickle
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

BEHAVIORS = {'view': 1, 'addtocart': 2, 'transaction': 3}
EVAL_FILES = ['eval_users_shared.npy', 'valid_candidates_shared.npy',
    'test_candidates_shared.npy', 'tie_priorities_valid_shared.npy',
    'tie_priorities_test_shared.npy']
SAS_FILES = ['metadata.json', 'item_to_idx.pkl', 'train_sequences_encoded.pkl',
    'trainable_users.npy', 'eval_users.npy', 'eval_valid_seq.npy',
    'eval_test_seq.npy', 'eval_valid_candidates.npy', 'eval_test_candidates.npy']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip_names(infos):
    total = 0
    seen = set()
    for info in infos:
        name = info.filename.replace('\\', '/')
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name or '\x00' in name:
            raise ValueError(f'Unsafe archive path: {name}')
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f'Symlink in archive: {name}')
        if name in seen or info.flag_bits & 1:
            raise ValueError(f'Duplicate or encrypted entry: {name}')
        seen.add(name)
        total += info.file_size
        if total > 4 * 1024**3:
            raise ValueError('Archive exceeds the 4 GiB uncompressed safety limit')


def extract_inputs(archive_dir, dest):
    """Extract only required data; never execute uploaded code or load checkpoints."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    required = {f'evaluation/{name}' for name in EVAL_FILES}
    required |= {f'sasrec_shared/{name}' for name in SAS_FILES}
    required.add('archive/clean_events_final.csv')
    found = set()
    manifest = []
    for path in sorted(Path(archive_dir).glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = {i.filename for i in infos}
            if not names.intersection(required) and not any(n.startswith('results_shared/') for n in names):
                continue
            validate_zip_names(infos)
            selected = []
            for info in infos:
                name = info.filename
                keep = name in required or (name.startswith('results_shared/') and name.endswith(('.json', '.csv')))
                keep |= name.startswith('data_checkpoint/') and name.endswith('.json')
                if not keep or info.is_dir():
                    continue
                if name in found:
                    raise ValueError(f'The input archives contain ambiguous duplicates: {name}')
                target = (dest / name).resolve()
                if not target.is_relative_to(dest):
                    raise ValueError(f'Extraction escaped destination: {name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with archive.open(info) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output)
                # Compare extracted content to the archive, including on repeated runs.
                with archive.open(info) as source:
                    expected = hashlib.file_digest(source, 'sha256').hexdigest()
                if sha256(target) != expected:
                    raise ValueError(f'Existing extracted input differs from archive: {name}')
                found.add(name)
                selected.append(name)
            manifest.append(dict(archive=path.name, sha256=sha256(path), extracted=selected))
    if required - found:
        raise FileNotFoundError(f'Missing archive entries: {sorted(required - found)}')
    (dest / 'archive_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return dest


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'Global objects are not permitted: {module}.{name}')


def load_plain_pickle(path):
    return DataOnlyUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def client_bucket(user, count=50):
    # Match the stable assignment from the earlier session-level pipeline.
    return int(hashlib.md5(str(int(user)).encode('utf-8')).hexdigest()[:12], 16) % count


def pad(sequence, length, dtype=np.int32):
    result = np.zeros(length, dtype=dtype)
    values = np.asarray(sequence)[-length:]
    if len(values):
        result[-len(values):] = values
    return result


def build_data(inputs, output, num_clients=50):
    inputs, output = Path(inputs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sas, evaluation = inputs / 'sasrec_shared', inputs / 'evaluation'
    reference = load_plain_pickle(sas / 'train_sequences_encoded.pkl')
    mapping = load_plain_pickle(sas / 'item_to_idx.pkl')
    metadata = json.loads((sas / 'metadata.json').read_text(encoding='utf-8'))
    length = int(metadata['maxlen'])
    catalog = np.array(sorted(mapping), dtype=np.int64)
    assert all(mapping[int(item)] == j + 1 for j, item in enumerate(catalog))
    eval_users = np.load(evaluation / EVAL_FILES[0], allow_pickle=False)
    assert np.array_equal(eval_users, np.load(sas / 'eval_users.npy', allow_pickle=False))
    assert len(eval_users) == metadata['num_eval_users']
    assert len(reference) == metadata['num_train_users']
    candidates = {}
    reference_valid = np.load(sas / 'eval_valid_seq.npy', mmap_mode='r', allow_pickle=False)
    reference_test = np.load(sas / 'eval_test_seq.npy', mmap_mode='r', allow_pickle=False)
    for split in ['valid', 'test']:
        raw = np.load(evaluation / f'{split}_candidates_shared.npy', allow_pickle=False)
        encoded = np.load(sas / f'eval_{split}_candidates.npy', allow_pickle=False)
        positions = np.searchsorted(catalog, raw)
        assert np.all(positions < len(catalog)) and np.array_equal(catalog[positions], raw)
        assert np.array_equal(positions + 1, encoded)
        assert encoded.shape == (len(eval_users), 101)
        assert np.all(np.diff(np.sort(encoded, axis=1), axis=1) > 0)
        priority = np.load(evaluation / f'tie_priorities_{split}_shared.npy', allow_pickle=False)
        assert priority.shape == encoded.shape
        # Priorities must form a permutation, not depend on the true target score.
        assert np.all(np.diff(np.sort(priority, axis=1), axis=1) > 0)
        candidates[split] = encoded
        np.save(output / f'{split}_candidates.npy', encoded)
        np.save(output / f'{split}_priorities.npy', priority)

    events = pd.read_csv(inputs / 'archive/clean_events_final.csv',
        usecols=['timestamp', 'visitorid', 'itemid', 'event'])
    source_rows = len(events)
    source_counts = {k: int(v) for k, v in events.event.value_counts().items()}
    if events.isna().any().any():
        raise ValueError('Null input values in required event columns')
    events = events[events.visitorid.isin(reference) & events.event.isin(BEHAVIORS)].copy()
    # mergesort preserves the original file order when two timestamps are equal.
    events = events.sort_values(['visitorid', 'timestamp'], kind='mergesort').reset_index(drop=True)
    items = events.itemid.map(mapping).fillna(0).to_numpy(dtype=np.int32)
    behaviors = events.event.map(BEHAVIORS).to_numpy(dtype=np.int8)
    times = events.timestamp.to_numpy()
    user_ids, starts = np.unique(events.visitorid.to_numpy(), return_index=True)
    bounds = np.append(starts, len(events))
    assert set(map(int, user_ids)) == set(reference)
    eval_row = {int(u): row for row, u in enumerate(eval_users)}
    train_items = np.zeros((len(user_ids), length + 1), dtype=np.int32)
    train_beh = np.zeros_like(train_items, dtype=np.int8)
    valid_items = np.zeros((len(eval_users), length), dtype=np.int32)
    valid_beh = np.zeros_like(valid_items, dtype=np.int8)
    test_items = np.zeros_like(valid_items)
    test_beh = np.zeros_like(valid_beh)
    histories, offsets = [], [0]
    trainable, counts, cutoffs = [], [], []
    timestamp_ties = 0
    behavior_counts = np.zeros(4, dtype=np.int64)
    negative_overlaps = {'valid': 0, 'test': 0}
    for idx, uid in enumerate(user_ids):
        uid = int(uid)
        start, end = bounds[idx:idx + 2]
        its, beh, ts = items[start:end], behaviors[start:end], times[start:end]
        vpos = np.flatnonzero(beh == 1)
        assert len(vpos) >= 3
        val_position, test_position = int(vpos[-2]), int(vpos[-1])
        observed = its[:val_position] > 0
        tri, trb = its[:val_position][observed], beh[:val_position][observed]
        if not np.array_equal(tri[trb == 1], reference[uid]):
            raise ValueError(f'Source does not reproduce the uploaded ML3 training sequence for user {uid}')
        timestamp_ties += int(np.sum(ts[:val_position] == ts[val_position]))
        train_items[idx], train_beh[idx] = pad(tri, length + 1), pad(trb, length + 1, np.int8)
        counts.append(len(tri))
        behavior_counts += np.bincount(trb, minlength=4)
        histories.append(tri)
        offsets.append(offsets[-1] + len(tri))
        cutoffs.append(int(ts[val_position]))
        if len(tri) > 1:
            trainable.append(idx)
        if uid in eval_row:
            row = eval_row[uid]
            observed_test = its[:test_position] > 0
            tei, teb = its[:test_position][observed_test], beh[:test_position][observed_test]
            assert its[val_position] == candidates['valid'][row, 0]
            assert its[test_position] == candidates['test'][row, 0]
            expected = list(reference[uid]) + [int(candidates['valid'][row, 0])]
            assert np.array_equal(tei[teb == 1], expected)
            valid_items[row], valid_beh[row] = pad(tri, length), pad(trb, length, np.int8)
            test_items[row], test_beh[row] = pad(tei, length), pad(teb, length, np.int8)
            assert np.array_equal(pad(reference[uid], length), reference_valid[row])
            assert np.array_equal(pad(expected, length), reference_test[row])
            # Preserve the original view-history negative sampling policy, but disclose
            # candidates that were observed via a non-view behavior in the MB input.
            for split, history, hist_beh in [('valid', tri, trb), ('test', tei, teb)]:
                neg = candidates[split][row, 1:]
                assert not np.isin(neg, history[hist_beh == 1]).any()
                negative_overlaps[split] += int(np.isin(neg, history[hist_beh != 1]).sum())
        if (idx + 1) % 25000 == 0:
            print(f'Checked and encoded {idx + 1:,}/{len(user_ids):,} users', flush=True)
    arrays = dict(user_ids=user_ids, train_items=train_items, train_behaviors=train_beh,
        trainable_rows=np.array(trainable, dtype=np.int64), history_items=np.concatenate(histories),
        history_offsets=np.array(offsets, dtype=np.int64), train_event_counts=np.array(counts, dtype=np.int32),
        train_cutoff_timestamp=np.array(cutoffs, dtype=np.int64), eval_users=eval_users,
        valid_items=valid_items, valid_behaviors=valid_beh, test_items=test_items, test_behaviors=test_beh,
        catalog=catalog, client_ids=np.array([client_bucket(u, num_clients) for u in user_ids], dtype=np.int16))
    for name, value in arrays.items():
        np.save(output / f'{name}.npy', value)
    membership = pd.DataFrame(dict(visitorid=user_ids, client_id=arrays['client_ids'],
        train_events=counts, eligible_training=np.isin(np.arange(len(user_ids)), trainable)))
    membership.to_csv(output / 'client_membership.csv.gz', index=False, compression='gzip')
    client_stats = membership.groupby('client_id').agg(users=('visitorid','size'),
        trainable_users=('eligible_training','sum'), train_events=('train_events','sum')).reset_index()
    client_stats.to_csv(output / 'client_metadata.csv', index=False)
    manifest = json.loads((inputs / 'archive_manifest.json').read_text(encoding='utf-8'))
    info = dict(schema_version=1, source='uploaded ML3 clean_events_final.csv', source_rows=source_rows,
        source_event_counts=source_counts, num_items=len(catalog), num_users=len(user_ids),
        num_trainable_users=len(trainable), num_eval_users=len(eval_users), num_clients=num_clients,
        maxlen=length, behaviors=BEHAVIORS, train_behavior_counts=behavior_counts[1:].tolist(),
        cutoff_timestamp_ties=timestamp_ties, candidate_nonview_history_overlaps=negative_overlaps,
        checks=dict(all_train_view_sequences_match_ml3=True, all_eval_view_sequences_match_ml3=True,
            all_candidate_indices_match_ml3=True, all_targets_match_source=True,
            training_stops_before_validation_view=True, test_history_stops_before_test_view=True),
        protocol='per-user leave-last-two-views-out; sampled 1 positive + 100 negatives; not global temporal split',
        loss='mean over valid next-event positions per user, then mean over users',
        privacy='ML4 is a single-machine FL simulation, NOT DP; preprocessing/roster/catalog/metrics are not private',
        archives=manifest, files={p.name: sha256(p) for p in sorted(output.glob('*.npy'))})
    (output / 'data_manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps({k:info[k] for k in ['source_rows','num_users','num_trainable_users','num_eval_users','num_items','checks']}, indent=2))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-clients', type=int, default=50)
    args = parser.parse_args()
    if args.num_clients < 1:
        parser.error('--num-clients must be positive')
    if args.archive_dir:
        extract_inputs(args.archive_dir, args.inputs)
    build_data(args.inputs, args.output, args.num_clients)


if __name__ == '__main__':
    main()
