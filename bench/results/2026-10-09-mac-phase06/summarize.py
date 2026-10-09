"""Validate measurements and summarize observed rates plus explicit tensor-read models."""
import collections
import json
import math
from pathlib import Path
import re
import statistics
from run_long import BAD

OUT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (OUT / 'trials.jsonl').read_text().splitlines()]
assert len(rows) == 36, 'Expect 3 rounds, 2 configs, 2 temps, 2x4K + 1x32K'
weights = json.loads((OUT / 'weight-accounting.json').read_text())
W = weights['single_decode_weight_bytes_per_token']
F = weights['fixed_main_bytes_per_forward']
R = weights['routed_bytes_per_token']
E = weights['embedding_row_bytes']
CAPACITY = 582.6 * 1e9
for config in ['off', 'mtp3']:
    for rnd in [1, 2, 3]:
        session_rows = [row for row in rows if row['config'] == config and row['round'] == rnd]
        text = (OUT / f'{config}-r{rnd}.log').read_text(errors='replace')
        assert not BAD.search(text), 'GPU/model error in a session log'
        task_order = re.findall(r'id\s+0\s*\|\s*task\s+(\d+).*?prompt eval time', text)
        assert len(task_order) == 8
        events = re.findall(r'task\s+(\d+).*?accepted\s+(\d+)/\s*(\d+) draft tokens( \(restore checkpoint\))?', text)
        assert len(session_rows) == 6
        for task, row in zip(task_order[2:], session_rows):
            row['verification_events'] = [{'accepted': int(a), 'drafted': int(d), 'restore_checkpoint': bool(c)}
                                         for t, a, d, c in events if t == task]
            tm = row['timings']
            assert row['valid_execution'] and not row['gpu_errors']
            assert tm['predicted_n'] == 256 and tm['cache_n'] == 0
            assert all(math.isfinite(tm[key]) and tm[key] > 0 for key in ['prompt_per_second', 'predicted_per_second', 'prompt_ms'])
            generated = tm['predicted_n'] - 1  # First token is counted in prompt time by this server.
            assert abs(tm['predicted_per_second'] - generated / (tm['predicted_ms'] / 1000)) < 1e-8
            row['decode_timed_output_tokens'] = generated
            if config == 'mtp3':
                v = row['verification_events']
                completed = [event for event in v if not event['restore_checkpoint']]
                assert len(completed) == row['metrics_delta']['llamacpp:spec_decode_num_drafts_total']
                assert sum(event['accepted'] for event in completed) == tm['draft_n_accepted']
                assert tm['draft_n'] == row['metrics_delta']['llamacpp:spec_decode_num_draft_tokens_total']
                n_input = sum(event['drafted'] + 1 for event in v)
                lower = sum(F + R + (event['drafted'] + 1) * E for event in v)
                upper = sum(F + min(event['drafted'] + 1, 512 / 10) * R + (event['drafted'] + 1) * E for event in v)
                total = n_input * W
                row['weight_model'] = {'verification_passes_including_replay': len(v),
                    'completed_verification_passes': len(completed), 'verified_input_token_positions': n_input,
                    'drafted': tm['draft_n'], 'accepted_drafts': tm['draft_n_accepted'],
                    'correction_bonus_outputs_before_limit': len(completed),
                    'mean_accepted_drafts_per_completed_pass': tm['draft_n_accepted'] / len(completed),
                    'mean_verification_no_reuse_bytes': total / len(v),
                    'mean_verification_unique_lower_bytes': lower / len(v),
                    'mean_verification_unique_upper_bytes': upper / len(v),
                    'target_no_reuse_bytes_per_timed_output': total / generated,
                    'target_unique_lower_bytes_per_timed_output': lower / generated,
                    'target_unique_upper_bytes_per_timed_output': upper / generated,
                    'target_no_reuse_equivalent_bandwidth_percent': total / generated * tm['predicted_per_second'] / CAPACITY * 100,
                    'target_unique_lower_equivalent_bandwidth_percent': lower / generated * tm['predicted_per_second'] / CAPACITY * 100,
                    'target_unique_upper_equivalent_bandwidth_percent': upper / generated * tm['predicted_per_second'] / CAPACITY * 100,
                    'generated_draft_only_bytes_per_timed_output': tm['draft_n'] * weights['draft']['single_step_one_tensor_read_bytes'] / generated,
                    'generated_draft_only_equivalent_bandwidth_percent': tm['draft_n'] * weights['draft']['single_step_one_tensor_read_bytes'] / generated * tm['predicted_per_second'] / CAPACITY * 100,
                    'unaccounted': 'Draft catch-up forward work, confidence-rejected draft attempts, activations, KV/recurrent state, CPU repack and tile rereads; router unions are bounds.'}
            else:
                row['weight_model'] = {'single_decode_weight_bytes_per_timed_output': W,
                    'equivalent_bandwidth_percent': W * tm['predicted_per_second'] / CAPACITY * 100}
(OUT / 'trials.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def stats(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'all': values}


groups = []
for cell in ['4k', '32k']:
    for temp in [0, 0.7]:
        for config in ['off', 'mtp3']:
            selected = [row for row in rows if (row['cell'], row['temperature'], row['config']) == (cell, temp, config)]
            assert len(selected) == (6 if cell == '4k' else 3)
            group = {'cell': cell, 'temperature': temp, 'config': config, 'n': len(selected),
                     'prompt_tok_s': stats([row['timings']['prompt_per_second'] for row in selected]),
                     'decode_tok_s': stats([row['timings']['predicted_per_second'] for row in selected]),
                     'ttft_server_prompt_seconds': stats([row['timings']['prompt_ms'] / 1000 for row in selected]),
                     'request_rss_peak_sampled_bytes': stats([row['request_rss_peak_sampled_bytes'] for row in selected]),
                     'weight_model': {key: stats([row['weight_model'][key] for row in selected])
                         for key, value in selected[0]['weight_model'].items() if isinstance(value, (int, float))}}
            if config == 'mtp3':
                group['draft_acceptance'] = stats([row['timings']['draft_n_accepted'] / row['timings']['draft_n'] for row in selected])
            groups.append(group)
quality = []
for row in rows:
    if row['config'] != 'off':
        continue
    pair = next(other for other in rows if other['config'] == 'mtp3' and
        (row['round'], row['cell'], row['temperature'], row['rep']) ==
        (other['round'], other['cell'], other['temperature'], other['rep']))
    assert row['prompt_sha256'] == pair['prompt_sha256']
    left = json.loads((OUT / row['response_file']).read_text())['response']['choices'][0]['message']
    right = json.loads((OUT / pair['response_file']).read_text())['response']['choices'][0]['message']
    quality.append({'round': row['round'], 'cell': row['cell'], 'temperature': row['temperature'], 'rep': row['rep'],
                    'prompt_identical': True, 'message_equal': left == right,
                    'scope': 'Decoded chat message comparison, not hidden-state or token-ID parity.'})
short = json.loads((OUT / 'short-trials.json').read_text())
assert len(short) == 6 and all(row['valid_execution'] for row in short)
short_summary = {version: {key: stats([row[key] for row in short if row['version'] == version])
    for key in ['prompt_tok_s', 'decode_tok_s', 'peak_rss_bytes']} for version in ['pinned', 'v0.6.0']}
targets = []
for cell in ['4k', '32k']:
    for temp in [0, 0.7]:
        candidates = [group for group in groups if group['cell'] == cell and group['temperature'] == temp]
        best_decode = max(candidates, key=lambda group: group['decode_tok_s']['median'])
        best_prompt = max(candidates, key=lambda group: group['prompt_tok_s']['median'])
        targets.append({'cell': cell, 'temperature': temp,
            'prompt_tok_s': best_prompt['prompt_tok_s']['median'], 'prompt_config': best_prompt['config'],
            'decode_tok_s': best_decode['decode_tok_s']['median'], 'decode_config': best_decode['config'],
            'definition': 'Best measured configuration median per metric, not best individual trial or a Strata prediction.'})
result = {'engine': 'Unmodified llama.cpp only; no Strata performance prediction', 'groups': groups,
          'short': short_summary, 'paired_messages': quality, 'strata_port_targets': targets,
          'bandwidth_denominator_GB_s': 582.6, 'bandwidth_interpretation': 'Native tensor-work equivalent, not a measured physical DRAM utilization counter.'}
(OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
for group in groups:
    print(group['cell'], group['temperature'], group['config'], 'n', group['n'],
          'pp', round(group['prompt_tok_s']['median'], 2), 'decode', round(group['decode_tok_s']['median'], 2),
          'TTFT', round(group['ttft_server_prompt_seconds']['median'], 3))
print('Greedy decoded-message matches', sum(row['message_equal'] for row in quality if row['temperature'] == 0),
      '/', sum(row['temperature'] == 0 for row in quality))
