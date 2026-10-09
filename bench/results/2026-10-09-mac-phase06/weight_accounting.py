"""Native tensor-byte model; this estimates logical work, not hardware DRAM counters."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
inventory = ROOT / 'bench/results/2026-10-09-mac-phase05/tensors.csv'
rows = list(csv.DictReader(inventory.open()))
groups = {name: {'bytes': 0, 'tensors': []} for name in ['routed_all', 'shared', 'dense', 'head', 'embedding', 'ple']}
for row in rows:
    name = row['name']
    if '_exps.' in name:
        group = 'routed_all'
    elif '_shexp.' in name:
        group = 'shared'
    elif name.startswith('output'):
        group = 'head'
    elif name == 'token_embd.weight':
        group = 'embedding'
    elif name == 'per_layer_token_embd.weight':
        group = 'ple'
    else:
        group = 'dense'
    groups[group]['bytes'] += int(row['bytes'])
    groups[group]['tensors'].append(name)
embedding = next(row for row in rows if row['name'] == 'token_embd.weight')
ple = next(row for row in rows if row['name'] == 'per_layer_token_embd.weight')
embedding_row = int(embedding['bytes']) // json.loads(embedding['dimensions'])[1]
ple_row = int(ple['bytes']) // json.loads(ple['dimensions'])[1]
headers = json.loads((ROOT / 'bench/results/2026-10-09-mac-phase05/gguf-headers.json').read_text())
meta = headers[0]['metadata']
experts = meta['qwen4exp.expert_count']['value']
topk = meta['qwen4exp.expert_used_count']['value']
ple_heads = (meta['qwen4exp.ple.ngram_size']['value'] - 1) * meta['qwen4exp.ple.heads_per_ngram']['value']
routed = groups['routed_all']['bytes'] * topk // experts
fixed = sum(groups[group]['bytes'] for group in ['shared', 'dense', 'head'])
single = fixed + routed + embedding_row
head = json.loads((OUT / 'mtp-header.json').read_text())
draft_fixed = draft_routed = draft_embedding = 0
for tensor in head['tensors']:
    if '_exps.' in tensor['name']:
        draft_routed += tensor['bytes'] * topk // experts
    elif tensor['name'] == 'token_embd.weight':
        draft_embedding += tensor['bytes'] // tensor['dimensions'][1]
    else:
        draft_fixed += tensor['bytes']
result = {'inventory': str(inventory.relative_to(ROOT)),
          'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest(), 'groups': groups,
          'experts': experts, 'top_k': topk, 'routed_bytes_per_token': routed,
          'fixed_main_bytes_per_forward': fixed, 'embedding_row_bytes': embedding_row,
          'single_decode_weight_bytes_per_token': single,
          'ple': {'heads': ple_heads, 'row_bytes': ple_row, 'logical_row_read_bytes_per_token': ple_heads * ple_row,
                  'vm_page_bytes': 16384, 'one_page_per_cold_row_bytes': ple_heads * 16384,
                  'max_two_pages_per_cold_row_bytes': ple_heads * 2 * 16384},
          'draft': {'fixed_bytes': draft_fixed, 'routed_bytes': draft_routed,
                    'embedding_row_bytes': draft_embedding,
                    'single_step_one_tensor_read_bytes': draft_fixed + draft_routed + draft_embedding},
          'interpretation': 'One native read per required tensor/selected expert/embedding row. Excludes activations, KV/state, shader/LUT reads, CPU repacks, and repeated tile reads. Not a physical DRAM measurement.',
          'mtp_verification_formula': {
              'logical_no_reuse': 'm * single_decode_weight_bytes_per_token',
              'ideal_unique_weight_lower_bound': 'fixed_main_bytes_per_forward + routed_bytes_per_token + m * embedding_row_bytes',
              'ideal_unique_weight_upper_bound': 'fixed_main_bytes_per_forward + min(m, experts/top_k) * routed_bytes_per_token + m * embedding_row_bytes',
              'm': '1 + actual number of drafted tokens in the verification trace',
              'output_denominator': 'accepted drafts plus correction/bonus token; emitted tokens capped at requested length',
              'limitations': 'No router-ID or hardware traffic capture; unique-expert reuse is a bound, not an observed hit rate. Draft catch-up work also occurs.'}}
(OUT / 'weight-accounting.json').write_text(json.dumps(result, indent=2) + '\n')
print('Main native read bytes/token', single, 'PLE', ple_heads * ple_row,
      'draft single-step tensor reads', result['draft']['single_step_one_tensor_read_bytes'])
