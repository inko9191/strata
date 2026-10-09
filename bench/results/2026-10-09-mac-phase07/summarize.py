"""Validate diagnostic captures and report nested medians, not pooled-token estimates."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RAW = ROOT / 'build-mac-phase07'
inventory = ROOT / 'bench/results/2026-10-09-mac-phase05/tensors.csv'
tensors = list(csv.DictReader(inventory.open()))
by_name = {r['name']: r for r in tensors}
median = statistics.median


def weight_category(name, dtype):
    if '_exps.' in name:
        return 'routed experts'
    if '_shexp.' in name:
        return 'shared expert'
    if name.startswith('output'):
        return 'head'
    if '.hc_' in name:
        return 'HC'
    if '.ple_' in name:
        return 'PLE GPU'
    if name.endswith('ssm_conv1d.weight'):
        return 'GDN'
    if name.startswith('blk.') and name in by_name and len(json.loads(by_name[name]['dimensions'])) >= 2:
        return 'non-expert matvec ' + dtype.upper()
    if '.ssm_' in name:
        return 'GDN'
    if '.attn_' in name or '.indexer.' in name:
        return 'QSA'
    return 'other'


weight_bytes = collections.Counter()
dense = collections.defaultdict(collections.Counter)
for r in tensors:
    name, dtype = r['name'], r['dtype']
    if name in ['token_embd.weight', 'per_layer_token_embd.weight']:
        continue
    cat = weight_category(name, dtype)
    weight_bytes[cat] += int(r['bytes']) * (10 / 512 if cat == 'routed experts' else 1)
    if '_exps.' not in name and '_shexp.' not in name and not name.startswith('output'):
        dense[re.sub(r'^blk\.\d+\.', '', name)][dtype] += int(r['bytes'])
assert sum(sum(v.values()) for v in dense.values()) == 2936153600
with (OUT / 'dense-by-kind-type.csv').open('w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['tensor_kind', 'dtype', 'bytes', 'decimal_GB'])
    for name, types in sorted(dense.items()):
        for dtype, size in sorted(types.items()):
            writer.writerow([name, dtype, size, size / 1e9])
dense_types = collections.Counter()
for types in dense.values():
    dense_types.update(types)


def category(f):
    op, name, source, dtype, kernels = f[9:14]
    exact = source.split(' (')[0]
    matrix = op in ['MUL_MAT', 'MUL_MAT_ID'] and exact in by_name
    if matrix:
        return weight_category(exact, dtype)
    if name in ['result_output', 'result_norm', 'hc_gate'] or source == 'hc_gate' or source.startswith('output'):
        return 'head'
    if 'ple_' in name or 'ple_' in source:
        return 'PLE GPU'
    if 'hc_' in name or 'hc_' in source or 'dsv4_hc_' in kernels:
        return 'HC'
    if 'ffn_moe' in name or 'ffn_moe' in source or 'topk_moe' in kernels:
        return 'routed experts'
    if re.match(r'(ffn_gate-|ffn_up-|ffn_swiglu-|shared_expert|ffn_shexp)', name):
        return 'shared expert'
    if 'delta_net' in kernels or op == 'SSM_CONV' or re.match(r'(linear_attn|conv_output|beta_sigmoid|a_softplus|z-|q_conv|k_conv|v_conv)', name):
        return 'GDN'
    if 'flash_attn' in kernels or ('fwht' in kernels and 'attn_inp_' in source) or re.match(r'(indexer|Qcur|Kcur|Vcur|attn_gated)', name):
        return 'QSA'
    if name.startswith('l_last') or name.startswith('ffn_out'):
        return 'other'
    # Graph reordering makes the preceding node an unreliable semantic parent.
    # Leave anonymous/unidentified intermediates unassigned rather than guess.
    return 'other'


def union(intervals):
    result = []
    for begin, end in sorted(intervals):
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((begin, end))
    return result


def read(label):
    record = json.loads((OUT / (label + '.json')).read_text())
    assert record['probe_source_sha256'] == hashlib.sha256((OUT/'metal_probe.mm').read_bytes()).hexdigest()
    assert not record.get('failure') and not record['gpu_errors']
    assert record['response_text_equal_before_profiled_after']
    for key in ['before', 'profiled', 'after']:
        timing = record[key]['response']['timings']
        assert timing['cache_n'] == 0 and timing['predicted_n'] == record['output_tokens']
    graphs, calls, cpu = {}, {}, []
    rows = [line.split('\t') for line in (RAW / (label + '.tsv')).read_text().splitlines()]
    for f in rows:
        if f[0] == 'graph':
            graphs[f[1]] = {'id': f[1], 'tokens': int(f[2]), 'main': bool(int(f[3])),
                            'stage': bool(int(f[4])), 'nodes': int(f[5]), 'call': f[6], 'cb': [], 'ops': []}
        elif f[0] == 'call_begin':
            calls[f[1]] = {'begin': int(f[4]) / 1e9}
        elif f[0] == 'call_end':
            assert int(f[2]) == 0
            calls[f[1]]['end'] = int(f[3]) / 1e9
        elif f[0] == 'cb':
            assert int(f[6]) == 4
            if float(f[4]) == 0:
                graphs[f[1]]['cb_start_unavailable'] = graphs[f[1]].get('cb_start_unavailable',0)+1
                continue  # Never substitute a zero start time into a duration.
            assert float(f[5]) >= float(f[4]) > 0
            graphs[f[1]]['cb'].append((float(f[4]), float(f[5])))
        elif f[0] == 'graph_cpu':
            graphs[f[1]]['encode_submit_cpu_ms'] = (int(f[3])-int(f[2]))/1e6
        elif f[0] == 'op':
            assert float(f[6]) >= 0
            graphs[f[1]]['ops'].append(f)
        elif f[0] == 'cpu':
            assert int(f[8]) == 0
            cpu.append({'call': f[1], 'tokens': int(f[2]), 'ms': (int(f[4])-int(f[3]))/1e6,
                        'ple_operand_bytes': int(f[5]), 'embedding_operand_bytes': int(f[6])})
    dispatch = [f for f in rows if f[0] in ['dispatch', 'op']]
    kernels = collections.defaultdict(collections.Counter)
    counts = collections.defaultdict(collections.Counter)
    for f in dispatch:
        g = graphs[f[1]]
        if not g['main'] or g['tokens'] not in [1, 4]:
            continue
        idx = 9 if f[0] == 'op' else 6
        if f[idx] in ['MUL_MAT', 'MUL_MAT_ID']:
            for kernel in f[-1].split(';'):
                kernels[g['tokens']][kernel] += 1
                counts[g['tokens']][('expert ' if f[idx] == 'MUL_MAT_ID' else 'dense ') +
                                   ('mul_mm' if 'kernel_mul_mm' in kernel else
                                    'mul_mv_mma' if 'mul_mv_mma' in kernel else
                                    'mul_mv_ext' if 'mul_mv_ext' in kernel else 'mul_mv')] += 1
    measured = []
    for g in graphs.values():
        if not g['cb']:
            raise AssertionError('No completion timing for graph')
        if g['tokens'] > 4:
            continue
        begin = min(s for s, e in g['cb'])
        end = max(e for s, e in g['cb'])
        busy_cb = sum(e-s for s, e in union(g['cb']))
        g.update(span_ms=(end-begin)*1000, cb_union_ms=busy_cb*1000,
                 between_cb_gap_ms=(end-begin-busy_cb)*1000,
                 call_cpu_wall_ms=(calls[g['call']]['end']-calls[g['call']]['begin'])*1000,
                 begin=begin, end=end)
        times = collections.Counter()
        for f in g['ops']:
            cat = category(f)
            times[cat] += float(f[6])/1e6
        g['category_ms'] = dict(times)
        g['op_sum_ms'] = sum(times.values())
        measured.append({k:v for k,v in g.items() if k not in ['cb','ops']})
    main = [g for g in measured if g['main'] and g['tokens'] == (1 if record['config']=='off' else 4)]
    if record['config'] == 'off':
        assert len(main) == record['output_tokens'] - 1
    else:
        # The first m=4 graph is the final four prompt tokens, not a verification.
        main = main[1:]
        events = re.findall(r'accepted\s+(\d+)/\s*(\d+) draft tokens',record['profiled_log'])
        assert events
        full = [event for event in events if int(event[1])==3]
        assert len(full)==len(main)
        for g,event in zip(main,full):
            g['accepted_drafts']=int(event[0])
            g['proposed_drafts']=int(event[1])
    if not main:
        raise AssertionError('No decode/verification groups')
    if '-stage-' in label:
        assert all(g['op_sum_ms'] > 0 for g in main)
    per_category = {cat: median(g['category_ms'].get(cat,0) for g in main)
                    for cat in set().union(*(g['category_ms'] for g in main))}
    result = {'label': label, 'main_groups': main, 'group_count': len(main),
              'main_graph_ms_median': median(g['span_ms'] for g in main),
              'main_cb_union_ms_median': median(g['cb_union_ms'] for g in main),
              'between_cb_gap_ms_median': median(g['between_cb_gap_ms'] for g in main),
              'public_call_cpu_wall_ms_median': median(g['call_cpu_wall_ms'] for g in main),
              'encode_submit_cpu_ms_median': median(g['encode_submit_cpu_ms'] for g in main),
              'op_sum_ms_median': median(g['op_sum_ms'] for g in main), 'category_ms': per_category,
              'matrix_kernel_counts_including_prompt_tail': {str(k):dict(v) for k,v in counts.items()},
              'kernel_names_including_prompt_tail': {str(k):dict(v) for k,v in kernels.items()},
              'response_text_equal': True,
              'decode_ms_per_output': {key:record[key]['response']['timings']['predicted_per_token_ms'] for key in ['before','profiled','after']},
              'raw_sha256': hashlib.sha256((RAW/(label+'.tsv')).read_bytes()).hexdigest()}
    # Count selected decode/verification groups only, excluding the prompt tail.
    ids = {g['id'] for g in main}
    selected_kernels = collections.Counter()
    audit = collections.Counter()
    for g in graphs.values():
        if g['id'] not in ids:
            continue
        for f in g['ops']:
            cat = category(f)
            audit[(cat, *f[9:14])] += 1
    for f in dispatch:
        if f[1] in ids:
            selected_kernels.update(k for k in f[-1].split(';') if k)
    result['selected_kernel_counts'] = dict(selected_kernels)
    result['cb_start_unavailable_count'] = sum(g.get('cb_start_unavailable',0) for g in main)
    if not record['label'].startswith('final-stage'):
        assert result['cb_start_unavailable_count'] == 0
    if audit:
        with (OUT/(label+'-op-audit.csv')).open('w') as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(['category','op','node','src0','src0_dtype','pipelines','count'])
            for key, count in sorted(audit.items()):
                writer.writerow([*key,count])
    if cpu:
        targets = {g['call'] for g in main}
        selected = [c for c in cpu if c['call'] in targets]
        if selected:
            result['cpu_graph'] = {'n': len(selected), 'wall_ms_median': median(c['ms'] for c in selected),
                                   'raw_operand_bytes': sorted({(c['ple_operand_bytes'],c['embedding_operand_bytes']) for c in selected})}
    return result


trials = []
for mode in ['final-stage','final-passive']:
    for config in ['off','mtp3']:
        for rnd in range(1,4):
            trials.append(read(f'{mode}-{config}-r{rnd}'))
summary = {'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest(),
           'dense_total_bytes': 2936153600, 'dense_bytes_by_dtype': dict(dense_types),
           'native_weight_bytes_by_category': dict(weight_bytes), 'trials': trials, 'groups': {}}
for mode in ['final-stage','final-passive']:
    for config in ['off','mtp3']:
        selected = [t for t in trials if t['label'].startswith(f'{mode}-{config}-')]
        result = {}
        for key in ['main_graph_ms_median','main_cb_union_ms_median','between_cb_gap_ms_median',
                    'public_call_cpu_wall_ms_median','encode_submit_cpu_ms_median','op_sum_ms_median']:
            values = [t[key] for t in selected]
            result[key] = {'median': median(values), 'min': min(values), 'max': max(values), 'n':len(values)}
        result['category_ms'] = {cat:median(t['category_ms'].get(cat,0) for t in selected)
                                 for cat in set().union(*(t['category_ms'] for t in selected))}
        result['group_counts'] = [t['group_count'] for t in selected]
        result['decode_ms_per_output'] = {key:median(t['decode_ms_per_output'][key] for t in selected) for key in ['before','profiled','after']}
        if all('cpu_graph' in t for t in selected):
            result['cpu_graph_ms_median'] = median(t['cpu_graph']['wall_ms_median'] for t in selected)
        summary['groups'][f'{mode}-{config}'] = result
summary['mtp_to_single_gpu_span_ratio'] = summary['groups']['final-passive-mtp3']['main_graph_ms_median']['median'] / summary['groups']['final-passive-off']['main_graph_ms_median']['median']
summary['limitations'] = [
    'Stage encoder boundaries perturb execution; stage times are diagnostic, not normal category times.',
    'Stage command-buffer spans with unavailable start timestamps are incomplete and not used for conclusions.',
    'Anonymous/unidentified intermediate nodes remain other; see per-trial op-audit CSV.',
    'Native weights are logical operands, not measured DRAM traffic; activation/KV/state traffic is unknown.',
    'Generated text equality is checked, not token-ID or logit parity.'
]
with (OUT/'diagnostic-category-times.csv').open('w') as file:
    writer=csv.writer(file, lineterminator='\n')
    writer.writerow(['category','single_stage_ms','mtp4_stage_ms','native_bytes_single','native_weight_equivalent_GB_s','physical_traffic_GB_s'])
    for cat, ms in sorted(summary['groups']['final-stage-off']['category_ms'].items()):
        size=weight_bytes.get(cat,0)
        # State-heavy/nonmatrix groups cannot be modeled using weights alone.
        gb_s=size/ms/1e6 if ms and (cat.startswith('non-expert matvec') or cat in ['head','routed experts','shared expert','HC','PLE GPU']) else ''
        writer.writerow([cat,ms,summary['groups']['final-stage-mtp3']['category_ms'].get(cat,0),size,gb_s,'unknown'])
(OUT / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:{key:value for key,value in v.items() if key!='category_ms'} for k,v in summary['groups'].items()},indent=2))
print('ratio',summary['mtp_to_single_gpu_span_ratio'])
print('diagnostic categories',summary['groups']['final-stage-off']['category_ms'])
