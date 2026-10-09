"""Export small trace tables and aggregate normal graph spans and shader samples.

Shader Sample Duration is not an exhaustive dispatch wall-time measurement.
Raw .trace/XML remain local. This tool never runs an inference process.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
RAW = OUT.parents[2]/'build-mac-phase07'


def rows(path):
    root = ET.parse(path).getroot()
    refs = {e.get('id'): e for e in root.iter() if e.get('id')}
    return [[refs.get(e.get('ref'),e) for e in row] for row in root.findall('.//row')]


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0],max(end,merged[-1][1]))
        else:
            merged.append((start,end))
    return merged


def shader_category(name):
    if 'mul_mv_id' in name:
        return 'routed experts matvec'
    if 'mul_mv' in name:
        dtype=re.search(r'mul_mv_(?:ext_|mma_)?(bf16|iq4_xs|iq3_s|q6_K|iq4_nl|q2_0|q8_0)',name)
        return 'matvec '+(dtype.group(1) if dtype else 'unclassified')+' (mixed semantic roles)'
    if 'delta_net' in name or 'ssm_conv' in name:
        return 'GDN named kernels'
    if 'flash_attn' in name or 'lightning_indexer' in name:
        return 'QSA named kernels'
    if 'dsv4_hc_' in name:
        return 'HC named kernels'
    if 'topk_moe' in name or 'moe_reduce' in name:
        return 'routing/reduction named kernels'
    return 'other / semantic role unavailable'


def analyze(label, export=True):
    record=json.loads((OUT/(label+'.json')).read_text())
    assert not record.get('failure') and not record['gpu_errors'] and record['trace_exit_code']==0
    assert len({record[k]['response']['choices'][0]['message'].get('content','') for k in ['before','profiled','after']})==1
    schemas=['metal-gpu-intervals','metal-shader-profiler-intervals',
             'metal-application-intervals','device-thermal-state-intervals']
    paths={}
    for schema in schemas:
        paths[schema]=RAW/(label+'-'+schema+'.xml')
        if export and not paths[schema].exists():
            subprocess.run(['xcrun','xctrace','export','--input',str(RAW/(label+'.trace')),
                '--xpath',f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
                '--output',str(paths[schema])],check=True,stdout=subprocess.DEVNULL)
    buffers=collections.defaultdict(list)
    for f in rows(paths['metal-gpu-intervals']):
        if 'llama-server' in f[10].get('fmt','') and 'Compute Command' in f[6].get('fmt',''):
            start=int(f[0].text)
            buffers[f[15].text].append((start,start+int(f[1].text)))
    ordered=sorted(buffers,key=lambda k:min(s for s,e in buffers[k]))
    assert len(ordered)%2==0
    graphs=[]
    for idx in range(0,len(ordered),2):
        keys=ordered[idx:idx+2]
        intervals=union([p for k in keys for p in buffers[k]])
        start,end=intervals[0][0],intervals[-1][1]
        busy=sum(e-s for s,e in intervals)
        graphs.append({'index':idx//2,'cb_ids':keys,'start_ns':start,'end_ns':end,
            'span_ms':(end-start)/1e6,'active_compute_ms':busy/1e6,
            'within_span_no_active_compute_ms':(end-start-busy)/1e6})
    if record['config']=='off':
        selected=graphs[3:]
        assert len(selected)==record['output_tokens']-1
        for g in selected:g['tokens']=1
    else:
        # Main/draft have two CBs each. First three prompt batches each have both graphs.
        # Decode main spans (>10 ms) are separated from draft spans (~1-2 ms).
        # Validate widths against ordered accepted/drafted events, rather than assuming m=4.
        candidates=[g for g in graphs[6:] if g['span_ms']>10]
        events=re.findall(r'accepted\s+(\d+)/\s*(\d+) draft tokens',record['profiled_log'])
        assert len(candidates) in [len(events),len(events)+1]
        for g,event in zip(candidates,events):
            g['tokens']=1+int(event[1])
            g['accepted_drafts']=int(event[0])
            g['proposed_drafts']=int(event[1])
        selected=[g for g in candidates if g.get('tokens')==4]
        assert selected and all(g['span_ms']>25 for g in selected)
    shaders=[]
    for f in rows(paths['metal-shader-profiler-intervals']):
        if 'llama-server' in f[11].get('fmt',''):
            shaders.append((int(f[0].text),int(f[1].text),f[2].text or f[2].get('fmt','')))
    encoding=collections.defaultdict(list)
    for f in rows(paths['metal-application-intervals']):
        if 'llama-server' in f[2].get('fmt','') and f[10].get('fmt')=='Encoding' and f[5].get('fmt','').startswith('Command Buffer'):
            start=int(f[0].text)
            encoding[f[11].text].append((start,start+int(f[1].text)))
    for g in selected:
        totals=collections.Counter()
        names=collections.Counter()
        for start,duration,name in shaders:
            if g['start_ns']<=start<g['end_ns']:
                names[name]+=duration/1e6
                totals[shader_category(name)]+=duration/1e6
        g['shader_sample_ms_by_name']=dict(names)
        g['shader_sample_ms_by_category']=dict(totals)
        g['shader_sample_ms_sum']=sum(totals.values())
        g['cpu_cb_encoding_ms']=sum(e-s for s,e in union([p for k in g['cb_ids'] for p in encoding[k]]))/1e6
    if record['config']=='off':
        for previous,next_graph in zip(selected,selected[1:]):
            previous['between_graph_no_active_compute_ms']=(next_graph['start_ns']-previous['end_ns'])/1e6
    keys=['span_ms','active_compute_ms','within_span_no_active_compute_ms',
          'cpu_cb_encoding_ms','shader_sample_ms_sum','between_graph_no_active_compute_ms']
    medians={k:statistics.median(g[k] for g in selected if k in g) for k in keys if any(k in g for g in selected)}
    categories=set().union(*(g['shader_sample_ms_by_category'] for g in selected))
    medians['shader_sample_ms_by_category']={k:statistics.median(g['shader_sample_ms_by_category'].get(k,0) for g in selected) for k in sorted(categories)}
    names=set().union(*(g['shader_sample_ms_by_name'] for g in selected))
    medians['shader_sample_ms_by_name']={k:statistics.median(g['shader_sample_ms_by_name'].get(k,0) for g in selected) for k in sorted(names)}
    result={'label':label,'config':record['config'],'compute_cb_count':len(ordered),
        'selected_graph_count':len(selected),'graphs':selected,'medians':medians,
        'thermal_states':sorted({f[3].get('fmt') for f in rows(paths['device-thermal-state-intervals'])}),
        'export_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        'limitations':['GPU Compute intervals are not ALU occupancy.',
            'Shader Sample Duration is sampled work, not exhaustive dispatch wall time; do not normalize to fill the GPU span.',
            'Pipeline names shared by HC, head, PLE, and other matrices cannot uniquely allocate those semantic roles.',
            'CPU encoding can overlap GPU work; gaps cannot be causally assigned to CPU/driver/submit from this summary.',
            'MTP main/draft span classification is checked against passive traces and ordered draft events; short tail groups excluded.']}
    (OUT/(label+'-instruments.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('labels',nargs='+')
    parser.add_argument('--no-export',action='store_true')
    args=parser.parse_args()
    for label in args.labels:
        r=analyze(label,not args.no_export)
        print(label,r['selected_graph_count'],r['medians']['span_ms'],r['medians']['shader_sample_ms_sum'],r['thermal_states'],flush=True)
