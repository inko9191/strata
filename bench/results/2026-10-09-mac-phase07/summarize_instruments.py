"""Nested medians for the six valid, alternating normal-build trace trials."""
import csv
import json
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent
groups={}
for config in ['off','mtp3']:
    labels=[f'shader-{config}-r{rnd}' for rnd in range(1,4)]
    if config=='off':labels[-1]+='-retry'
    trials=[json.loads((OUT/(label+'-instruments.json')).read_text()) for label in labels]
    records=[json.loads((OUT/(label+'.json')).read_text()) for label in labels]
    for record in records:
        assert not record.get('failure') and not record['gpu_errors']
        assert record['response_text_equal_before_profiled_after']
        assert record['profiled']['response']['timings']['cache_n']==0
        assert record['profiled']['response']['timings']['prompt_n']==4034
    keys=set(trials[0]['medians'])-{'shader_sample_ms_by_category','shader_sample_ms_by_name'}
    medians={key:{'median':statistics.median(t['medians'][key] for t in trials),
                  'min':min(t['medians'][key] for t in trials),
                  'max':max(t['medians'][key] for t in trials)} for key in keys}
    cats=set().union(*(t['medians']['shader_sample_ms_by_category'] for t in trials))
    medians['shader_sample_ms_by_category']={cat:statistics.median(t['medians']['shader_sample_ms_by_category'].get(cat,0) for t in trials) for cat in sorted(cats)}
    groups[config]={'labels':labels,'n':3,'graphs_per_trial':[t['selected_graph_count'] for t in trials],
        'medians':medians,'thermal_states':[t['thermal_states'] for t in trials],
        'server_ms_per_output':{key:statistics.median(r[key]['response']['timings']['predicted_per_token_ms'] for r in records) for key in ['before','profiled','after']}}
summary={'method':'Unmodified original v0.6.0 build; normal encoder topology; Instruments Shader Timeline samples are not exhaustive dispatch durations.',
    'groups':groups,'mtp_to_single_span_ratio':groups['mtp3']['medians']['span_ms']['median']/groups['off']['medians']['span_ms']['median']}
(OUT/'instruments-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'shader-samples.csv').open('w') as file:
    writer=csv.writer(file, lineterminator='\n')
    writer.writerow(['kernel_category','off_sample_ms_per_graph','mtp4_sample_ms_per_graph','normal_elapsed_ms_per_category'])
    for cat in sorted(set().union(*(g['medians']['shader_sample_ms_by_category'] for g in groups.values()))):
        writer.writerow([cat,groups['off']['medians']['shader_sample_ms_by_category'].get(cat,0),groups['mtp3']['medians']['shader_sample_ms_by_category'].get(cat,0),'unknown'])
print(json.dumps(summary,indent=2))
