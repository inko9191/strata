"""Compare full-vocabulary logits only while the input token prefixes are identical."""
from array import array
import hashlib,json,math,statistics,sys
from pathlib import Path
root=Path(__file__).resolve().parent
cpu=json.loads((root/'capture-cpu/capture.json').read_text())
metal=json.loads((root/'capture-metal/capture.json').read_text())
result={'comparison':'pinned llama.cpp CPU vs Metal; not Strata parity','prompts':[]}
for c,m in zip(cpu['runs'],metal['runs']):
 if c['id']!=m['id'] or c['input_token_ids']!=m['input_token_ids']:raise ValueError('Input ids differ')
 ct,mt=c['generated_token_ids'],m['generated_token_ids']
 divergence=next((i for i,(a,b) in enumerate(zip(ct,mt)) if a!=b),None)
 if divergence is None and len(ct)!=len(mt):divergence=min(len(ct),len(mt))
 rows=min(len(ct),len(mt),divergence+1 if divergence is not None else len(ct))
 report={'id':c['id'],'input_ids_equal':True,'generated_tokens_equal':ct==mt,
         'first_divergence_zero_based':divergence,'generated_tokens_cpu':len(ct),'generated_tokens_metal':len(mt),
         'compared_logit_rows_with_equal_prefix':rows,'logit_rows':[]}
 n=c['logits_columns']
 paths=[root/('capture-'+backend)/(c['id']+'.logits.bin') for backend in ('cpu','metal')]
 report['logits_sha256']={backend:hashlib.sha256(path.read_bytes()).hexdigest() for backend,path in zip(('cpu','metal'),paths)}
 with paths[0].open('rb') as cf,paths[1].open('rb') as mf:
  for i in range(rows):
   a,b=array('f'),array('f');a.frombytes(cf.read(n*4));b.frombytes(mf.read(n*4))
   if sys.byteorder!='little':a.byteswap();b.byteswap()
   if len(a)!=n or len(b)!=n:raise ValueError('Truncated logits')
   norm=math.fsum(x*x for x in a)
   error=math.fsum((x-y)**2 for x,y in zip(a,b))
   max_error=max(abs(x-y) for x,y in zip(a,b))
   top=sorted(range(n),key=a.__getitem__,reverse=True)[:2]
   report['logit_rows'].append({'row':i,'relative_L2':math.sqrt(error/max(norm,1e-300)),
     'max_abs_error':max_error,'cpu_top1_top2_margin':a[top[0]]-a[top[1]],
     'cpu_top1':ct[i],'metal_top1':mt[i],'logits_bitwise_equal':a.tobytes()==b.tobytes()})
 report['max_relative_L2']=max(r['relative_L2'] for r in report['logit_rows'])
 report['max_absolute_logit_error']=max(r['max_abs_error'] for r in report['logit_rows'])
 result['prompts'].append(report)
 print(report['id'],'tokens_equal',report['generated_tokens_equal'],'divergence',divergence,
       'max_rel_L2',report['max_relative_L2'],'max_abs',report['max_absolute_logit_error'])
(root/'capture-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
r=json.loads((root/'reference-results.json').read_text())
for row in r['cli_trials']:
 text=(root/(row['log_prefix']+'.stderr.log')).read_text(errors='replace')
 import re
 footprint=re.findall(r'(\d+)\s+peak memory footprint',text)
 row['peak_memory_footprint_bytes_time_l']=int(footprint[-1]) if footprint else None
 if 'prompt_tok_s' not in row and 'ui_prompt_tok_s' in row:
  row['prompt_tok_s']=row['ui_prompt_tok_s'];row['prompt_rate_source']='UI rounded to 0.1 tok/s; detailed timing not logged'
 if 'decode_tok_s' not in row and 'ui_decode_tok_s' in row:
  row['decode_tok_s']=row['ui_decode_tok_s'];row['decode_rate_source']='UI rounded to 0.1 tok/s; detailed timing not logged'
summary={}
for backend in ('cpu','metal'):
 trials=[v for v in r['cli_trials'] if v['backend']==backend]
 summary[backend]={}
 for key in ('prompt_tok_s','decode_tok_s','peak_rss_bytes','peak_memory_footprint_bytes_time_l'):
  values=[v[key] for v in trials if v.get(key) is not None]
  summary[backend][key]={'all':values,'median':statistics.median(values),'min':min(values),'max':max(values)}
r['summary']=summary
different=[p['id'] for p in result['prompts'] if not p['generated_tokens_equal']]
r['status']='Measurements complete; mismatched CPU/Metal prompts: '+str(different)+'; no Strata correctness or performance claim'
r['all_cli_trials_valid']=all(t.get('valid_execution') for t in r['cli_trials'])
r['capture_comparison']='capture-comparison.json'
(root/'reference-results.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2))
