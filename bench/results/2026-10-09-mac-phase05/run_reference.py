"""Pinned llama-cli runs plus external full-logit capture; no llama.cpp changes."""
import fcntl,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path
ROOT=Path.cwd()
OUT=ROOT/'bench/results/2026-10-09-mac-phase05'
BUILD=ROOT/'build-mac-phase05'
LOCK=(BUILD/'reference.lock').open('a+')
try:
 fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 raise SystemExit('Another reference runner holds the lock; no tests started')
CONFIG=json.loads((OUT/'prompts.json').read_text())
HF=json.loads((OUT/'huggingface-revision.json').read_text())
REPORT={'strata_commit':'fb58e0dbc8399662c0e47c76578c6e878b14f6cf',
        'llama_commit':'3cf03257f219afbe7334045ff7c6a06ac68c627d','model_revision':HF['sha'],
        'benchmark_engine':'unmodified pinned llama.cpp, not Strata','files':[],'cli_trials':[],'captures':[]}
def save(): (OUT/'reference-results.json').write_text(json.dumps(REPORT,indent=2,ensure_ascii=False)+'\n')
print('Waiting for complete downloads before SHA verification and model execution',flush=True)
while 'DOWNLOAD COMPLETE' not in (BUILD/'download-ranges.log').read_text(): time.sleep(2)
for f in HF['siblings']:
 if not f['rfilename'].startswith('IQ2_XS/'):continue
 path=BUILD/'models'/Path(f['rfilename']).name
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(16*1024*1024),b''):digest.update(chunk)
 actual=digest.hexdigest()
 row={'filename':path.name,'size_bytes':path.stat().st_size,'expected_sha256':f['lfs']['sha256'],'actual_sha256':actual,
      'passed':actual==f['lfs']['sha256'] and path.stat().st_size==f['size']}
 REPORT['files'].append(row); save()
 print('SHA',path.name,row['passed'],flush=True)
 if not row['passed']:raise SystemExit('Model SHA/length mismatch: no inference attempted')
model=BUILD/'models'/'Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf'
for shard in (BUILD/'models').glob('*.gguf'):
 with (OUT/(shard.name+'.strata-reader.log')).open('w') as log:
  result=subprocess.run([str(ROOT/'build-mac-phase0/strata-gguf'),str(shard),'--check'],stdout=log,stderr=subprocess.STDOUT)
  REPORT.setdefault('strata_header_checks',[]).append({'file':shard.name,'exit_code':result.returncode});save()
def execute(command,prefix,timeout=1200):
 with (OUT/(prefix+'.stdout.log')).open('w') as out,(OUT/(prefix+'.stderr.log')).open('w') as err:
  start=time.monotonic()
  process=subprocess.Popen(['/usr/bin/time','-l']+command,stdout=out,stderr=err,start_new_session=True)
  try:code=process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   code='timeout'
 text=(OUT/(prefix+'.stdout.log')).read_text(errors='replace')+'\n'+(OUT/(prefix+'.stderr.log')).read_text(errors='replace')
 rss=re.findall(r'(\d+)\s+maximum resident set size',text)
 errors=re.findall(r'(?im)^.*(?:Insufficient Memory|command buffer.*failed|error loading model).*$',text)
 return {'command':command,'exit_code':code,'wall_seconds_including_load':time.monotonic()-start,
         'peak_rss_bytes':int(rss[-1]) if rss else None,'backend_errors':errors,
         'valid_execution':code==0 and not errors},text
english=next(p['text'] for p in CONFIG['prompts'] if p['id']==CONFIG['benchmark_prompt_id'])
for trial in range(1,4):
 for backend in ('cpu','metal'):
  command=[str(BUILD/('llama-'+backend)/'bin/llama-cli'),'-m',str(model),'-ngl','0' if backend=='cpu' else '99',
           '-ot','per_layer_token_embd.weight=CPU','-c','512','-b','64','-ub','64','-t','8','-tb','8',
           '--fit','off','--lazy-mode','on','--flash-attn','off','--no-warmup','--offline','--single-turn',
           '--temp','0','--seed','1234','--repeat-penalty','1.0','-n','32','--jinja',
           '--chat-template-file',str(OUT/'raw-template.jinja'),'-p',english,'--log-verbosity','3','--log-colors','off']
  prefix=f'{backend}-cli-{trial}'
  row,text=execute(command,prefix)
  row.update({'backend':backend,'trial':trial,'prompt_id':'english','log_prefix':prefix})
  for label,pattern in [('prompt',r'prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second'),
                        ('decode',r'(?m)^.*?\beval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second')]:
   matches=re.findall(pattern,text)
   if label=='decode':matches=[m for m in re.finditer(pattern,text) if 'prompt eval' not in m.group(0)]
   if matches:
    item=matches[-1] if isinstance(matches[-1],tuple) else matches[-1].groups()
    row[label+'_ms']=float(item[0]);row[label+'_tokens']=int(item[1]);row[label+'_tok_s']=float(item[2])
  ui=re.findall(r'Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s',text)
  if ui:row['ui_prompt_tok_s'],row['ui_decode_tok_s']=map(float,ui[-1])
  REPORT['cli_trials'].append(row);save()
  print(prefix,'exit',row['exit_code'],'RSS',row['peak_rss_bytes'],flush=True)
  if not row['valid_execution']:
   REPORT['status']='llama-cli failed; no source changes or throughput summary';save();raise SystemExit(0)
for backend in ('cpu','metal'):
 destination=OUT/('capture-'+backend)
 command=[str(BUILD/('capture-'+backend)),str(model),backend,str(OUT/'prompts.json'),str(destination)]
 row,text=execute(command,backend+'-capture',timeout=1800)
 row.update({'backend':backend,'directory':str(destination)})
 REPORT['captures'].append(row);save()
 print(backend,'capture exit',row['exit_code'],flush=True)
 if not row['valid_execution']:
  REPORT['status']='capture failed; no correctness-approved throughput summary';save();raise SystemExit(0)
REPORT['status']='CLI trials and captures complete; compare tokens/logits before reporting throughput'
save()
print(REPORT['status'],flush=True)
