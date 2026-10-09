"""Read GGUF headers only; verify expert descriptors against Strata's v2 fixture."""
import argparse, csv, hashlib, importlib.util, json, math, struct
from collections import Counter
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('models',type=Path)
p.add_argument('llama_source',type=Path)
p.add_argument('manifest',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('pinned_constants',a.llama_source/'gguf-py/gguf/constants.py')
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def number(f,fmt):
 n=struct.calcsize('<'+fmt)
 b=f.read(n)
 if len(b)!=n: raise ValueError('truncated header')
 return struct.unpack('<'+fmt,b)[0]
def string(f):
 n=number(f,'Q')
 b=f.read(n)
 if len(b)!=n: raise ValueError('truncated string')
 return b.decode('utf-8')
def value(f,ty):
 if ty==8: return string(f)
 if ty!=9: return number(f,formats[ty])
 subtype=number(f,'I'); count=number(f,'Q'); start=f.tell()
 preview=[]
 for i in range(count):
  v=value(f,subtype)
  if i<8: preview.append(v)
 end=f.tell(); f.seek(start); digest=hashlib.sha256(f.read(end-start)).hexdigest(); f.seek(end)
 return {'array_type':subtype,'count':count,'preview':preview,'raw_payload_sha256':digest}
all_tensors=[]; headers=[]
for path in sorted(a.models.glob('*.gguf')):
 with path.open('rb') as f:
  magic=f.read(4)
  if magic!=b'GGUF': raise ValueError('not little-endian GGUF')
  version=number(f,'I'); nt=number(f,'Q'); nk=number(f,'Q')
  metadata={}
  for _ in range(nk):
   key=string(f); ty=number(f,'I'); metadata[key]={'type':ty,'value':value(f,ty)}
  tensors=[]
  for _ in range(nt):
   name=string(f); ndim=number(f,'I'); dims=[number(f,'Q') for _ in range(ndim)]
   ty=number(f,'I'); offset=number(f,'Q')
   enum=c.GGMLQuantizationType(ty); block,byte_size=c.GGML_QUANT_SIZES[enum]
   elements=math.prod(dims)
   if elements%block: raise ValueError('incomplete quant block')
   tensors.append({'file':path.name,'name':name,'dimensions':dims,'type':ty,'dtype':enum.name,
                   'relative_offset':offset,'bytes':elements//block*byte_size})
  alignment=metadata.get('general.alignment',{}).get('value',32)
  header_end=f.tell(); data_start=(header_end+alignment-1)//alignment*alignment
  f.seek(0); raw=f.read(data_start)
  (a.output/(path.name+'.header.bin')).write_bytes(raw)
  for t in tensors: t['absolute_offset']=data_start+t['relative_offset']
  header={'file':path.name,'magic':'GGUF','version':version,'endian':'little','tensor_count':nt,
          'metadata_count':nk,'alignment':alignment,'header_end':header_end,'data_start':data_start,
          'raw_header_sha256':hashlib.sha256(raw).hexdigest(),'metadata':metadata,
          'tensor_bytes':sum(t['bytes'] for t in tensors),'dtype_counts':dict(Counter(t['dtype'] for t in tensors))}
  headers.append(header); all_tensors+=tensors
  print(path.name,'architecture',metadata.get('general.architecture',{}).get('value','absent in this split shard'),'tensors',nt,'data_start',data_start)
(a.output/'gguf-headers.json').write_text(json.dumps(headers,indent=2,ensure_ascii=False)+'\n')
(a.output/'tensors.json').write_text(json.dumps(all_tensors,indent=2)+'\n')
with (a.output/'tensors.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(all_tensors[0]),lineterminator='\n'); w.writeheader(); w.writerows(all_tensors)
by_name={t['name']:t for t in all_tensors}
checks=[]; packed_offset=0
for line in a.manifest.read_text().splitlines():
 if not line or line.startswith('#'): continue
 layer,gu,down,offset,blob,gate_abs,up_abs,down_abs=map(int,line.split())
 errors=[]; sizes=[]; actual={}
 for role,ty,want_abs in [('gate',gu,gate_abs),('up',gu,up_abs),('down',down,down_abs)]:
  name=f'blk.{layer}.ffn_{role}_exps.weight'; t=by_name.get(name)
  if not t: errors.append('missing '+name); continue
  dims=[640,2560,512] if role=='down' else [2560,640,512]
  if t['type']!=ty: errors.append(role+' type mismatch')
  if t['dimensions']!=dims: errors.append(role+' shape mismatch')
  if t['absolute_offset']!=want_abs: errors.append(role+' absolute offset mismatch')
  sizes.append(t['bytes']); actual[role]=t
 if sum(sizes)!=blob*512: errors.append('expert size mismatch')
 if offset!=packed_offset: errors.append('packed offset mismatch')
 packed_offset+=blob*512
 checks.append({'layer':layer,'passed':not errors,'errors':errors,'tensors':actual})
result={'manifest':str(a.manifest),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        'layers':len(checks),'experts_per_layer':512,'expert_bytes':packed_offset,
        'passed':len(checks)==48 and all(r['passed'] for r in checks),'checks':checks}
(a.output/'manifest-check.json').write_text(json.dumps(result,indent=2)+'\n')
print('manifest',result['passed'],'layers',len(checks),'expert bytes',packed_offset)
if not result['passed']: raise SystemExit(1)
