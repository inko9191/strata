"""GGUF header-only inventory for the separately downloaded MTP head."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
source = ROOT / 'build-mac-phase06/llama.cpp-d81235049384534c167caea52b85a694f6103d14'
spec = importlib.util.spec_from_file_location('gguf_constants', source / 'gguf-py/gguf/constants.py')
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)


def number(file, fmt):
    return struct.unpack('<' + fmt, file.read(struct.calcsize('<' + fmt)))[0]


def string(file):
    return file.read(number(file, 'Q')).decode('utf-8')


def value(file, kind):
    if kind == 8:
        return string(file)
    if kind == 9:
        subtype, count = number(file, 'I'), number(file, 'Q')
        return [value(file, subtype) for _ in range(count)]
    return number(file, {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}[kind])


path = ROOT / 'build-mac-phase06/mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
with path.open('rb') as file:
    assert file.read(4) == b'GGUF'
    version, count, kv_count = number(file, 'I'), number(file, 'Q'), number(file, 'Q')
    metadata = {}
    for _ in range(kv_count):
        key, kind = string(file), number(file, 'I')
        data = value(file, kind)
        if isinstance(data, list) and len(data) > 64:
            data = {'count': len(data), 'preview': data[:8],
                    'decoded_json_sha256': hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()}
        metadata[key] = data
    tensors = []
    for _ in range(count):
        name = string(file)
        dims = [number(file, 'Q') for _ in range(number(file, 'I'))]
        dtype, offset = number(file, 'I'), number(file, 'Q')
        quant = constants.GGMLQuantizationType(dtype)
        block, size = constants.GGML_QUANT_SIZES[quant]
        assert math.prod(dims) % block == 0
        tensors.append({'name': name, 'dimensions': dims, 'dtype': quant.name,
                        'offset': offset, 'bytes': math.prod(dims) // block * size})
    alignment = metadata.get('general.alignment', 32)
    data_start = (file.tell() + alignment - 1) // alignment * alignment
    file.seek(0)
    raw = file.read(data_start)
(OUT / 'mtp-header.bin').write_bytes(raw)
(OUT / 'mtp-header.json').write_text(json.dumps({'file': path.name, 'version': version,
    'architecture': metadata.get('general.architecture'), 'tensor_count': count, 'metadata_count': kv_count,
    'data_start': data_start, 'header_sha256': hashlib.sha256(raw).hexdigest(),
    'metadata': metadata, 'tensors': tensors}, indent=2, ensure_ascii=False) + '\n')
print('MTP GGUF', metadata.get('general.architecture'), 'tensors', count)
