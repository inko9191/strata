"""Build external diagnostics from the original v0.6.0 build's object files.

No source file or Phase 0.6 artifact is overwritten. Version-specific, macOS only.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BUILD = ROOT / 'build-mac-phase06/llama-metal'
SOURCE = ROOT / 'build-mac-phase06/llama.cpp-d81235049384534c167caea52b85a694f6103d14'
RAW = ROOT / 'build-mac-phase07'


def build():
    RAW.mkdir(exist_ok=True)
    shutil.copytree(BUILD/'bin', RAW/'bin', dirs_exist_ok=True)
    commands = []
    for name, directory in [('ggml-metal', BUILD/'ggml/src/ggml-metal'),
                            ('ggml-cpu', BUILD/'ggml/src')]:
        command = shlex.split((directory/f'CMakeFiles/{name}.dir/link.txt').read_text())
        command.insert(1, '-Wl,-interposable')
        target = RAW/'bin'/Path(command[command.index('-o')+1]).name
        command[command.index('-o')+1] = str(target)
        subprocess.run(command, cwd=directory, check=True)
        # Existing copies may have been dereferenced in an earlier pilot.
        for alias in [RAW/'bin'/f'lib{name}.0.dylib', RAW/'bin'/f'lib{name}.dylib']:
            if not alias.is_symlink():
                shutil.copyfile(target, alias)
        commands.append({'cwd': str(directory), 'command': command})
    command = ['clang++', '-std=c++17', '-O2', '-dynamiclib', '-fblocks',
               str(OUT/'metal_probe.mm'), '-o', str(RAW/'metal_probe.dylib')]
    for directory in ['include', 'src', 'ggml/include', 'ggml/src', 'ggml/src/ggml-metal']:
        command.extend(['-I', str(SOURCE/directory)])
    command.extend(['-L', str(BUILD/'bin'), '-Wl,-rpath,'+str(BUILD/'bin'),
                    '-lllama', '-lggml-metal', '-lggml-cpu', '-lggml-base',
                    '-framework', 'Metal', '-framework', 'Foundation'])
    subprocess.run(command, check=True)
    commands.append({'command': command})
    (RAW/'probe-build.json').write_text(json.dumps({
        'commands': commands,
        'source_sha256': hashlib.sha256((OUT/'metal_probe.mm').read_bytes()).hexdigest(),
        'binary_sha256': hashlib.sha256((RAW/'metal_probe.dylib').read_bytes()).hexdigest()
    }, indent=2)+'\n')


if __name__ == '__main__':
    with (ROOT/'build-mac-phase05/reference.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A reference run is active; nothing rebuilt')
        build()
