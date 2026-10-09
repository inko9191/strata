"""Same-day interleaved short-prompt comparison, retaining Phase 0.5 CLI settings."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from run_long import BAD

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
with (ROOT / 'build-mac-phase05/reference.lock').open('a+') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another reference runner holds the lock; no tests started')
    settings = json.loads((ROOT / 'bench/results/2026-10-09-mac-phase05/prompts.json').read_text())
    prompt = next(p['text'] for p in settings['prompts'] if p['id'] == 'english')
    rows = []
    for trial in range(1, 4):
        for version in (['pinned', 'v0.6.0'] if trial % 2 else ['v0.6.0', 'pinned']):
            build = ROOT / ('build-mac-phase05' if version == 'pinned' else 'build-mac-phase06')
            command = [str(build / 'llama-metal/bin/llama-cli'), '-m',
                str(ROOT / 'build-mac-phase05/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf'),
                '-ngl', '99', '-ot', 'per_layer_token_embd.weight=CPU', '-c', '512', '-b', '64',
                '-ub', '64', '-t', '8', '-tb', '8', '--fit', 'off', '--lazy-mode', 'on',
                '--flash-attn', 'off', '--no-warmup', '--offline', '--single-turn', '--temp', '0',
                '--seed', '1234', '--repeat-penalty', '1.0', '-n', '32', '--jinja',
                '--chat-template-file', str(ROOT / 'bench/results/2026-10-09-mac-phase05/raw-template.jinja'),
                '-p', prompt, '--log-verbosity', '3', '--log-colors', 'off']
            prefix = f'short-{version}-{trial}'
            with (OUT / (prefix + '.stdout.log')).open('w') as stdout, (OUT / (prefix + '.stderr.log')).open('w') as stderr:
                process = subprocess.Popen(['/usr/bin/time', '-l'] + command, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise RuntimeError('timeout')
            text = (OUT / (prefix + '.stderr.log')).read_text(errors='replace')
            row = {'version': version, 'trial': trial, 'command': command, 'exit_code': code,
                   'gpu_errors': BAD.findall(text)}
            for kind, label in [('prompt', 'prompt eval time'), ('decode', 'eval time')]:
                timing_text = text if kind == 'prompt' else '\n'.join(line for line in text.splitlines() if 'prompt eval time' not in line)
                matches = re.findall(r'(?m)^.*?\b' + label + r'\s*=\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second', timing_text)
                if not matches:
                    raise RuntimeError('Missing detailed timings')
                ms, tokens, rate = matches[-1]
                row[kind + '_ms'] = float(ms)
                row[kind + '_tokens'] = int(tokens)
                row[kind + '_tok_s'] = float(rate)
            row['peak_rss_bytes'] = int(re.findall(r'(\d+)\s+maximum resident set size', text)[-1])
            row['valid_execution'] = code == 0 and not row['gpu_errors']
            rows.append(row)
            (OUT / 'short-trials.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(prefix, row['prompt_tok_s'], row['decode_tok_s'], flush=True)
            if not row['valid_execution']:
                raise RuntimeError('Invalid CLI trial')
