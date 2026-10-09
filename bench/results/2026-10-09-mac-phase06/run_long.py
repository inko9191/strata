"""Serial, locked llama.cpp reference measurements; no Strata engine changes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BUILD = ROOT / 'build-mac-phase06'
PORT = 8196
BASE = f'http://127.0.0.1:{PORT}'
BAD = re.compile(r'(?im)^.*(?:Insufficient Memory|command buffer.*(?:failed|error)|error loading model|GGML_ASSERT|GGML_ABORT|failed to (?:allocate|create.*context)|Metal.*(?:out of memory|error)).*$')


def http(path, body=None, timeout=1800):
    request = urllib.request.Request(BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def metrics():
    with urllib.request.urlopen(BASE + '/metrics', timeout=10) as response:
        text = response.read().decode()
    return {key: float(value) for key, value in
            re.findall(r'^([\w:]+) ([0-9.eE+-]+)$', text, re.M)}


class Memory:
    def __init__(self, parent):
        self.parent = parent
        self.stop = threading.Event()
        self.peak = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stop.is_set():
            result = subprocess.run(['ps', '-axo', 'pid=,ppid=,rss='], capture_output=True, text=True)
            entries = [tuple(map(int, line.split())) for line in result.stdout.splitlines() if line.strip()]
            ids = {self.parent}
            while True:
                enlarged = ids | {pid for pid, ppid, _ in entries if ppid in ids}
                if enlarged == ids:
                    break
                ids = enlarged
            self.peak = max(self.peak, sum(rss * 1024 for pid, _, rss in entries if pid in ids))
            self.stop.wait(0.5)

    def close(self):
        self.stop.set()
        self.thread.join()


def chat(prompt, temp, count=256):
    body = {'model': 'x', 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': count, 'temperature': temp, 'seed': 1, 'stream': False,
            'cache_prompt': False, 'ignore_eos': True, 'return_tokens': True,
            'chat_template_kwargs': {'enable_thinking': False}}
    if temp:
        body.update(top_k=20, top_p=0.95)
    start = time.monotonic()
    response = http('/v1/chat/completions', body)
    return body, response, time.monotonic() - start


def run_session(config, rnd, smoke=False):
    name = f'{config}-r{rnd}' + ('-smoke' if smoke else '')
    if args.verbose:
        name += '-verbose'
    log_path = OUT / (name + '.log')
    model = ROOT / 'build-mac-phase05/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf'
    command = [str(BUILD / 'llama-metal/bin/llama-server'), '-m', str(model),
               '--host', '127.0.0.1', '--port', str(PORT), '-np', '1', '--no-webui', '--jinja',
               '-ngl', '99', '-ot', 'per_layer_token_embd.weight=CPU', '--fit', 'off',
               '--lazy-mode', 'on', '-b', '2048', '-ub', '2048', '-c', '36864',
               '-ctk', 'q8_0', '-ctv', 'q8_0', '-fa', 'on', '-t', '8', '-tb', '8',
               '--metrics', '--no-warmup', '--log-colors', 'off']
    if args.verbose:
        command += ['--log-verbosity', '4']
    if config == 'mtp3':
        command += ['-md', str(BUILD / 'mtp-Qwen3.8-Flash-Next-Q8_0.gguf'),
                    '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '-ngld', '99']
    env = dict(os.environ, LLAMA_TRACE='1')
    session = {'config': config, 'round': rnd, 'command': command, 'env': {'LLAMA_TRACE': '1'},
               'log': log_path.name, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with log_path.open('w') as log:
        process = subprocess.Popen(['/usr/bin/time', '-l'] + command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        try:
            start = time.monotonic()
            while time.monotonic() - start < 600:
                if process.poll() is not None:
                    raise RuntimeError('Server exited during model loading')
                if BAD.search(log_path.read_text(errors='replace')):
                    raise RuntimeError('GPU/model error during loading')
                try:
                    if http('/health', timeout=3).get('status') == 'ok':
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Server readiness timeout')
            session['load_seconds'] = time.monotonic() - start
            warmups = []
            for prompt, count in [('Say hello.', 16), (PROMPTS['4k'][:6000], 32)]:
                body, response, wall = chat(prompt, 0, count)
                warmups.append({'body': body, 'response': response, 'wall_seconds': wall})
            (OUT / (name + '-warmups.json')).write_text(json.dumps(warmups, indent=2, ensure_ascii=False) + '\n')
            cells = [('4k', 2), ('32k', 1)] if not smoke else [('4k', 1)]
            for cell, reps in cells:
                for temp in ([0.0, 0.7] if not smoke else [0.0]):
                    for rep in range(reps):
                        # Same nonce across the paired configurations; unique between cells/rounds.
                        nonce = f'[bench mac-phase06 r{rnd} {cell} t{temp:.1f} k{rep}]\n'
                        prompt = nonce + PROMPTS[cell] + '\n\nSummarize the text above in three sentences.'
                        prefix = f'{name}-{cell}-t{temp:.1f}-k{rep}'
                        before = metrics()
                        log_start = log_path.stat().st_size
                        sampler = Memory(process.pid)
                        print('START', prefix, flush=True)
                        try:
                            body, response, wall = chat(prompt, temp, 256)
                        finally:
                            sampler.close()
                        after = metrics()
                        with log_path.open('rb') as file:
                            file.seek(log_start)
                            text = file.read().decode(errors='replace')
                        timings = response.get('timings', {})
                        row = {'config': config, 'round': rnd, 'cell': cell, 'temperature': temp, 'rep': rep,
                               'wall_seconds': wall, 'timings': timings, 'usage': response.get('usage'),
                               'request_rss_peak_sampled_bytes': sampler.peak,
                               'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                               'gpu_errors': BAD.findall(text),
                               'metrics_delta': {key: after[key] - before.get(key, 0) for key in after},
                               'verification_events': [{'accepted': int(a), 'drafted': int(d),
                                   'restore_checkpoint': bool(checkpoint)} for a, d, checkpoint in
                                   re.findall(r'accepted\s+(\d+)/\s*(\d+) draft tokens( \(restore checkpoint\))?', text)],
                               'response_file': prefix + '.json'}
                        row['valid_execution'] = not row['gpu_errors'] and process.poll() is None and bool(timings)
                        (OUT / row['response_file']).write_text(json.dumps({'request': body, 'response': response},
                                                                         indent=2, ensure_ascii=False) + '\n')
                        with (OUT / ('smoke.jsonl' if smoke else 'trials.jsonl')).open('a') as output:
                            output.write(json.dumps(row, ensure_ascii=False) + '\n')
                        print('DONE', prefix, 'pp', timings.get('prompt_per_second'), 'decode',
                              timings.get('predicted_per_second'), 'tokens', timings.get('predicted_n'),
                              'draft', timings.get('draft_n_accepted'), '/', timings.get('draft_n'), flush=True)
                        if not row['valid_execution']:
                            raise RuntimeError('Invalid request, see saved row')
        except Exception as exc:
            session['failure'] = repr(exc)
            raise
        finally:
            if process.poll() is None:
                listing = subprocess.run(['ps', '-axo', 'pid=,ppid='], capture_output=True, text=True)
                children = [int(line.split()[0]) for line in listing.stdout.splitlines()
                            if len(line.split()) == 2 and int(line.split()[1]) == process.pid]
                # Keep the time wrapper alive so it reports the server's maximum RSS.
                if children:
                    for pid in children:
                        os.kill(pid, signal.SIGTERM)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            session['exit_code_after_shutdown'] = process.returncode
            text = log_path.read_text(errors='replace')
            session['gpu_errors'] = BAD.findall(text)
            match = re.findall(r'(\d+)\s+maximum resident set size', text)
            session['time_l_maximum_rss_bytes'] = int(match[-1]) if match else None
            (OUT / (name + '-session.json')).write_text(json.dumps(session, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', choices=['off', 'mtp3'])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--verbose', action='store_true', help='Extra diagnostic logging; keep separate from measured rounds')
    args = parser.parse_args()
    with (ROOT / 'build-mac-phase05/reference.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another reference runner holds the lock; no tests started')
        PROMPTS = {cell: (ROOT / f'bench/results/2026-10-06-llamacpp-0.6.0/prompt-{cell}.txt').read_text()
                   for cell in ['4k', '32k']}
        if args.smoke:
            run_session(args.smoke, 0, True)
        else:
            for rnd in range(1, args.rounds + 1):
                for config in (['off', 'mtp3'] if rnd % 2 else ['mtp3', 'off']):
                    run_session(config, rnd)
