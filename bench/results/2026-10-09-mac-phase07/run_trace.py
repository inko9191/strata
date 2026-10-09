"""External Instruments capture; leaves both inference engines unchanged."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RAW = ROOT / 'build-mac-phase07'
spec = importlib.util.spec_from_file_location('phase06', OUT.parent / '2026-10-09-mac-phase06/run_long.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def session(config, label, outputs):
    command = json.loads((OUT.parent / '2026-10-09-mac-phase06' / f'{config}-r1-session.json').read_text())['command']
    log_path = OUT / f'{label}.log'
    record = {'config': config, 'label': label, 'command': command, 'output_tokens': outputs,
              'env': {'LLAMA_TRACE': '1'}, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    env = dict(os.environ, LLAMA_TRACE='1')
    control = RAW / f'{label}.active'
    if args.probe:
        if control.exists():
            raise RuntimeError('Stale diagnostic control file; use a fresh label')
        env.update(DYLD_INSERT_LIBRARIES=str(RAW / 'metal_probe.dylib'),
                   DYLD_FORCE_FLAT_NAMESPACE='1',
                   DYLD_LIBRARY_PATH=str(RAW / 'bin'),
                   PHASE07_CONTROL=str(control), PHASE07_OUTPUT=str(RAW / f'{label}.tsv'))
        if args.probe == 'stage':
            env['PHASE07_STAGE'] = '1'
        record['env'].update({key: value for key, value in env.items() if key.startswith(('PHASE07_', 'DYLD_'))})
        record['probe_source_sha256'] = hashlib.sha256((OUT / 'metal_probe.mm').read_bytes()).hexdigest()
        record['probe_binary_sha256'] = hashlib.sha256((RAW / 'metal_probe.dylib').read_bytes()).hexdigest()
    with log_path.open('w') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)
        trace = None
        try:
            start = time.monotonic()
            while time.monotonic() - start < 600:
                if server.poll() is not None or reference.BAD.search(log_path.read_text(errors='replace')):
                    raise RuntimeError('Server/GPU failure while loading')
                try:
                    if reference.http('/health', timeout=2).get('status') == 'ok':
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Server readiness timeout')
            prompt = (ROOT / 'bench/results/2026-10-06-llamacpp-0.6.0/prompt-4k.txt').read_text()
            reference.chat('Say hello.', 0, 16)
            reference.chat(prompt[:6000], 0, 32)
            # Same text before/under/after profiling; caching is disabled in reference.chat.
            prompt = '[bench mac-phase07 4k]\n' + prompt + '\n\nSummarize the text above in three sentences.'
            body, response, wall = reference.chat(prompt, 0, outputs)
            record['before'] = {'response': response, 'wall_seconds': wall}
            trace_command = ['xcrun', 'xctrace', 'record', '--template', args.template,
                             '--time-limit', f'{args.trace_seconds}s', '--output', str(RAW / f'{label}.trace'),
                             '--attach', str(server.pid)]
            record['trace_command'] = trace_command
            with (OUT / f'{label}-xctrace.log').open('w') as trace_log:
                if not args.probe:
                    trace = subprocess.Popen(trace_command, stdout=trace_log, stderr=subprocess.STDOUT)
                    time.sleep(5)
                    if trace.poll() is not None:
                        raise RuntimeError('Trace exited before request; see xctrace log')
                else:
                    control.touch()
                before = reference.metrics()
                log_start = log_path.stat().st_size
                body, response, wall = reference.chat(prompt, 0, outputs)
                record['profiled'] = {'request': body, 'response': response, 'wall_seconds': wall,
                                      'metrics_before': before, 'metrics_after': reference.metrics()}
                with log_path.open('rb') as file:
                    file.seek(log_start)
                    record['profiled_log'] = file.read().decode(errors='replace')
                if trace is not None:
                    trace.wait(timeout=300)
                    record['trace_exit_code'] = trace.returncode
                else:
                    control.unlink()
                    record['trace_exit_code'] = 0
            body, response, wall = reference.chat(prompt, 0, outputs)
            record['after'] = {'response': response, 'wall_seconds': wall}
            record['gpu_errors'] = reference.BAD.findall(log_path.read_text(errors='replace'))
            record['response_text_equal_before_profiled_after'] = len({record[key]['response']['choices'][0]['message'].get('content', '') for key in ['before', 'profiled', 'after']}) == 1
            if args.probe:
                raw = RAW / f'{label}.tsv'
                record['raw_probe_bytes'] = raw.stat().st_size if raw.exists() else 0
                if not record['raw_probe_bytes']:
                    raise RuntimeError('No diagnostic events were captured')
            if record['gpu_errors'] or server.poll() is not None or record['trace_exit_code'] != 0:
                raise RuntimeError('Invalid capture')
        except Exception as exc:
            record['failure'] = repr(exc)
            raise
        finally:
            if trace is not None and trace.poll() is None:
                trace.send_signal(signal.SIGINT)
                try:
                    trace.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    trace.kill()
                    trace.wait(timeout=10)
                    record['trace_cleanup'] = 'killed after SIGINT timeout'
            if control.exists():
                control.unlink()
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            record['server_exit_after_shutdown'] = server.returncode
            (OUT / f'{label}.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(label, record.get('failure', 'complete'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=['off', 'mtp3'])
    parser.add_argument('label')
    parser.add_argument('--outputs', type=int, default=64)
    parser.add_argument('--template', default='Metal System Trace')
    parser.add_argument('--trace-seconds', type=int, default=25)
    parser.add_argument('--probe', choices=['passive', 'stage'])
    args = parser.parse_args()
    RAW.mkdir(exist_ok=True)
    with (ROOT / 'build-mac-phase05/reference.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another reference runner holds the lock; nothing started')
        session(args.config, args.label, args.outputs)
