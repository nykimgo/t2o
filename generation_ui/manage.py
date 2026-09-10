"""Start/stop only this service; persistent state lives outside source files."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]/'.runtime'
PID=RUNTIME/'generation-studio.pid'


def ours(pid):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return str(HERE/'app.py').encode() in args
    except OSError:
        return False


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['start','stop','status'])
    p.add_argument('--host',default='10.254.62.200')
    p.add_argument('--port',type=int,default=8766)
    p.add_argument('--gpu',default='1')
    args=p.parse_args()
    RUNTIME.mkdir(exist_ok=True)
    pid=int(PID.read_text()) if PID.exists() else None
    alive=bool(pid and ours(pid))
    if args.action=='status':
        print('running' if alive else 'stopped')
        return
    if args.action=='stop':
        if alive:
            os.kill(pid,signal.SIGTERM)
            for _ in range(30):
                if not ours(pid):
                    break
                time.sleep(.5)
            if ours(pid):
                raise RuntimeError('Server is still shutting down; no unrelated process was stopped.')
        PID.unlink(missing_ok=True)
        print('stopped')
        return
    if alive:
        print(f'already running: pid {pid}')
        return
    with (RUNTIME/'generation-studio.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,str(HERE/'app.py'),'--host',args.host,'--port',str(args.port),'--gpu',args.gpu],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=HERE.parent)
    PID.write_text(str(child.pid))
    time.sleep(1)
    if child.poll() is not None:
        raise RuntimeError('Server did not start. See .runtime/generation-studio.log')
    print(f'http://{args.host}:{args.port}')


if __name__=='__main__':
    main()
