"""Start/reuse the local digit UI and open the browser. No training required."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def ready(url):
    try:
        with urllib.request.urlopen(url+'/api/info', timeout=2) as response:
            info = json.load(response)
        return (info.get('service') == 'lenet-digit-lab' and info.get('ui_version',0) >= 2
                and info.get('test_size') == 10000
                and set(info.get('variants', [])) == {'baseline', 'dw_c3', 'dw', 'se', 'dw_se'})
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7872)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    url = f'http://127.0.0.1:{args.port}'
    if not ready(url):
        with socket.socket() as check:
            check.settimeout(1)
            if check.connect_ex(('127.0.0.1', args.port)) == 0:
                raise RuntimeError(f'Port {args.port} is used by another service. Run with --port 7873.')
        logs = ROOT/'logs'; logs.mkdir(exist_ok=True)
        log = logs/'visual_interface.log'
        executable = Path(sys.executable)
        if executable.name.lower() == 'pythonw.exe':
            executable = executable.with_name('python.exe')
        with log.open('ab') as output:
            process = subprocess.Popen(
                [str(executable), str(ROOT/'app.py'), '--port', str(args.port)],
                cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for _ in range(120):
            if ready(url):
                break
            if process.poll() is not None:
                raise RuntimeError(f'Interface startup failed. See {log}')
            time.sleep(.25)
        else:
            raise RuntimeError(f'Interface startup timed out. See {log}')
    if not args.no_browser and not webbrowser.open(url):
        raise RuntimeError(f'Browser could not open automatically. Open {url}')
    print(f'CNN visual interface: {url}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        message = f'{error}\n\nProject: {ROOT}'
        if sys.executable.lower().endswith('pythonw.exe'):
            import tkinter as tk
            from tkinter import messagebox
            window=tk.Tk();window.withdraw()
            messagebox.showerror('CNN interface startup failed', message)
            window.destroy()
        else:
            print(message, file=sys.stderr)
        sys.exit(1)
