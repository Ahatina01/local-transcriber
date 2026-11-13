import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from settings import runtime_directory

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
(ROOT / 'logs').mkdir(exist_ok=True)
log = open(ROOT / 'logs' / 'launcher.log', 'a', encoding='utf-8', buffering=1)
sys.stdout = sys.stderr = log
URL = 'http://127.0.0.1:8765'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url):
    with opener.open(url, timeout=2) as r:
        return r.read()


def main():
    try:
        if 'Локальная расшифровка'.encode() in get(URL):
            webbrowser.open(URL)
            return
        raise RuntimeError('Порт 8765 занят другой программой.')
    except urllib.error.URLError:
        pass
    runtime = runtime_directory(ROOT)
    env = os.environ.copy()
    env.update(OLLAMA_MODELS=str(runtime / 'ollama-models'), OLLAMA_HOST='127.0.0.1:11435',
               OLLAMA_NO_CLOUD='1', OLLAMA_NUM_PARALLEL='1', OLLAMA_MAX_LOADED_MODELS='1',
               OLLAMA_CONTEXT_LENGTH='8192', OLLAMA_KEEP_ALIVE='0', OLLAMA_NOPRUNE='1')
    # This instance is owned by this app and receives the privacy environment at creation.
    try:
        get('http://127.0.0.1:11435/api/version')
    except urllib.error.URLError:
        pass
    else:
        raise RuntimeError('Порт 11435 занят. Закройте прежний запуск программы перед повторным запуском.')
    ollama_log = open(ROOT / 'logs' / 'ollama.log', 'a', encoding='utf-8')
    child = subprocess.Popen([str(runtime / 'ollama' / 'ollama.exe'), 'serve'], env=env,
                             stdout=ollama_log, stderr=ollama_log, creationflags=subprocess.CREATE_NO_WINDOW)
    atexit.register(child.terminate)
    deadline = time.monotonic() + 45
    while True:
        if child.poll() is not None:
            raise RuntimeError('Ollama не запустилась. Подробности в logs/ollama.log.')
        try:
            tags = json.loads(get('http://127.0.0.1:11435/api/tags'))
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() > deadline:
                raise RuntimeError('Ollama не ответила за 45 секунд.')
            time.sleep(.5)
    if not any(m['name'] == 'qwen3:4b-instruct' for m in tags['models']):
        raise RuntimeError('Не найдена существующая модель qwen3:4b-instruct.')
    import server
    httpd = server.make_server()
    threading.Timer(.8, lambda: webbrowser.open(URL)).start()
    print('Ready: ' + URL, flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        traceback.print_exc()
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, str(error), 'Локальная расшифровка — ошибка запуска', 0x10)
