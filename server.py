import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
import core

TOKEN = secrets.token_urlsafe(32)
STATE = {'busy': False, 'stage': 'Готово к работе', 'text': '', 'summary': '', 'files': {}, 'error': ''}
LOCK = threading.Lock()
ALLOWED = {'.txt', '.ogg', '.oga', '.opus', '.mp3', '.wav', '.m4a', '.flac', '.webm', '.mp4', '.mkv', '.mov', '.avi'}


def update(**values):
    with LOCK:
        STATE.update(values)


def process(path, name):
    transcript = ''
    files = {}
    try:
        transcript = (Path(path).read_text(encoding='utf-8-sig') if Path(path).suffix.lower() == '.txt'
                      else core.transcribe(path, lambda stage: update(stage=stage)))
        # Persist before starting the LLM, so a failed summary never loses recognition.
        files = core.export_result(core.ROOT / 'results', name, transcript, 'Конспект ещё не сформирован.')
        update(text=transcript, files=files)
        summary = core.summarize(transcript, lambda stage: update(stage=stage))
        Path(files['md']).write_text('# Конспект записи\n\n' + summary + '\n\n---\n\n# Полная расшифровка\n\n' + transcript, encoding='utf-8')
        update(summary=summary, stage='Готово — результаты сохранены')
    except Exception as error:
        update(error=str(error), stage='Ошибка. Расшифровка сохранена.' if files else 'Не удалось обработать запись')
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        finally:
            update(busy=False)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, status, body, content_type='application/json; charset=utf-8'):
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def permitted(self, token=True):
        expected = f'127.0.0.1:{self.server.server_address[1]}'
        host_ok = self.headers.get('Host') == expected
        origin_ok = self.headers.get('Origin') in (None, 'http://' + expected)
        token_ok = not token or secrets.compare_digest(self.headers.get('X-Token', ''), TOKEN)
        if not (host_ok and origin_ok and token_ok):
            self.send(403, {'error': 'Доступ запрещён'})
            return False
        return True

    def do_GET(self):
        path = urlsplit(self.path).path
        if not self.permitted(token=path != '/'):
            return
        if path == '/':
            self.send(200, (core.ROOT / 'index.html').read_text(encoding='utf-8').replace('__TOKEN__', TOKEN), 'text/html; charset=utf-8')
        elif path == '/status':
            with LOCK:
                state = dict(STATE)
            self.send(200, state)
        elif path in ('/download/txt', '/download/md'):
            with LOCK:
                filename = STATE['files'].get(path.rsplit('/', 1)[1])
            if filename:
                self.send(200, Path(filename).read_bytes(), 'text/plain; charset=utf-8')
            else:
                self.send(404, {'error': 'Нет результата'})
        else:
            self.send(404, {'error': 'Не найдено'})

    def do_POST(self):
        if not self.permitted():
            return
        parsed = urlsplit(self.path)
        if parsed.path == '/shutdown':
            with LOCK:
                busy = STATE['busy']
            if busy:
                self.send(409, {'error': 'Сначала дождитесь завершения обработки.'})
                return
            self.send(200, {'ok': True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if parsed.path != '/upload':
            self.send(404, {})
            return
        name = parse_qs(parsed.query).get('name', ['audio'])[0]
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0
        suffix = Path(name).suffix.lower()
        if length <= 0 or length > 512 * 1024 * 1024 or suffix not in ALLOWED:
            self.send(400, {'error': 'Выберите аудио или видео до 512 МБ: OGG, MP3, WAV, MP4, MKV, MOV, AVI, WebM.'})
            return
        with LOCK:
            if STATE['busy']:
                self.send(409, {'error': 'Дождитесь завершения текущей записи.'})
                return
            STATE.update(busy=True, stage='Читаю файл…', text='', summary='', files={}, error='')
        filename = None
        try:
            (core.ROOT / 'temp').mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=core.ROOT / 'temp', suffix=suffix, delete=False) as dest:
                filename = dest.name
                remaining = length
                self.connection.settimeout(60)
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError('Передача файла прервана.')
                    dest.write(chunk)
                    remaining -= len(chunk)
            threading.Thread(target=process, args=(filename, name), daemon=True).start()
        except Exception as error:
            if filename:
                Path(filename).unlink(missing_ok=True)
            update(busy=False, error=str(error))
            self.send(400, {'error': str(error)})
            return
        # The worker now owns its input; an HTTP disconnect must not cancel it.
        self.send(202, {'ok': True})


def make_server(port=8765):
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


if __name__ == '__main__':
    make_server().serve_forever()
