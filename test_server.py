import http.client
import threading
import unittest
import io
import tempfile
from pathlib import Path
from unittest.mock import patch
import server


class ServerTests(unittest.TestCase):
    def test_txt_reuses_transcript_without_recognition(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'input.txt'
            source.write_text('Полный текст записи', encoding='utf-8-sig')
            with patch.object(server.core, 'ROOT', root), patch.object(server.core, 'transcribe') as stt, patch.object(server.core, 'summarize', return_value='Общий итог'):
                server.process(str(source), 'input.txt')
                stt.assert_not_called()
                self.assertEqual(server.STATE['text'], 'Полный текст записи')
                self.assertEqual(server.STATE['summary'], 'Общий итог')
                self.assertIn('Полный текст записи', Path(server.STATE['files']['md']).read_text(encoding='utf-8'))
        server.update(text='', summary='', files={}, error='')

    @classmethod
    def setUpClass(cls):
        cls.httpd = server.make_server(0)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port)
        conn.request(method, path, body, headers or {})
        r = conn.getresponse()
        status, data = r.status, r.read()
        conn.close()
        return status, data

    def test_foreign_origin_cannot_upload(self):
        status, _ = self.request('POST', '/upload?name=x.wav', b'123',
                                 {'Origin': 'https://evil.example', 'X-Token': server.TOKEN})
        self.assertEqual(status, 403)

    def test_missing_token_cannot_read_transcript(self):
        self.assertEqual(self.request('GET', '/status')[0], 403)

    def test_local_page_and_authenticated_status(self):
        self.assertEqual(self.request('GET', '/')[0], 200)
        self.assertEqual(self.request('GET', '/status', headers={'X-Token': server.TOKEN})[0], 200)

    def test_empty_upload_rejected(self):
        self.assertEqual(self.request('POST', '/upload?name=x.wav', b'', {'X-Token': server.TOKEN})[0], 400)

    def test_bad_host_rejected(self):
        self.assertEqual(self.request('GET', '/', headers={'Host': 'evil.example'})[0], 403)

    def test_disconnect_after_upload_does_not_delete_worker_input(self):
        handler = object.__new__(server.Handler)
        handler.path = '/upload?name=test.wav'
        handler.headers = {'Content-Length': '3'}
        handler.rfile = io.BytesIO(b'abc')
        handler.connection = type('Connection', (), {'settimeout': lambda *_: None})()
        handler.permitted = lambda: True
        def disconnected(*args):
            raise BrokenPipeError('browser disconnected')
        handler.send = disconnected
        server.update(busy=False)
        with patch.object(server.threading, 'Thread') as thread:
            try:
                with self.assertRaises(BrokenPipeError):
                    handler.do_POST()
                path = Path(thread.call_args.kwargs['args'][0])
                self.assertTrue(path.exists())
                self.assertTrue(server.STATE['busy'])
            finally:
                if thread.call_args:
                    Path(thread.call_args.kwargs['args'][0]).unlink(missing_ok=True)
                server.update(busy=False)


if __name__ == '__main__':
    unittest.main()
