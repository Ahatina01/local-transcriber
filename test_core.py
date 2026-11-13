import tempfile
import unittest
from pathlib import Path
import core
import wave
import json
from unittest.mock import patch


class CoreTests(unittest.TestCase):
    def test_tasks_require_verbatim_evidence(self):
        result = {'summary': 'Встреча', 'points': ['Бюджет 120 тысяч.'], 'tasks': [
            {'quote': 'Анна подготовит предложение до пятницы.', 'owner': 'Анна', 'deadline': 'до пятницы'},
            {'quote': 'Утвердить бюджет завтра', 'owner': 'Иван', 'deadline': 'завтра'}], 'unresolved': []}
        with patch.object(core, 'ollama_call', return_value=json.dumps(result)):
            summary = core.summarize('Анна подготовит предложение до пятницы. Бюджет 120 тысяч.')
        self.assertIn('Анна подготовит предложение до пятницы.', summary)
        self.assertNotIn('Утвердить бюджет', summary)

    def test_deadline_preserved_when_model_spells_number_differently(self):
        source = 'Иван проверит расчеты завтра до 3 часов дня'
        raw = json.dumps({'summary': '', 'points': [], 'tasks': [
            {'quote': source, 'owner': 'Иван', 'deadline': 'завтра до трёх часов дня'}], 'unresolved': []})
        self.assertIn('срок: завтра до 3 часов дня', core.render_notes(raw, source))

    def test_decode_actual_wav_with_installed_av(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'тишина.wav'
            with wave.open(str(p), 'wb') as w:
                w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                w.writeframes(b'\x00\x00' * 16000)
            audio = core.decode_audio(p)
            self.assertEqual(len(audio), 16000)
            self.assertEqual(float(abs(audio).max()), 0)

    def test_corrupt_audio_has_readable_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'broken.ogg'
            p.write_bytes(b'not audio')
            with self.assertRaisesRegex(ValueError, 'повреждён'):
                core.decode_audio(p)

    def test_chunks_keep_all_text_including_long_word_and_tail(self):
        text = 'Привет. ' * 2000 + 'я' * 10000 + ' КОНЕЦ'
        chunks = core.split_text(text, 6000)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(c) <= 6000 for c in chunks))

    def test_export_never_overwrites_and_keeps_full_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            a = core.export_result(Path(d), '../тест.ogg', 'ПОЛНЫЙ ТЕКСТ', 'Итог')
            b = core.export_result(Path(d), '../тест.ogg', 'ВТОРОЙ', '')
            self.assertNotEqual(a, b)
            self.assertEqual(Path(a['txt']).read_text(encoding='utf-8'), 'ПОЛНЫЙ ТЕКСТ')
            self.assertIn('ПОЛНЫЙ ТЕКСТ', Path(a['md']).read_text(encoding='utf-8'))
            self.assertEqual(Path(a['txt']).parent.parent, Path(d))

    def test_offline_boundary_rejects_external_hosts(self):
        for host in ['example.com', '8.8.8.8', '127.0.0.1.evil.org', '0.0.0.0']:
            self.assertFalse(core.is_loopback(host))
        self.assertTrue(core.is_loopback('127.0.0.1'))
        self.assertTrue(core.is_loopback('::1'))

    def test_empty_summary_does_not_call_model(self):
        self.assertIn('не обнаружена', core.summarize('   '))


if __name__ == '__main__':
    unittest.main()
