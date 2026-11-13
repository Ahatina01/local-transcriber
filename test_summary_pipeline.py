import json
import tempfile
import unittest
from unittest.mock import patch
import core
from summary_pipeline import summarize


class PipelineTests(unittest.TestCase):
    def test_split_merge_and_resume_preserve_tasks(self):
        calls = []
        source = 'Анна сдаст отчёт завтра. ' * 180 + 'Иван проверит проект в пятницу.'
        def model(text, prompt):
            calls.append(text)
            if len(text) > 1800 and 'единое' not in prompt:
                raise core.SummaryLimitError()
            merged = 'единое' in prompt
            return json.dumps({'summary': 'Единый итог' if merged else 'Промежуточный итог',
                'points': [], 'unresolved': [], 'tasks': [
                    {'quote': 'Иван проверит проект в пятницу.', 'owner': 'Иван', 'deadline': 'в пятницу'}
                ] if 'Иван проверит проект в пятницу.' in text and not merged else []})
        with tempfile.TemporaryDirectory() as d, patch.object(core, 'ollama_call', side_effect=model):
            result = summarize(source, lambda _: None, d)
            self.assertIn('Единый итог', result)
            self.assertIn('Иван проверит проект в пятницу.', result)
            first_calls = len(calls)
            self.assertEqual(result, summarize(source, lambda _: None, d))
            # Oversized parents can fail again; successful children/merge must not regenerate.
            self.assertTrue(all(len(t) > 1800 for t in calls[first_calls:]))

    def test_failure_is_bounded_and_successful_chunks_survive(self):
        with tempfile.TemporaryDirectory() as d, patch.object(core, 'ollama_call', side_effect=core.SummaryLimitError()) as call:
            with self.assertRaisesRegex(RuntimeError, 'Готовые части сохранены'):
                summarize('слово ' * 600, lambda _: None, d)
            self.assertLess(call.call_count, 10)
