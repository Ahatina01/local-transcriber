import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from settings import runtime_directory


class SettingsTests(unittest.TestCase):
    def test_local_configuration_and_environment_override(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'TRANSCRIBER_RUNTIME': ''}):
            root = Path(d)
            self.assertEqual(runtime_directory(root), Path.home() / '.local-transcriber' / 'runtime')
            (root / 'settings.local.json').write_text(json.dumps({'runtime_dir': str(root / 'engine')}), encoding='utf-8')
            self.assertEqual(runtime_directory(root), root / 'engine')
            with patch.dict(os.environ, {'TRANSCRIBER_RUNTIME': str(root / 'override')}):
                self.assertEqual(runtime_directory(root), root / 'override')
