"""Machine-specific paths stay in an ignored local configuration file."""
import json
import os
from pathlib import Path


def runtime_directory(root):
    config_file = Path(root) / 'settings.local.json'
    config = json.loads(config_file.read_text(encoding='utf-8')) if config_file.exists() else {}
    value = os.environ.get('TRANSCRIBER_RUNTIME') or config.get('runtime_dir')
    return Path(value).expanduser() if value else Path.home() / '.local-transcriber' / 'runtime'
