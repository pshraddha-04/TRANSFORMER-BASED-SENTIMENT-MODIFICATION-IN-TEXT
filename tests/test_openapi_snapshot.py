import json
from pathlib import Path

from app.api import app

SNAPSHOT_PATH = Path(__file__).with_name('fixtures') / 'openapi_snapshot.json'
STRIP_KEYS = {'description', 'summary', 'examples', 'example', 'externalDocs'}


def _normalize(value):
    if isinstance(value, dict):
        return {
            key: _normalize(inner)
            for key, inner in sorted(value.items())
            if key not in STRIP_KEYS
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def test_openapi_snapshot_matches_fixture():
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
    actual = _normalize(app.openapi())
    assert actual == expected

