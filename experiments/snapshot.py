"""Check the complete numerical inputs used by figures, tables and notebooks."""
import hashlib
import json
from pathlib import Path


def require_snapshot(summary_path='results/summary.json', repository='.'):
    root=Path(repository).resolve()
    path=Path(summary_path).resolve()
    assert path.is_relative_to(root), 'Summary must be inside the project'
    raw=path.read_bytes();summary=json.loads(raw)
    for key in ['research','course_curves','main_statistics']:
        assert summary.get(key,{}).get('complete'), key+' incomplete'
    assert summary['complete'] and not summary['missing']
    assert summary['reproduction']['e4_complete']
    manifest=root/'results/INPUTS.json'
    if manifest.exists():
        for name,expected in json.loads(manifest.read_bytes()).items():
            item=(root/name).resolve()
            assert item.is_relative_to(root) and item.is_file(),name
            assert hashlib.sha256(item.read_bytes()).hexdigest()==expected,name+' changed'
    return {'summary_sha256':hashlib.sha256(raw).hexdigest()}
