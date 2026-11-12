"""schema order and byte-level acquisition integrity, entirely offline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import ifutsvm.data as data


def test_keel_declared_input_order_and_nominal_values(tmp_path):
    # Output is first, one declared attribute is unused, and inputs are reordered.
    text = '''% synthetic loader contract
@relation boundary
@attribute "target class" {minority, majority}
@attribute 'nominal feature' {"blue,rare", red}
@attribute unused integer [-9, 9]
@attribute 'numeric feature' real [-10, 100]
@inputs "numeric feature", 'nominal feature'
@outputs "target class"
@data
majority, "blue,rare", 9, 1e2
minority, red, -9, -3e-2
majority, ?, 0, ?
'''
    path = tmp_path/'schema.dat'
    path.write_bytes(text.encode('utf-8-sig'))
    X, y, schema = data.load_dat(path, return_schema=True)
    assert X.shape == (3, 2) and X.dtype == object
    np.testing.assert_array_equal(y, ['majority', 'minority', 'majority'])
    np.testing.assert_allclose(np.asarray(X[:, 0], float), [100., -.03, np.nan])
    assert X[:2, 1].tolist() == ['blue,rare', 'red']
    assert np.isnan(X[2, 1])
    assert [a['name'] for a in schema] == ['numeric feature', 'nominal feature']
    assert [a['nominal'] for a in schema] == [False, True]


@pytest.mark.parametrize('corruption', ['whitespace', 'numeric_value', 'truncated', 'manifest_digest'])
def test_changed_dataset_rejected_before_parser(tmp_path, monkeypatch, corruption):
    source = Path(__file__).resolve().parents[1]/'data/keel'
    name = 'ecoli4'
    manifest = json.loads((source/'MANIFEST.json').read_bytes())
    entry = dict(manifest['datasets'][name])
    raw = (source/entry['file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry['sha256']
    entry['file'] = 'local-copy.dat'
    path = tmp_path/entry['file']
    path.write_bytes(raw)
    def write_manifest():
        (tmp_path/'MANIFEST.json').write_text(json.dumps({'datasets': {name: entry}}), encoding='utf-8')
    write_manifest()
    X, y = data.load(name, tmp_path)
    assert len(X) == len(y) == entry['rows']

    if corruption == 'whitespace':
        path.write_bytes(raw+b'\n')
    elif corruption == 'numeric_value':
        changed = raw.replace(b'0.49', b'0.50', 1)
        assert changed != raw
        path.write_bytes(changed)
    elif corruption == 'truncated':
        path.write_bytes(raw[:len(raw)//2])
    else:
        entry['sha256'] = '0'*64
        write_manifest()
    parsed = []
    def forbidden_parser(*args, **kwargs):
        parsed.append(True)
        raise AssertionError('Changed bytes reached the parser')
    monkeypatch.setattr(data, 'load_dat', forbidden_parser)
    with pytest.raises(ValueError, match='checksum mismatch'):
        data.load(name, tmp_path)
    assert not parsed


def test_unknown_dataset_rejected_without_files_or_network(tmp_path, monkeypatch):
    def forbidden_network(*args, **kwargs):
        raise AssertionError('Loading attempted network access')
    monkeypatch.setattr(data, 'urlopen', forbidden_network)
    with pytest.raises(ValueError, match='unknown issued dataset'):
        data.load('unissued-dataset', tmp_path/'does-not-exist')
