"""independently parse committed KEEL files with SciPy, entirely offline."""
import io
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import arff
from ifutsvm.data import DATASETS,load

ROOT=Path(__file__).resolve().parents[1]/'data'/'keel'


@pytest.mark.parametrize('name',DATASETS)
def test_published_rows_equal_independent_arff_parser(name):
    X,y=load(name,ROOT)
    raw=(ROOT/f'{name}.dat').read_text(encoding='utf-8')
    stripped='\n'.join(line for line in raw.splitlines() if not line.lower().startswith(('@input','@output')))
    records,metadata=arff.loadarff(io.StringIO(stripped))
    columns=records.dtype.names
    assert X.shape==(len(records),len(columns)-1)
    for j,name in enumerate(columns[:-1]):
        expected=records[name]
        if expected.dtype.kind=='S':
            np.testing.assert_array_equal(X[:,j],np.char.decode(expected))
        else:
            np.testing.assert_allclose(np.asarray(X[:,j],float),expected)
    np.testing.assert_array_equal(y,np.char.decode(records[columns[-1]]))


def test_manifest_covers_every_issued_dataset():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    assert set(manifest['datasets'])==set(DATASETS)
    assert all(v['minority_label']=='positive' for v in manifest['datasets'].values())
