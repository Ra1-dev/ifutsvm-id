"""Verify committed published bytes and report size differences from Table S.II."""
import json
from pathlib import Path
import numpy as np
from ifutsvm.data import DATASETS,load


def main():
    source=json.loads(Path('data/published_table_sii.json').read_text())['datasets']
    manifest=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets']
    rows=[]
    for name in DATASETS:
        X,y=load(name)
        labels,counts=np.unique(y,return_counts=True)
        assert X.shape==(manifest[name]['rows'],manifest[name]['features'])
        assert dict(zip(labels.tolist(),counts.tolist()))==manifest[name]['class_counts']
        row=dict(dataset=name,rows=len(X),features=X.shape[1],paper_rows=source[name]['rows'],
                 paper_features=source[name]['features'],row_difference=len(X)-source[name]['rows'],
                 feature_difference=X.shape[1]-source[name]['features'],sha256=manifest[name]['sha256'])
        rows.append(row)
        print(json.dumps(row),flush=True)
    Path('results/data_audit.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'Data verified: {len(rows)}/{len(DATASETS)}; table shape discrepancies: {sum(bool(r["row_difference"] or r["feature_difference"]) for r in rows)}')


if __name__=='__main__':
    main()
