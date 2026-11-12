"""Measure training-only fit costs before selecting the shared search budget."""
import json
import argparse
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from ifutsvm.data import DATASETS,load
from experiments.registry import MODELS,make_model


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results/reduced_pilot.json')
    args=parser.parse_args()
    manifest=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets']
    rows=[]
    # Declared before any model scores are inspected. No test scores computed.
    probes=[dict(C=1.,Cu=1.,C3=1.,eps=.3,mu=1.),
            dict(C=100.,Cu=10.,C3=.001,eps=.6,mu=8.)]
    with threadpool_limits(limits=1):
        for dataset in DATASETS:
            X,y=load(dataset)
            train,_=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=0)
            for model in MODELS:
                for probe,p in enumerate(probes):
                    start=time.perf_counter(); error=None
                    try:
                        make_model(model,manifest[dataset]['attributes'],p,0).fit(X[train],y[train])
                    except Exception as ex:
                        error=f'{type(ex).__name__}: {ex}'
                    row=dict(dataset=dataset,model=model,probe=probe,parameters=p,baseline_basis='reduced',seconds=time.perf_counter()-start,error=error)
                    rows.append(row)
                    print(json.dumps(row),flush=True)
                    Path(args.output).write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
