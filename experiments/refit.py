"""Re-fit seeded random outer runs without using any cache or tuned test scores."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from ifutsvm.data import load
from ifutsvm.evaluation import minority_scores
from experiments.registry import make_model
from experiments.run_all import source_hash,write_json
from experiments.evidence import read_table,iter_tables


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results')
    parser.add_argument('--count',type=int,default=3); parser.add_argument('--seed',type=int,default=24102026)
    args=parser.parse_args(); root=Path(args.output)
    config=json.loads((root/'experiment_config.json').read_text())
    assert config['source_hash']==source_hash()
    entries=[]
    for path in iter_tables(root/'runs'):
        group=read_table(path)
        entries.extend((path,k) for k in sorted(group) if k.endswith('/evaluation'))
    expected=len(config['datasets'])*10*len(config['models'])
    if len(entries)!=expected:
        raise RuntimeError(f'Expected all {expected} outer runs before random selection; got {len(entries)}')
    selected=np.random.default_rng(args.seed).choice(len(entries),size=args.count,replace=False)
    manifest=json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets']; rows=[]
    with threadpool_limits(limits=1):
        for i in selected:
            path,key=entries[i]; record=read_table(path)[key]; m=record['metadata']
            X,y=load(m['dataset']); start=time.perf_counter()
            model=make_model(m['model'],manifest[m['dataset']]['attributes'],m['parameters'],m['seed'])
            model.fit(X[record['train_indices']],y[record['train_indices']])
            pred=model.predict(X[record['test_indices']])
            scores=minority_scores(model,X[record['test_indices']],record['positive_label'])
            np.testing.assert_array_equal(pred,record['prediction'])
            np.testing.assert_allclose(scores,record['scores'],rtol=1e-9,atol=1e-9)
            row=dict(path=path.relative_to(root).as_posix()+'#'+key,predictions_equal=True,
                     max_score_difference=float(np.max(np.abs(scores-record['scores']))),seconds=time.perf_counter()-start)
            rows.append(row); print(json.dumps(row),flush=True)
    write_json(root/'refit_audit.json',dict(seed=args.seed,count=len(rows),source_hash=source_hash(),runs=rows))
    print(f'Independent refits: {len(rows)}/{args.count} PASS')


if __name__=='__main__': main()
