"""Fix an equal budget using fit timings only, before benchmark outcomes."""
import json
from pathlib import Path
import statistics
import os
from experiments.registry import MODELS


def main():
    rows=json.loads(Path('results/reduced_pilot.json').read_text())
    datasets=sorted({r['dataset'] for r in rows})
    maxima=sum(max(r['seconds'] for r in rows if r['dataset']==d and r['model']==m)
               for d in datasets for m in MODELS)
    budget=40
    workers=4
    # Conservative planning only: inner sets have 2/3 of outer rows; use 0.5
    # rather than cubic 0.296, and a 2x parallel/large-outer overhead reserve.
    estimate=10*(3*budget*.5+1)*maxima/workers*2
    record=dict(budget=budget,workers=workers,logical_cpus=os.cpu_count(),inner_folds=3,
        baseline_basis='reduced',protocol_revision=3,gate_seconds=6*3600,pilot_fits=len(rows),
        pilot_failed_fits=sum(r['error'] is not None for r in rows),
        sum_dataset_model_max_fit_seconds=maxima,estimated_full_seconds=estimate,
        estimated_default_40_seconds=10*(3*40*.5+1)*maxima/workers*2,
        median_fit_seconds={m:statistics.median(r['seconds'] for r in rows if r['model']==m) for m in MODELS},
        rationale='The implementation specification requires at least 24 shared candidates. Reduced-basis timings support the original 40-candidate budget with three inner folds. '
                  'The earlier four-candidate results are rejected pilot evidence only. No prediction is computed by this timing pilot. '
                  'Estimate uses 0.5 inner-fit scaling and 2x overhead reserve; actual gate is execution.json. '
                  'The conservative estimate fits the amended six-hour gate; research ablations keep the same candidate budget.',
        selection_metric='balanced_accuracy',reproduction_metric='accuracy',
        side2_fixed_apriori='symmetric',seed_entropy=20261110)
    Path('results/compute_budget.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
