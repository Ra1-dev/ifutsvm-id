"""Arithmetic properties of reported percentages; no classifier fitting."""
import json
from pathlib import Path
import math
from experiments.run_all import write_json


def main():
    table=json.loads(Path('data/published_table_sii.json').read_text())['datasets']
    rows=[]
    for dataset,r in table.items():
        candidates=[]
        for count in sorted({math.floor(.3*r['rows']),math.ceil(.3*r['rows'])}):
            correct=round(r['accuracy_percent']*count/100)
            achievable=100*correct/count
            candidates.append(dict(test_rows=count,correct=correct,accuracy_percent=achievable,
                                   rounding_distance_pp=abs(achievable-r['accuracy_percent'])))
        closest=min(candidates,key=lambda v:v['rounding_distance_pp'])
        rows.append(dict(dataset=dataset,reported_percent=r['accuracy_percent'],candidates=candidates,
                         compatible_with_single_split_accuracy=closest['rounding_distance_pp']<=.0050000001))
    write_json('results/table_arithmetic.json',rows)
    for r in rows:
        if not r['compatible_with_single_split_accuracy']:
            print(json.dumps(r))
    print(f"Off single-split accuracy lattice: {sum(not r['compatible_with_single_split_accuracy'] for r in rows)}/{len(rows)}")


if __name__=='__main__': main()
