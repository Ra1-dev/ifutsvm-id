"""Build an executable course notebook from the canonical numerical summary.

Development previews are separate artifacts. Final generation requires the
complete main comparison, research experiment, and training-only diagnostic curves.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import nbformat as nbf
from experiments.notebook_code import (PLOT_METHODS,setup_code,method_sections,
    equality_code,result_code,prediction_code)


def build(summary_path,figures,preview):
    summary_path=Path(summary_path);data=summary_path.read_bytes();summary=json.loads(data)
    if not preview:
        assert summary['complete'] and summary.get('research',{}).get('complete') and summary.get('curves',{}).get('complete'), 'Final notebook requires complete main, research and curve evidence'
        from experiments.snapshot import require_snapshot
        require_snapshot(summary_path)
    cells=[nbf.v4.new_markdown_cell('# CSCI447P031 - IFUTSVM-ID\n\n'+
        ('**DEVELOPMENT PREVIEW: experiment and course artifacts remain incomplete.**\n\n' if preview else '')+
        'Verified faithful port, corrected estimator, and a minority-generative Universum. The pre-specified small-data example performs fresh inner tuning under both outer protocols; full comparisons use persisted results.\n\n'+summary.get('headline',{}).get('text',''))]
    def step(title,description,code):
        if title=='Environment and source of truth':code+='\n'+setup_code()
        if title in PLOT_METHODS:
            code=result_code(code.strip(),PLOT_METHODS[title])
            if title=='Class errors and decision-score ROC':
                code=prediction_code(hashlib.sha256(Path('results/presentation_predictions.json').read_bytes()).hexdigest())+'\n'+code
        cells.append(nbf.v4.new_markdown_cell('## '+title+'\n\n'+description))
        cells.append(nbf.v4.new_code_cell(code.strip()))
    step('Environment and source of truth','Run from the repository root after installing `.[dev,artifacts]`. The summary checksum checks that the loaded numerical inputs match the saved comparisons.',f'''
# Read the single numerical source; never substitute sample-template numbers.
from pathlib import Path
import hashlib, json, platform
import numpy as np
from IPython.display import display, Markdown, Image
from experiments.display import model_name, display_structure
ROOT = Path.cwd()
assert (ROOT / 'pyproject.toml').is_file(), 'Run the notebook from the repository root'
summary_path = ROOT / {summary_path.as_posix()!r}
summary_bytes = summary_path.read_bytes()
assert hashlib.sha256(summary_bytes).hexdigest() == {hashlib.sha256(data).hexdigest()!r}, 'Summary changed; regenerate this notebook'
summary = json.loads(summary_bytes)
FIGURES = ROOT / {str(Path(figures).as_posix())!r}
figure_manifest = json.loads((FIGURES / 'MANIFEST.json').read_text())
assert figure_manifest['summary_sha256'] == hashlib.sha256(summary_bytes).hexdigest(), 'Figures use another summary; regenerate them'
assert figure_manifest['preview'] == {preview!r}, 'Figure and notebook preview status differ'
for figure in figure_manifest['figures']:
    for artifact in figure['files']:
        relative = Path('src') / Path(artifact['path']).name if artifact['path'].endswith('.drawio') else Path(artifact['path']).name
        content = (FIGURES / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact['sha256'], artifact['path']
print('Python:', platform.python_version())
print('Main results complete:', summary['complete'])
print('Recorded model dependencies:', summary['config']['versions'])
''')
    step('Official data and checksums','The full KEEL files are local. Loading and notebook execution perform no network requests. The minority label is defined from training class counts.',r'''
# Verify every raw file against the acquisition manifest.
manifest = summary['data_manifest']['datasets']
dataset_rows = []
for name in summary['config']['datasets']:
    record = manifest[name]
    raw = ROOT / 'data' / 'keel' / record['file']
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == record['sha256']
    dataset_rows.append([name, record['rows'], record['features'],
                         record['class_counts'], f"{record['imbalance_ratio']:.3f}"])
def table(headers, rows):
    lines = ['| ' + ' | '.join(map(str, headers)) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]
    display(Markdown('\n'.join(lines)))
table(['Dataset', 'Rows', 'Raw features', 'Class counts', 'IR'], dataset_rows)
# Visual EDA: majority/minority proportions in every official dataset.
import matplotlib.pyplot as plt
names = summary['config']['datasets']
minority_fraction = [min(manifest[n]['class_counts'].values()) / manifest[n]['rows'] for n in names]
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(names, minority_fraction, label='Minority', color='#009E73')
ax.barh(names, 1-np.array(minority_fraction), left=minority_fraction, label='Majority', color='#CCCCCC')
ax.set(xlim=(0, 1), xlabel='Fraction of rows', title='Official-data class balance')
ax.legend(); fig.tight_layout(); plt.show()
''')
    step('Evaluation protocol','Both outer protocols use the same candidate proposals for every model. Encoding, scaling, SMOTE and Universum fitting occur inside training partitions. Main selection uses balanced accuracy; the separate reproduction uses accuracy.',r'''
# Display pre-specified choices instead of copying numbers into the notebook.
config = summary['config']
print('Candidate budget:', config['budget'], '| Inner folds:', config['inner_folds'])
print('Models:', ', '.join(model_name(name) for name in config['models']))
for protocol, details in summary['protocols'].items():
    complete = sum(all(model in group for model in config['models'])
                   for group in details['datasets'].values())
    print(protocol, ': complete common-dataset groups =', complete)
for filename in ['fig01_methodology_block_diagram.png', 'fig03_evaluation_protocol.png']:
    if (FIGURES / filename).exists(): display(Image(filename=str(FIGURES / filename)))
''')
    cells.append(nbf.v4.new_markdown_cell(Path('docs/ALGORITHM.md').read_text(encoding='utf-8')))
    for title,description,code in method_sections():step(title,description,code)
    step('Didactic implementation equality','The fixed real-data check below compares both Universum choices with the package. It requires identical predictions and maximum absolute plane error at most 1e-6; it makes no performance claim.',equality_code())
    cells.append(nbf.v4.new_markdown_cell(r'''## Fresh end-to-end experiment

The following cell runs the pre-specified smallest dataset from scratch. Each
protocol repeats inner selection and an outer refit on its original fixed splits.
This teaching execution does not replace any full-benchmark result.'''))
    cells.append(nbf.v4.new_code_cell(r'''
# Execute fresh tuning, refitting and evaluation; the unique path forbids fitted-cache reuse.
display(Image(filename=str(FIGURES / 'fig02_model_architecture.png')))
import time, uuid, contextlib, io
from importlib.metadata import version
from experiments.run_all import run_group, source_hash, write_json
from experiments.storage import build_index
from experiments.audit import audit
from experiments.evidence import read_table
plan = summary['artifact_plan']['notebook']
dataset = plan['dataset']
assert dataset == min(config['datasets'], key=lambda n: (manifest[n]['rows'], n))
current_source_hash = source_hash()
live_root = ROOT / 'tmp' / ('notebook-live-' + uuid.uuid4().hex)
live_root.mkdir(parents=True, exist_ok=False)
live_config = {**config, 'source_hash': current_source_hash, 'output': str(live_root), 'datasets': [dataset],
               'models': plan['models'], 'workers': 1,
               'versions': {name: version(name) for name in config['versions']}}
assert live_config['budget'] == plan['budget'] and live_config['inner_folds'] == plan['inner_folds']
write_json(live_root / 'experiment_config.json', live_config)
start = time.perf_counter()
runner_log = io.StringIO()
with contextlib.redirect_stdout(runner_log):
    for protocol in plan['protocols']:
        result = run_group((dataset, protocol, 'paper', live_config))
        assert not result['errors'] and len(result['outcomes']) == 5
    write_json(live_root / 'runs/index.json', build_index(live_root, live_config))
    checked = audit(live_root)
(live_root / 'runner.log').write_text(runner_log.getvalue(), encoding='utf-8')
print('Live model:', model_name('paper'))
assert checked['complete'] and checked['outer_records'] == 10 and checked['inner_records'] == 1200
live = dict(dataset=dataset, protocols=plan['protocols'], fresh_output=str(live_root.relative_to(ROOT)),
            fitted_cache_reused=False, outer_evaluations=checked['outer_records'],
            inner_evaluations=checked['inner_records'], failed_inner_fits=checked['failed_inner_fits'],
            seconds=time.perf_counter()-start, fingerprint=source_hash())
write_json(live_root / 'live_execution.json', live)
print('LIVE_EXPERIMENT_JSON=' + json.dumps(live, separators=(',', ':')))
rows=[]
for protocol in plan['protocols']:
    records=read_table(live_root / 'runs' / f'{dataset}__paper__{protocol}.parquet')
    outer=[r for key,r in records.items() if key.endswith('/evaluation')]
    rows.append([protocol, len(outer), np.mean([r['accuracy'] for r in outer]),
                 np.mean([r['balanced_accuracy'] for r in outer])])
table(['Live protocol', 'Outer runs', 'Accuracy (fraction)', 'Balanced accuracy (fraction)'], rows)
'''.strip()))
    step('Main metrics and comparison','The table shows the deterministic illustrative dataset. Full tables retain every dataset and every run. Plots aggregate equal-weight dataset means and label their uncertainty definition.',r'''
# Read all displayed means and standard deviations from the summary.
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'gmean', 'balanced_accuracy']
for protocol in ['holdout', 'nested_cv']:
    group = summary['protocols'][protocol]['datasets'][dataset]
    rows = []
    for name in config['models']:
        if name not in group: continue
        rows.append([model_name(name)] + [f"{100*group[name]['metrics'][metric]['mean']:.2f} +/- {100*group[name]['metrics'][metric]['std']:.2f}"
                    for metric in metrics])
    display(Markdown('### ' + protocol + ' / ' + dataset + ' (percent)'))
    table(['Model'] + metrics, rows)
if (FIGURES / 'fig04_main_performance.png').exists(): display(Image(filename=str(FIGURES / 'fig04_main_performance.png')))
''')
    step('Every generated comparison table','Load the complete generated CSV tables and verify their source/hash manifest. The files retain all models, datasets and folds; the preceding display is the live-example dataset only.',f'''
# Load every table, including literature, technology, all folds and research when complete.
import csv
TABLES = ROOT / {'tmp/tables-preview' if preview else 'results/tables'!r}
table_manifest = json.loads((TABLES / 'MANIFEST.json').read_bytes())
assert table_manifest['summary_sha256'] == hashlib.sha256(summary_bytes).hexdigest()
assert table_manifest['preview'] == {preview!r}
loaded_tables = {{}}
for entry in table_manifest['tables']:
    for artifact in entry['files']:
        payload = (TABLES / Path(artifact['path']).name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == artifact['sha256']
    with (TABLES / (entry['name'] + '.csv')).open(encoding='utf-8', newline='') as stream:
        loaded_tables[entry['name']] = list(csv.DictReader(stream))
table(['Table', 'Loaded rows', 'CSV file'], [[name, len(rows), str(TABLES / (name + '.csv'))]
      for name, rows in loaded_tables.items()])
print('Pending table sections:', table_manifest['pending'])
''')
    step('Class errors and decision-score ROC','The confusion matrices use majority/minority row order and row percentages. ROC-AUC uses decision scores oriented toward the training-defined minority, rather than predicted labels.',r'''
# Saved comparisons use predeclared low, middle and high imbalance ratios.
for filename in ['fig06_confusion_matrices.png', 'fig05_roc_curves.png']:
    if (FIGURES / filename).exists(): display(Image(filename=str(FIGURES / filename)))
print('Illustration selection rule:', summary['presentation_illustrations']['representative_rule'])
''')
    step('Learning and validation curves','SVMs do not have neural-network epochs. These curves vary training sample size or a fixed diagnostic parameter inside outer training only; they do not choose another comparison model.',r'''
# Curves use no original outer-test rows.
curves = summary.get('course_curves', {})
if curves.get('complete'):
    print(curves['plan']['learning']['rule'])
    print(curves['plan']['validation']['rule'])
    for filename in ['fig07_learning_curves.png', 'fig08_validation_curves.png']:
        display(Image(filename=str(FIGURES / filename)))
else:
    print('PREVIEW: curve evidence is not complete.')
''')
    step('Reproduction against the paper','Accuracy-selected reproduction is distinct from the main balanced-accuracy comparison. A numerical match is not evidence that the authors used the same undocumented configuration.',r'''
# Retain unmatched numbers and their measured diagnostic categories.
reproduction = summary.get('reproduction', {})
if reproduction:
    rows = []
    for name, result in reproduction['datasets'].items():
        r = result['comparison']
        rows.append([name, f"{r['published_accuracy_percent']:.4f}",
            f"{r['tuned_accuracy_mean_percent']:.4f}", f"{r['gap_pp']:+.4f}", r['evidence_category']])
    table(['Dataset','Published ACC %','Reproduction ACC %','Gap pp','Evidence'], rows)
    display(Image(filename=str(FIGURES / 'fig09_reproduction.png')))
else:
    print('PREVIEW: final reproduction table is pending.')
''')
    step('Generative Universum, ablation and falsification','The strategy conditions a fitted minority Gaussian on equal class-conditional density. The population identity and threshold surrogate do not guarantee improved IFUTSVM predictions. The original refuted transfer prediction remains in the record.',r'''
# Report every pre-specified variant and the original hypothesis outcome.
research = summary.get('research', {})
if research.get('complete'):
    theory = research['theory']
    print('Original transfer hypothesis supported:', theory['supported'])
    print('High-IR balanced-accuracy change (pp):', 100*theory['balanced_accuracy_delta'])
    print('Absolute boundary-distance change:', theory['boundary_distance_delta'])
    print('Completed ablation cells:', research['ablation']['completed_outer'])
    print('Completed noise cells:', research['noise']['completed_outer'])
    for filename in ['fig11_ablation.png', 'fig12_label_noise.png']:
        display(Image(filename=str(FIGURES / filename)))
else:
    print('PREVIEW: full ablation and label-noise evidence is pending; no superiority claim.')
''')
    step('Across-dataset statistics','Dataset means, rather than overlapping folds, are the statistical units. Holm correction covers the declared paired contrasts; related Ecoli/Yeast tasks also receive a source-family sensitivity analysis. The four identity contrasts stay in the original Holm family to preserve its declared scope and saved values. Their p=1 is bookkeeping, not evidence for an independent method; retaining them is conservative. The smallest raw p exceeds 0.05, so removing them would not yield significance. Non-significance is not equivalence.',r'''
# The complete main benchmark supports its omnibus analysis independently.
main_stats = summary.get('main_statistics', {})
if main_stats.get('complete'):
    print(main_stats['omnibus_families'])
    display(display_structure(main_stats['family_sensitivity']))
    rows = []
    for protocol, metrics in main_stats['friedman_nemenyi'].items():
        for metric, result in metrics.items():
            rows.append([protocol, metric, result['datasets'], result['friedman_statistic'], result['friedman_p'], result['critical_difference']])
    table(['Protocol','Metric','Datasets','Friedman statistic','p','Critical difference'], rows)
    display(Image(filename=str(FIGURES / 'fig10_critical_difference.png')))
# Paired generative comparisons still require every pre-specified research cell.
if research.get('complete'):
    print(research['statistics']['paper_identity'])
    display(display_structure(research['statistics']['paired']))
    display(display_structure(research['statistics']['family_sensitivity']))
else:
    print('PREVIEW: paired generative comparisons require complete pre-specified research cells.')
''')
    step('Reproducibility and scope','The raw data and predictions, manifests, deterministic seeds, scripts and source revisions permit independent checks.',r'''
# State status from the evidence rather than marking the assignment complete.
print('Main complete:', summary['complete'])
print('Research complete:', bool(research.get('complete')))
print('Curves complete:', bool(curves.get('complete')))
print('Model source fingerprint:', config['source_hash'])
print('Verification: python -m experiments.audit; python -m experiments.refit')
print('Data acquisition is explicit; notebook execution is offline.')
# Numbered factual conclusions are generated from the numerical source.
facts = [summary['headline']['text'],
         'Ordinary-accuracy reproduction matches: ' + str(summary['reproduction']['within_tolerance'])
             + '; matched or explained: ' + str(summary['reproduction']['matched_or_explained']) + '.',
         'Live experiment: ' + str(live['outer_evaluations']) + ' outer evaluations and '
             + str(live['inner_evaluations']) + ' inner evaluations under both pre-specified protocols.']
if research.get('complete'):
    facts.append('Original synthetic transfer prediction supported: ' + str(research['theory']['supported']) + '.')
display(Markdown('\n'.join(str(i) + '. ' + fact for i,fact in enumerate(facts, 1))))
''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata=dict(
        kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        language_info=dict(name='python',version=platform_version())))
    for index,cell in enumerate(notebook.cells):cell.id=f'step-{index:02d}'
    nbf.validate(notebook)
    assert all(cells[i-1].cell_type=='markdown' for i,c in enumerate(cells) if c.cell_type=='code')
    return notebook


def platform_version():
    import platform
    return platform.python_version()


def execute(notebook,root):
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    runtime=root/'tmp/notebook-runtime';spec=runtime/'kernels/ifutsvm-local';spec.mkdir(parents=True,exist_ok=True)
    (spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Repository Python',language='python')),encoding='utf-8',newline='\n')
    os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(runtime))
    os.environ.setdefault('IPYTHONDIR',str(runtime/'ipython'))
    manager=KernelManager(kernel_name='ifutsvm-local',ip='127.0.0.1',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec.parent)]))
    try:
        NotebookClient(notebook,km=manager,timeout=600,resources={'metadata':{'path':str(root)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    assert not any(output.output_type=='error' for cell in notebook.cells if cell.cell_type=='code' for output in cell.outputs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--summary',default='results/summary.json')
    parser.add_argument('--figures');parser.add_argument('--preview',action='store_true')
    parser.add_argument('--output');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--verification-output');args=parser.parse_args()
    output=Path(args.output or ('tmp/notebook-preview.ipynb' if args.preview else 'CSCI447P031.ipynb'))
    if args.preview and output.resolve()==Path('CSCI447P031.ipynb').resolve():raise ValueError('Preview cannot overwrite the final notebook')
    figures=args.figures or ('tmp/presentation-preview' if args.preview else 'figures')
    notebook=build(args.summary,figures,args.preview)
    started=time.perf_counter()
    if args.execute:execute(notebook,Path.cwd())
    elapsed=time.perf_counter()-started
    notebook.metadata={k:v for k,v in notebook.metadata.items() if k in ('kernelspec','language_info')}
    for cell in notebook.cells:cell.metadata={}
    assert not args.execute or elapsed<=900, 'Notebook exceeds the fifteen-minute execution contract'
    output.parent.mkdir(parents=True,exist_ok=True)
    # Windows text translation would make the saved notebook differ from its
    # committed LF bytes and invalidate the execution certificate in archives.
    with output.open('w',encoding='utf-8',newline='\n') as stream:
        nbf.write(notebook,stream)
    if args.verification_output:
        code=[c for c in notebook.cells if c.cell_type=='code']
        record=dict(notebook=output.as_posix(),notebook_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            executed=args.execute,cells=len(notebook.cells),code_cells=len(code),
            execution_seconds=elapsed if args.execute else None,
            executed_code_cells=sum(c.execution_count is not None for c in code),
            error_outputs=sum(o.output_type=='error' for c in code for o in c.outputs),
            markdown_before_every_code_cell=all(notebook.cells[i-1].cell_type=='markdown' for i,c in enumerate(notebook.cells) if c.cell_type=='code'))
        for cell in code:
            for output_cell in cell.outputs:
                if output_cell.output_type=='stream':
                    for line in output_cell.text.splitlines():
                        if line.startswith('LIVE_EXPERIMENT_JSON='):record['live_experiment']=json.loads(line.split('=',1)[1])
                        if line.startswith('DIDACTIC_EQUIVALENCE_JSON='):record['didactic_equivalence']=json.loads(line.split('=',1)[1])
        Path(args.verification_output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'Notebook: {len(notebook.cells)} cells; executed={args.execute}; preview={args.preview}; {output}')


if __name__=='__main__':main()
