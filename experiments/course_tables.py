"""Course CSV/IEEE LaTeX tables with per-cell summary pointers."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from experiments.presentation_data import resolve
from experiments.display import MODEL_NAMES,model_name,published_name
import textwrap


def escape(value):
    replacements={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
        '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
        '\\':r'\textbackslash{}','±':r'\(\pm\)','×':r'\(\times\)'}
    raw=str(value)
    if (raw in MODEL_NAMES.values() or raw==published_name('IFUTSVM-ID')) and len(raw)>25:
        return r'\shortstack[l]{'+r'\\'.join(''.join(replacements.get(c,c) for c in line)
            for line in textwrap.wrap(raw,25,break_long_words=False,break_on_hyphens=False))+'}'
    return ''.join(replacements.get(c,c) for c in raw)


class Tables:
    def __init__(self,summary,output):
        self.summary=summary;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.tables=[];self.pointer_checks=0

    def cell(self,pointer,scale=1,decimals=None):
        value=resolve(self.summary,pointer)
        self.pointer_checks+=1
        display=value if decimals is None else f'{scale*value:.{decimals}f}'
        return dict(value=value,source=pointer,scale=scale,decimals=decimals,display=str(display))

    def metric(self,pointer,percent=True):
        value=resolve(self.summary,pointer);scale=100 if percent else 1
        a=self.cell(pointer+'/mean',scale,2 if percent else 4)
        b=self.cell(pointer+'/std',scale,2 if percent else 4)
        return dict(value=dict(mean=a['value'],std=b['value']),source=pointer,
            scale=scale,decimals=2 if percent else 4,display=a['display']+' ± '+b['display'],bold=bool(value.get('best',False)))

    def date(self,pointer):
        cell=self.cell(pointer)
        cell.update(display=datetime.fromisoformat(cell['value']).date().isoformat(),display_format='iso_date')
        return cell

    def write(self,name,caption,columns,rows,existing_csv=False,latex_panels=None):
        path=self.output/(name+'.csv')
        if not existing_csv:
            with path.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream,lineterminator='\n');writer.writerow(columns)
                writer.writerows([cell['display'] if isinstance(cell,dict) else cell for cell in row] for row in rows)
        # A long result table is split into per-dataset IEEE table* fragments;
        # no row is dropped to make a page fit.
        chunks=[rows]
        if len(rows)>20:
            chunks=[]
            for row in rows:
                if not chunks or chunks[-1][0][0]!=row[0]:chunks.append([])
                chunks[-1].append(row)
        lines=['% Generated from results/summary.json; requires graphicx.']
        panels=latex_panels or [dict(label='',columns=list(range(len(columns))))]
        assert set(i for panel in panels for i in panel['columns'])==set(range(len(columns)))
        for chunk,panel in ((chunk,panel) for chunk in chunks for panel in panels):
            indices=panel['columns'];panel_columns=[columns[i] for i in indices]
            detail=(' / '+str(chunk[0][0]) if len(chunks)>1 else '')+(' / '+panel['label'] if panel['label'] else '')
            lines += [r'\begin{table*}[!t]',r'\centering',r'\caption{'+escape(caption+detail)+'}',
                r'\footnotesize',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{'+'l'*len(indices)+'}',r'\hline',
                ' & '.join(escape(c) for c in panel_columns)+r' \\',r'\hline']
            for row in chunk:
                values=[]
                for cell in (row[i] for i in indices):
                    value=escape(cell['display'] if isinstance(cell,dict) else cell)
                    if isinstance(cell,dict) and cell.get('bold'):value=r'\textbf{'+value+'}'
                    values.append(value)
                lines.append(' & '.join(values)+r' \\')
            lines += [r'\hline',r'\end{tabular}%',r'}',r'\end{table*}','']
        tex=self.output/(name+'.tex');tex.write_text('\n'.join(lines),encoding='utf-8',newline='\n')
        self.tables.append(dict(name=name,caption=caption,columns=columns,rows=rows,
            files=[dict(path=p.as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (path,tex)]))
        if latex_panels:self.tables[-1]['latex_panels']=latex_panels


def build(summary,output,preview=False):
    assert summary['complete'],'Main tables require all pre-specified groups'
    if not preview:assert summary.get('research',{}).get('complete'),'Final tables require complete research'
    t=Tables(summary,output);data=summary['data_manifest']['datasets'];names=summary['config']['datasets']
    env='/environment' if 'environment' in summary else '/submission_context/environment';environment=resolve(summary,env)
    rows=[[name,t.cell(env+'/'+name)] for name in ('python','platform','architecture','processor','logical_cpus','physical_memory_bytes')]
    rows += [[name,t.cell(env+'/versions/'+name)] for name in environment['versions']]
    t.write('technology','Measured software versions and experiment hardware',['Item','Recorded value'],rows)
    rows=[]
    for name in names:
        prefix='/data_manifest/datasets/'+name;r=data[name]
        label=min(r['class_counts'],key=lambda k:(r['class_counts'][k],k))
        rows.append([name,t.cell(prefix+'/rows'),t.cell(prefix+'/features'),label,
            t.cell(prefix+'/class_counts/'+label),t.cell(prefix+'/imbalance_ratio',decimals=4)])
    t.write('datasets','Official full KEEL files; raw feature counts, minority label and imbalance ratio',
        ['Dataset','Rows','Features','Minority label','Minority rows','IR'],rows)
    methods=summary['literature']['methods']
    rows=[[name]+[t.cell('/literature/datasets/'+name+'/accuracy_percent/'+m,decimals=2) for m in methods] for name in names]
    t.write('related_work','Table S.II accuracies as published (%); author splits and metric interpretation are unverified',
        ['Dataset',*[published_name(m) for m in methods]],rows)
    metrics=['accuracy','precision','recall','f1','roc_auc','gmean','balanced_accuracy','training_seconds']
    headers=['ACC %','Precision %','Recall %','F1 %','ROC-AUC %','G-mean %','Balanced ACC %','Fit seconds']
    for protocol,suffix in [('holdout','holdout'),('nested_cv','cv')]:
        rows=[]
        for name in names:
            for model in summary['config']['models']:
                prefix=f'/protocols/{protocol}/datasets/{name}/{model}/metrics'
                rows.append([name,model_name(model)]+[t.metric(prefix+'/'+metric,metric!='training_seconds') for metric in metrics])
        # Preserve the independently audited canonical CSV schema.
        existing=Path('results/tables')/f'main_{suffix}.csv';target=Path(output)/existing.name
        if target.resolve()!=existing.resolve():target.write_bytes(existing.read_bytes())
        t.write(f'main_{suffix}','Five outer runs: mean ± sample SD; bold best per dataset',
            ['Dataset','Model',*headers],rows,existing_csv=True,
            latex_panels=[dict(label='Class metrics',columns=[0,1,2,3,4,5]),
                          dict(label='AUC, balance and time',columns=[0,1,6,7,8,9])])
    baseline=max((m for m in summary['config']['models'] if m not in ('paper','faithful_raw','faithful_scaled')),
        key=lambda m:summary['protocols']['holdout']['aggregate']['models'][m]['balanced_accuracy']['mean'])
    rows=[]
    for name in names:
        for model in ('paper',baseline):
            prefix=f'/protocols/nested_cv/datasets/{name}/{model}/metrics'
            for fold in range(5):
                rows.append([name,model_name(model),str(fold)]+[t.cell(prefix+'/'+metric+'/values/'+str(fold),
                    1 if metric=='training_seconds' else 100,4 if metric=='training_seconds' else 2) for metric in metrics])
    t.write('cv_folds','Nested-CV folds; baseline chosen by hold-out mean balanced accuracy (descriptive)',
        ['Dataset','Model','Outer fold',*headers],rows,
        latex_panels=[dict(label='Class metrics',columns=[0,1,2,3,4,5,6]),
                      dict(label='AUC, balance and time',columns=[0,1,2,7,8,9,10])])
    pending=[]
    if summary.get('research',{}).get('complete'):
        rows=[]
        for name in names:
            for variant in summary['research']['plan']['variants']:
                prefix=f'/research/ablation/datasets/{name}/0.0/{variant}/metrics'
                rows.append([name,model_name(variant)]+[t.metric(prefix+'/'+metric) for metric in ('gmean','balanced_accuracy','f1','roc_auc')]+
                    [t.metric(prefix+'/'+metric+'/paired_vs_random') for metric in ('gmean','balanced_accuracy','f1','roc_auc')])
        t.write('ablation','All pre-specified variants: percent means/SDs and paired percentage-point differences versus random averaging',
            ['Dataset','Variant','G-mean %','Balanced ACC %','F1 %','ROC-AUC %','Delta G-mean pp','Delta balanced ACC pp','Delta F1 pp','Delta ROC-AUC pp'],rows,
            latex_panels=[dict(label='Absolute performance',columns=[0,1,2,3,4,5]),
                          dict(label='Paired changes',columns=[0,1,6,7,8,9])])
    else:pending.append('ablation')
    rows=[]
    correspondence={'UTSVM':'utsvm','IFUTSVM-ID':'paper'}
    for name in names:
        for method in methods:
            measured=correspondence.get(method)
            prefix=f'/protocols/holdout/datasets/{name}/{measured}/metrics'
            rows.append([name,published_name(method),model_name(measured) if measured else 'not implemented',
                t.cell('/literature/datasets/'+name+'/accuracy_percent/'+method,decimals=2),
                t.metric(prefix+'/accuracy') if measured else 'not implemented',
                t.metric(prefix+'/balanced_accuracy') if measured else 'not implemented',
                'Different raw feature count' if not summary['literature']['datasets'][name]['dimension_match'] else 'Same raw dimensions'])
    t.write('sota','Published versus measured models: author splits/metrics unknown; ours use protocol A and balanced-accuracy selection. No direct superiority claim',
        ['Dataset','Published method','Measured model','Published ACC %','Our ACC %','Our balanced ACC %','Dimensions'],rows)
    return dict(preview=preview,tables=t.tables,pending=pending,pointer_checks=t.pointer_checks,
        best_baseline=baseline,headline=summary['headline'],command='python -m experiments.course_tables')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--output')
    args=parser.parse_args();raw=Path('results/summary.json').read_bytes();summary=json.loads(raw)
    output=Path(args.output or ('tmp/tables-preview' if args.preview else 'results/tables'))
    if args.preview and output.resolve()==Path('results/tables').resolve():raise ValueError('Preview cannot overwrite final tables')
    if not args.preview:
        from experiments.snapshot import require_snapshot
        require_snapshot()
    result=build(summary,output,args.preview);result['summary_sha256']=hashlib.sha256(raw).hexdigest()
    (output/'MANIFEST.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(f"Tables: {len(result['tables'])}; source pointers: {result['pointer_checks']}; pending={result['pending']}; preview={args.preview}")


if __name__=='__main__':main()
