"""One generated main-comparison finding for every downstream artifact."""
from pathlib import Path
import re
from experiments.display import model_name


def headline(summary):
    group=summary['protocols']['holdout']['aggregate']
    names=['paper','svm_balanced','smote_svm','faithful_raw']
    if not all(name in group['models'] for name in names):
        return dict(complete=False,text='The main comparison is incomplete; no final headline is available.')
    values={name:100*group['models'][name]['balanced_accuracy']['mean'] for name in names}
    loses=all(values['paper']<values[name] for name in ('svm_balanced','smote_svm'))
    text=(model_name('paper')+' does not beat the simple imbalance-aware baselines under the shared tuning protocol on these datasets. '
          if loses else 'The full main-comparison means are reported without selecting a favorable subset. ')
    text+=(f"Equal-weight hold-out balanced accuracy over {len(group['datasets'])} datasets: "
           f"{model_name('paper')} {values['paper']:.1f}%, {model_name('svm_balanced')} {values['svm_balanced']:.1f}%, "
           f"{model_name('smote_svm')} {values['smote_svm']:.1f}%, {model_name('faithful_raw')} {values['faithful_raw']:.1f}%.")
    return dict(complete=summary['complete'],text=text,paper_below_both_baselines=loses,
        balanced_accuracy_percent=values,datasets=len(group['datasets']),
        source='results/summary.json#/protocols/holdout/aggregate',command='python -m experiments.summarize',
        scope='Descriptive equal-weight means on the predeclared complete twelve-task benchmark; no universal superiority claim.')


def write_headline(summary):
    finding=summary['headline']
    text=finding['text']
    Path('docs/MAIN_FINDINGS.md').write_text('# Main benchmark facts\n\n'+text+
        '\n\n- Source: `'+finding['source']+'`.\n- Regenerate: `'+finding['command']+'`.\n- '+finding['scope']+'\n',
        encoding='utf-8',newline='\n')
    path=Path('README.md');readme=path.read_text(encoding='utf-8')
    block='<!-- MAIN FINDING -->\n'+text+'\n<!-- END MAIN FINDING -->'
    if '<!-- MAIN FINDING -->' in readme:
        readme=re.sub(r'<!-- MAIN FINDING -->.*?<!-- END MAIN FINDING -->',lambda _:block,readme,flags=re.S)
    else:
        title,rest=readme.split('\n',1);readme=title+'\n\n'+block+'\n'+rest
    path.write_text(readme,encoding='utf-8',newline='\n')
