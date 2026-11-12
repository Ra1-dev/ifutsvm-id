"""Emit literal teaching and matplotlib source, not runtime source introspection."""
import ast
import inspect
import json
import textwrap
from experiments import teaching
from experiments.presentation_figures import Pack,COLORS
from experiments.display import MODEL_NAMES

PLOT_METHODS={
    'Main metrics and comparison':['performance'],
    'Class errors and decision-score ROC':['confusion','roc'],
    'Learning and validation curves':['curves'],
    'Reproduction against the paper':['reproduction'],
    'Generative Universum, ablation and falsification':['ablation','noise'],
    'Across-dataset statistics':['critical_difference'],
}


def method_sections():
    groups=[
        ('RBF kernel','The same Gaussian kernel is used at every stage.',[teaching.teaching_rbf]),
        ('Intuitionistic-fuzzy scores','Membership is proximity to a class centre; non-membership is local opposite-class ambiguity.',[teaching.teaching_if_scores]),
        ('Training-only Universum constructions','Random averaging and minority-Gaussian conditioning use only the supplied training classes.',[teaching.teaching_random_average,teaching.teaching_conditional_gaussian]),
        ('Two direct primal QPs','This didactic fit shows the default reduced basis and symmetric second-plane constraint explicitly.',[teaching.teaching_plane,teaching.teaching_fit]),
        ('Equation 32 prediction','Choose the nearer normalized plane; a distance tie selects the majority.',[teaching.teaching_predict])]
    for title,description,functions in groups:
        imports='import cvxpy as cp\nfrom sklearn.covariance import LedoitWolf\n' if title=='RBF kernel' else ''
        yield title,description,imports+'\n\n'.join(inspect.getsource(f) for f in functions)


def setup_code():
    return '''
# All plotting operations below remain visible in this notebook.
import copy, textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
plot_summary = copy.deepcopy(summary)
pending = []
COLORS = '''+repr(COLORS)+'''
LABELS = '''+repr(MODEL_NAMES)+'''
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.titlesize':17,
    'axes.labelsize':15,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,
    'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':100})

def finish_plot(fig, title, caption):
    # Titles, uncertainty definitions and sources are drawn as text artists.
    fig.suptitle(title, x=.045, y=.975, ha='left', fontsize=24,
                 fontweight='bold', color='#172F54')
    fig.text(.045, .915, '\\n'.join(textwrap.wrap(caption, width=105)),
             va='top', fontsize=14, color='#3B4A59')
    fig.text(.045, .023, 'CSCI447P031 | Saved source: results/summary.json',
             fontsize=14, color='#3B4A59')
    plt.show()
'''


def equality_code():
    return '''
# A fixed real-data split checks implementation equality, not performance.
from ifutsvm import IFUTSVMID
from ifutsvm.data import load
from ifutsvm.universum import ConditionalGaussianUniversum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
check_dataset = summary['artifact_plan']['notebook']['dataset']
check_X, check_y = load(check_dataset)
train, test = train_test_split(np.arange(len(check_y)), test_size=.3,
    stratify=check_y, random_state=71)
scaler = StandardScaler().fit(check_X[train])
X_train = scaler.transform(check_X[train])
X_test = scaler.transform(check_X[test])
equivalence = []
for generative in [False, True]:
    didactic = teaching_fit(X_train, check_y[train], generative=generative)
    strategy = ConditionalGaussianUniversum() if generative else 'random_average'
    production = IFUTSVMID(C=.7, C3=.4, Cu=.3, eps=.3, mu=1.2,
        random_state=47, universum=strategy).fit(X_train, check_y[train])
    plane_error = max(float(np.max(np.abs(didactic[key] - production.trace_[key])))
                      for key in ['z1', 'z2'])
    assert plane_error <= 1e-6, plane_error
    predicted = teaching_predict(didactic, X_test)
    assert np.array_equal(predicted, production.predict(X_test))
    equivalence.append(dict(generative=generative, maximum_plane_error=plane_error,
        predictions_identical=True, test_rows=len(test)))
didactic_check = dict(dataset=check_dataset, training_rows=len(train),
    plane_tolerance=1e-6, comparisons=equivalence)
print('DIDACTIC_EQUIVALENCE_JSON=' + json.dumps(didactic_check))
'''


class VisiblePlot(ast.NodeTransformer):
    def visit_Attribute(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='self':
            if node.attr=='s':return ast.Name(id='plot_summary',ctx=ast.Load())
            if node.attr=='pending':return ast.Name(id='pending',ctx=ast.Load())
        return self.generic_visit(node)

    def visit_Call(self,node):
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='self' and node.func.attr=='save':
            return ast.Call(func=ast.Name(id='finish_plot',ctx=ast.Load()),
                args=[self.visit(node.args[i]) for i in [0,2,3]],keywords=[])
        return self.generic_visit(node)


def plot_code(methods):
    blocks=[]
    for method in methods:
        tree=ast.parse(textwrap.dedent(inspect.getsource(getattr(Pack,method))))
        function=tree.body[0];function.name='plot_'+method;function.args.args=[]
        tree=VisiblePlot().visit(tree);ast.fix_missing_locations(tree)
        blocks.append('# Draw the saved '+method.replace('_',' ')+' data with visible matplotlib calls.\n'+
            ast.unparse(tree)+'\nplot_'+method+'()')
    return '\n\n'.join(blocks)


class RemoveResultImages(ast.NodeTransformer):
    def visit_Expr(self,node):
        if any(isinstance(part,ast.Call) and isinstance(part.func,ast.Name) and part.func.id=='Image' for part in ast.walk(node)):
            return None
        return self.generic_visit(node)

    def visit_If(self,node):
        node=self.generic_visit(node)
        if not node.body and not node.orelse:return None
        if not node.body:node.body=[ast.Pass()]
        return node

    def visit_For(self,node):
        node=self.generic_visit(node)
        return node if node.body else None


def result_code(source,methods):
    tree=RemoveResultImages().visit(ast.parse(source));ast.fix_missing_locations(tree)
    return '# Read the saved values and retain their uncertainty and inference scope.\n'+ast.unparse(tree)+'\n\n'+plot_code(methods)


def prediction_code(digest):
    return '''
# Recompute every ROC and pooled confusion matrix from persisted labels/scores.
# All prescribed seeds are retained; no new fits or performance-based selection.
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
prediction_bytes = (ROOT / 'results/presentation_predictions.json').read_bytes()
assert hashlib.sha256(prediction_bytes).hexdigest() == '''+repr(digest)+'''
prediction_data = json.loads(prediction_bytes)
assert prediction_data['summary_sha256'] == hashlib.sha256(summary_bytes).hexdigest()
for dataset_name, models in prediction_data['datasets'].items():
    for name, records in models.items():
        target = plot_summary['presentation_illustrations']['datasets'][dataset_name]['models'][name]
        total = np.zeros((2, 2), dtype=int)
        for record in records:
            y_positive = np.array(record['y']) == record['positive_label']
            pred_positive = np.array(record['prediction']) == record['positive_label']
            total += confusion_matrix(y_positive, pred_positive, labels=[False, True])
            if record['seed'] == summary['presentation_illustrations']['roc_seed']:
                fpr, tpr, thresholds = roc_curve(y_positive, record['scores'])
                auc = roc_auc_score(y_positive, record['scores'])
                assert abs(auc - target['roc_auc']) < 1e-12
                target['roc'] = dict(fpr=fpr.tolist(), tpr=tpr.tolist())
                target['roc_auc'] = auc
        np.testing.assert_array_equal(total, target['summed_confusion_matrix'])
        target['summed_confusion_matrix'] = total.tolist()
print('Saved evaluation records used:', prediction_data['source_evaluations'])
'''
