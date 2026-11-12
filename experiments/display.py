"""Canonical model names for human-facing artifacts; IDs remain machine keys."""
import textwrap

MODEL_NAMES={
    'svm_balanced':'SVM (balanced)',
    'smote_svm':'SMOTE + SVM',
    'tsvm':'TSVM',
    'utsvm':'UTSVM',
    'faithful_raw':'IFUTSVM-ID (published code)',
    'faithful_scaled':'IFUTSVM-ID (published code, scaled)',
    'paper':'IFUTSVM-ID (corrected)',
    'no_universum':'IFUTSVM-ID (no Universum)',
    'code_universum':'IFUTSVM-ID (published Universum)',
    'isotropic_noise':'IFUTSVM-ID (Gaussian-noise Universum)',
    'conditional_gaussian':'IFUTSVM-ID (conditional Gaussian)',
    'svm_rbf':'SVM (RBF, accuracy diagnostic)',
    'random_forest':'Random forest',
    'gradient_boosting':'Gradient boosting',
    'knn':'k-nearest neighbours',
}
MODEL_NAMES['random_average']=MODEL_NAMES['paper']
MODEL_NAMES['paper_main']=MODEL_NAMES['paper']
CONTROL_NAMES={
    'demo_ridge':'Executable ridge', 'drop_nominal':'Nominal feature omitted',
    'full_basis':'Full kernel basis', 'paper_dual':'Printed dual sign',
    'paper_primal':'Printed primal sign', 'published_scaled':'Published parameters, scaled',
    'raw_features':'Raw features', 'rkhs_norm':'RKHS normalization',
    'unstratified':'Unstratified split',
}
for base in ['random_average','conditional_gaussian']:
    for suffix,description in [('size_0.25','u × 0.25'),('size_0.5','u × 0.5'),
            ('size_2','u × 2'),('if_off','IF scores off'),('paper_dual','paper-dual sign')]:
        MODEL_NAMES[base+'_'+suffix]=MODEL_NAMES[base][:-1]+'; '+description+')'


def model_name(identifier,width=None):
    value=MODEL_NAMES[identifier]
    return '\n'.join(textwrap.wrap(value,width=width,break_long_words=False,break_on_hyphens=False)) if width else value


def published_name(identifier):
    return 'IFUTSVM-ID (as published)' if identifier=='IFUTSVM-ID' else identifier


def display_structure(value):
    """Readable model keys in displayed statistics, leaving machine data intact."""
    if isinstance(value,dict):return {MODEL_NAMES.get(key,key):display_structure(child) for key,child in value.items()}
    if isinstance(value,list):return [display_structure(child) for child in value]
    return MODEL_NAMES.get(value,value) if isinstance(value,str) else value
