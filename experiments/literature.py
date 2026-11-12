"""Extract factual comparison values from the supplied supplementary PDF.

These published single-split accuracies are contextual evidence, not another
member of our matched-split statistical comparison. No MATLAB source is copied.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics


METHODS = ['UTSVM', 'RUTSVM-CIL', 'IF-USVM', 'IFW-LSTSVM', 'CGFTSVM-ID', 'IFUTSVM-ID']
ALIASES = {'ecoli-0-1_vs_5': 'ecoli-0-1 vs 5', 'haberman': 'haber',
           'yeast-2_vs_4': 'yeast-2 vs 4'}


def extract(pdf_path):
    import pypdfium2 as pdfium
    published = json.loads(Path('data/published_table_sii.json').read_text())['datasets']
    manifest = json.loads(Path('data/keel/MANIFEST.json').read_text())['datasets']
    document = pdfium.PdfDocument(pdf_path)
    pages = [page.get_textpage().get_text_range() for page in document]
    table_page = next(i for i, text in enumerate(pages) if 'TABLE S.II:' in text)
    lines = [line.strip() for line in pages[table_page].splitlines() if line.strip()]
    datasets = {}
    for name, expected in published.items():
        label = ALIASES.get(name, name)
        matches = [(i, line) for i, line in enumerate(lines) if line.startswith(label + ' ')]
        assert len(matches) == 1, (name, matches)
        index, line = matches[0]
        values = [float(v) for v in line[len(label):].split()]
        assert len(values) == len(METHODS) and all(0 <= v <= 100 for v in values)
        dimensions = re.match(r'\((\d+)\s*×\s*(\d+)\)', lines[index + 1])
        assert dimensions, (name, lines[index + 1])
        rows, features = map(int, dimensions.groups())
        assert values[-1] == expected['accuracy_percent']
        assert (rows, features) == (expected['rows'], expected['features'])
        datasets[name] = dict(published_label=label, accuracy_percent=dict(zip(METHODS, values)),
            published_rows=rows, published_features=features, raw_rows=manifest[name]['rows'],
            raw_features=manifest[name]['features'], source_line=line,
            dimension_match=(rows, features) == (manifest[name]['rows'], manifest[name]['features']))
    references_text = next(text.split('REFERENCES', 1)[1] for text in pages if 'REFERENCES' in text)
    references = {}
    for number, body in re.findall(r'\[(\d+)\]\s+(.*?)(?=\[\d+\]|\Z)', references_text, re.S):
        if int(number) >= 5:
            references[number] = ' '.join(body.replace('\ufffe', '').split())
    return dict(source=dict(path=str(pdf_path), sha256=hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest(),
        table='S.II', page=table_page + 1, title='Robust Universum Twin Support Vector Machine for Imbalanced Data',
        authors=['M. Tanveer', 'A. Quadir'], arxiv='2410.20335v2'),
        methods=METHODS, datasets=datasets, references_as_printed=references,
        reference_numbers=dict(zip(METHODS[:-1], ['6', '7', '8', '9', '10'])),
        subset_mean_accuracy_percent={method: statistics.mean(row['accuracy_percent'][method]
            for row in datasets.values()) for method in METHODS},
        scope='Only the pre-specified twelve datasets; these subset means are not the paper full-table means.',
        comparability=['Published single random 70/30 split; indices, seed and full preprocessing unavailable.',
            'Our main comparison uses repeated stratified holdout and nested CV with training-only tuning.',
            'Raw Abalone has a nominal feature beyond the published feature count; the difference is retained.',
            'Published ACC is not paired with our runs; no across-protocol significance or fair-ranking claim.',
            'SVM and TSVM do not have columns in Table S.II; their published values are unavailable here.'])


def main():
    result = extract(Path('reference/IFUTSVM-ID/IFUTSVM-ID-Supplementary.pdf'))
    Path('results/literature.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8', newline='\n')
    print(f"Published comparison: {len(result['datasets'])} datasets x {len(result['methods'])} methods; "
          f"IFUTSVM column and dimensions cross-checked; {sum(not row['dimension_match'] for row in result['datasets'].values())} raw-dimension mismatch")


if __name__ == '__main__':
    main()
