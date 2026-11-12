"""Full KEEL data acquisition with checksums; loading never uses the network."""
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin,unquote,urlparse
from urllib.request import urlopen,Request
import numpy as np

DATASETS=('abalone9-18','ecoli-0-1_vs_5','ecoli3','ecoli4','glass4','haberman',
          'new-thyroid1','pima','vehicle1','yeast1','yeast-2_vs_4','yeast3')
INDEX='https://sci2s.ugr.es/keel/imbalanced.php'


def load_dat(path, *, return_schema=False):
    """Return published columns/labels, preserving nominal values and row order."""
    text=Path(path).read_text(encoding='utf-8-sig')
    attributes=[]; records=[]; data=False; inputs=None; outputs=None
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('%'):
            continue
        if data:
            records.append(next(csv.reader([line],skipinitialspace=True)))
        elif line.lower().startswith('@attribute'):
            match=re.match(r'@attribute\s+(?:\'([^\']+)\'|"([^"]+)"|(\S+))\s+(.+)',line,re.I)
            if not match:
                raise ValueError(f'invalid attribute declaration: {line}')
            name=next(g for g in match.groups()[:3] if g is not None)
            kind=match.group(4).strip()
            attributes.append(dict(name=name,kind=kind,nominal=kind.startswith('{')))
        elif line.lower().startswith('@input'):
            inputs=[s.strip().strip('\'"') for s in line.split(maxsplit=1)[1].split(',') if s.strip()]
        elif line.lower().startswith('@output'):
            outputs=[s.strip().strip('\'"') for s in line.split(maxsplit=1)[1].split(',') if s.strip()]
        elif line.lower()=='@data':
            data=True
    names=[a['name'] for a in attributes]
    if outputs is None:
        outputs=names[-1:]
    if len(outputs)!=1:
        raise ValueError('one output attribute required')
    output=names.index(outputs[0])
    columns=[names.index(n) for n in inputs] if inputs is not None else [i for i in range(len(names)) if i!=output]
    if not records or any(len(row)!=len(attributes) for row in records):
        raise ValueError('data row width differs from header')
    schema=[attributes[j] for j in columns]
    X=np.empty((len(records),len(columns)),dtype=object if any(a['nominal'] for a in schema) else float)
    for i,row in enumerate(records):
        for k,j in enumerate(columns):
            value=row[j].strip().strip('\'"')
            X[i,k]=np.nan if value=='?' else value if attributes[j]['nominal'] else float(value)
    y=np.array([row[output].strip().strip('\'"') for row in records])
    return (X,y,schema) if return_schema else (X,y)


def load(name,root='data/keel'):
    if name not in DATASETS:
        raise ValueError(f'unknown issued dataset: {name}')
    root=Path(root)
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    entry=manifest['datasets'][name]
    path=root/entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
        raise ValueError(f'checksum mismatch: {name}')
    return load_dat(path)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            self.urls.extend(v for k,v in attrs if k=='href' and v)


def _download(url):
    with urlopen(Request(url,headers={'User-Agent':'CSCI447P031 reproducibility study'}),timeout=60) as response:
        return response.read()


def fetch(root='data/keel',refresh=False):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    previous=json.loads((root/'MANIFEST.json').read_text()) if (root/'MANIFEST.json').exists() else {'datasets':{}}
    parser=_Links(); parser.feed(_download(INDEX).decode('utf-8',errors='replace'))
    links={Path(unquote(urlparse(u).path)).name.lower():urljoin(INDEX,u) for u in parser.urls}
    def one(name):
        old=previous['datasets'].get(name)
        if old and not refresh:
            load(name,root)
            return name,old
        if name.lower()+'.zip' not in links:
            raise ValueError(f'full-data ZIP link missing from official index: {name}')
        url=links[name.lower()+'.zip']
        payload=_download(url)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            matches=[n for n in archive.namelist() if Path(n).name.lower()==name.lower()+'.dat']
            if len(matches)!=1:
                raise ValueError(f'expected exactly one full .dat in {name}: {archive.namelist()}')
            raw=archive.read(matches[0])
        checksum=hashlib.sha256(raw).hexdigest()
        if old and old['sha256']!=checksum:
            raise ValueError(f'published file changed since manifest: {name}')
        path=root/f'{name}.dat'; path.write_bytes(raw)
        X,y,schema=load_dat(path,return_schema=True)
        labels,counts=np.unique(y,return_counts=True)
        entry=dict(file=path.name,source_url=url,sha256=checksum,
                   archive_sha256=hashlib.sha256(payload).hexdigest(),rows=len(X),features=X.shape[1],
                   class_counts=dict(zip(labels.tolist(),counts.tolist())),minority_label=str(labels[np.argmin(counts)]),
                   imbalance_ratio=float(max(counts)/min(counts)),attributes=schema)
        print(f'{name}: {X.shape}, classes={entry["class_counts"]}, sha256={checksum}',flush=True)
        return name,entry
    with ThreadPoolExecutor(max_workers=4) as executor:
        entries=dict(executor.map(one,DATASETS))
    manifest=dict(collection_url=INDEX,datasets=entries)
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['fetch'])
    parser.add_argument('--root',default='data/keel')
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    fetch(args.root,args.refresh)
