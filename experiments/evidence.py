"""Lossless dataset bundles containing one Parquet table per model/protocol.

The saved fitter keeps its loose-table and SQLite contracts. Read-only tools
can consume canonical ZIPs directly; unpack restores exact loose bytes for a
fresh-checkout resume without replacing any newer local checkpoint.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq
from experiments.storage import SCHEMA,unpack as unpack_row,read_table as read_loose


def bundle_path(path):
    path=Path(path)
    return path.parent/(path.name.split('__')[0]+'.zip')


def table_exists(path):
    path=Path(path)
    if path.exists(): return True
    bundle=bundle_path(path)
    if not bundle.exists(): return False
    with zipfile.ZipFile(bundle) as archive:return path.name in archive.namelist()


def read_table(path):
    path=Path(path)
    if path.exists(): return read_loose(path)
    with zipfile.ZipFile(bundle_path(path)) as archive:
        manifest=json.loads(archive.read('MANIFEST.json'))
        contents=archive.read(path.name)
        assert hashlib.sha256(contents).hexdigest()==manifest['tables'][path.name]['sha256']
    return dict(unpack_row(row) for row in pq.read_table(pa.BufferReader(contents),schema=SCHEMA).to_pylist())


def iter_tables(root):
    root=Path(root);paths=set(root.glob('*.parquet'))
    for path in root.glob('*.zip'):
        with zipfile.ZipFile(path) as archive:
            names=json.loads(archive.read('MANIFEST.json'))['tables']
            paths.update(root/name for name in names)
    return sorted(paths)


def validate_bundle(contents):
    with zipfile.ZipFile(io.BytesIO(contents)) as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('MANIFEST.json'))
        assert manifest['format']=='ifutsvm-group-tables-v1'
        assert set(archive.namelist())==set(manifest['tables'])|{'MANIFEST.json'}
        for name,record in manifest['tables'].items():
            assert Path(name).name==name and name.startswith(manifest['dataset']+'__') and name.endswith('.parquet')
            data=archive.read(name);assert hashlib.sha256(data).hexdigest()==record['sha256']
            table=pq.read_table(pa.BufferReader(data),schema=SCHEMA)
            assert len(table)==record['records']
        return manifest


def pack(root, completed_only=False):
    root=Path(root);groups={};output=[];previous={};run_groups=[]
    if completed_only:
        index=json.loads((root/'runs/index.json').read_text())
        config=json.loads((root/'experiment_config.json').read_text())
        assert index['source_hash']==config['source_hash']
        # Finished groups will not be rewritten by this running invocation.
        # Retain prior incomplete snapshots without opening their live files.
        for path in sorted((root/'runs').glob('*.zip')):
            previous[path.stem]=path;groups[path.stem]=[]
        for group in index['groups']:
            if group['outer_evaluations']!=5:continue
            path=root/group['file'];assert path.parent.resolve()==(root/'runs').resolve()
            groups.setdefault(group['dataset'],[]).append((path,group['sha256']))
    else:
        for path in sorted((root/'runs').glob('*.parquet')):
            groups.setdefault(path.name.split('__')[0],[]).append((path,None))
    for dataset,paths in sorted(groups.items()):
        contents={}
        if dataset in previous:
            with zipfile.ZipFile(previous[dataset]) as archive:
                prior=json.loads(archive.read('MANIFEST.json'));assert prior['dataset']==dataset
                for name,entry in prior['tables'].items():
                    data=archive.read(name);assert hashlib.sha256(data).hexdigest()==entry['sha256']
                    contents[name]=data
        for path,expected_hash in paths:
            data=path.read_bytes()
            if expected_hash is not None:assert hashlib.sha256(data).hexdigest()==expected_hash
            contents[path.name]=data
        manifest=dict(format='ifutsvm-group-tables-v1',dataset=dataset,tables={})
        for name,data in contents.items():
            table=pq.read_table(pa.BufferReader(data),schema=SCHEMA)
            keys=table['record_id'].to_pylist()
            manifest['tables'][name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),records=len(table),
                outer_evaluations=sum(k.endswith('/evaluation') for k in keys))
            _,model,protocol=Path(name).stem.split('__')
            run_groups.append(dict(dataset=dataset,model=model,protocol=protocol,file='runs/'+name,
                records=len(table),outer_evaluations=manifest['tables'][name]['outer_evaluations'],keys=keys,
                sha256=manifest['tables'][name]['sha256']))
        target=root/'runs'/(dataset+'.zip');temporary=target.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as archive:
            for name,data in sorted(contents.items()):archive.writestr(zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0)),data)
            archive.writestr(zipfile.ZipInfo('MANIFEST.json',date_time=(2026,1,1,0,0,0)),json.dumps(manifest,indent=2)+'\n')
        validate_bundle(temporary.read_bytes());temporary.replace(target)
        output.append(dict(path=target.relative_to(root).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),**manifest))
    index=dict(bundles=output,tables=sum(len(b['tables']) for b in output),
        outer_evaluations=sum(t['outer_evaluations'] for b in output for t in b['tables'].values()),
        layout='One original typed Parquet table per dataset/model/protocol, stored losslessly inside a dataset ZIP. No records, predictions or metadata rewritten.')
    config_path=root/'experiment_config.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    index['run_index']=dict(schema_version=1,source_hash=config.get('source_hash'),groups=run_groups)
    (root/'bundles.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f"Bundled {index['tables']} exact group tables / {index['outer_evaluations']} outer records into {len(output)} dataset ZIPs")
    return index


def restore(root):
    root=Path(root);written=0;kept=0
    for path in sorted((root/'runs').glob('*.zip')):
        data=path.read_bytes();manifest=validate_bundle(data)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in manifest['tables']:
                target=root/'runs'/name
                if target.exists():kept+=1;continue
                target.write_bytes(archive.read(name));written+=1
    print(f'Restored {written} exact tables; kept {kept} existing local checkpoints')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['pack','unpack','verify'])
    parser.add_argument('--output',default='results')
    parser.add_argument('--completed-only',action='store_true',help='Snapshot only finished immutable groups while a main run is active')
    args=parser.parse_args();root=Path(args.output)
    if args.action=='pack':pack(root,args.completed_only)
    elif args.action=='unpack':restore(root)
    else:
        paths=sorted((root/'runs').glob('*.zip'));tables=0;outer=0
        for path in paths:
            manifest=validate_bundle(path.read_bytes());tables+=len(manifest['tables'])
            outer+=sum(t['outer_evaluations'] for t in manifest['tables'].values())
        print(f'Verified {len(paths)} bundles / {tables} tables / {outer} outer records; zero changed bytes')


if __name__=='__main__':main()
