"""One typed Parquet table per dataset/model/protocol, with resumable local cache."""
import json
from pathlib import Path
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq

METRICS=('accuracy','precision','recall','specificity','f1','roc_auc','gmean','balanced_accuracy','training_seconds')
LIST_INT=pa.list_(pa.int64()); LIST_FLOAT=pa.list_(pa.float64())
SCHEMA=pa.schema([
    ('record_id',pa.string()),('positive_label',pa.string()),
    ('train_indices',LIST_INT),('test_indices',LIST_INT),
    ('y',pa.list_(pa.string())),('prediction',pa.list_(pa.string())),('scores',LIST_FLOAT),
    *[(name,pa.float64()) for name in METRICS],
    ('confusion_matrix',pa.list_(LIST_INT)),
    ('roc',pa.struct([('fpr',LIST_FLOAT),('tpr',LIST_FLOAT),('thresholds',LIST_FLOAT)])),
    ('error',pa.string()),('error_stage',pa.string()),('metadata_json',pa.string()),('extra_json',pa.string())])
FIELDS={f.name for f in SCHEMA}-{'record_id','metadata_json','extra_json'}


def pack(key,record):
    row={name:record.get(name) for name in FIELDS}
    row.update(record_id=key,metadata_json=json.dumps(record.get('metadata',{}),separators=(',',':')),
               extra_json=json.dumps({k:v for k,v in record.items() if k not in FIELDS and k!='metadata'},separators=(',',':')))
    # KEEL labels are published strings. Synthetic experiments use their own
    # compact table schema rather than changing the benchmark label contract.
    return row


def unpack(row):
    record={k:v for k,v in row.items() if k in FIELDS and v is not None}
    record.update(json.loads(row['extra_json']))
    metadata=json.loads(row['metadata_json'])
    if metadata: record['metadata']=metadata
    return row['record_id'],record


def table_path(root,dataset,model,protocol):
    return Path(root)/'runs'/f'{dataset}__{model}__{protocol}.parquet'


def read_table(path):
    return dict(unpack(row) for row in pq.read_table(path,schema=SCHEMA).to_pylist())


def write_table(path,records):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.parquet.tmp')
    table=pa.Table.from_pylist([pack(k,r) for k,r in sorted(records.items())],schema=SCHEMA)
    pq.write_table(table,temporary,compression='zstd')
    temporary.replace(path)


class RunStore:
    """A single worker owns a group. SQLite checkpoints every attempted fit.

    SQLite is disposable local cache under tmp, never a deliverable. Parquet is
    exported after each completed outer split and can restore the cache in a
    clean checkout. Thus every candidate is resumable without thousands of files.
    """
    def __init__(self,root,dataset,model,protocol,fingerprint):
        self.path=table_path(root,dataset,model,protocol)
        token=f'{dataset}__{model}__{protocol}'
        # Include the output path so smoke runs cannot contaminate real caches.
        import hashlib
        location=hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:12]
        cache=Path('tmp')/'experiment-cache'/fingerprint/location/f'{token}.sqlite'
        cache.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(cache)
        self.db.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        if self.path.exists():
            for key,r in read_table(self.path).items():
                self.db.execute('INSERT OR IGNORE INTO records VALUES (?,?)',(key,json.dumps(r,separators=(',',':'))))
            self.db.commit()

    def get(self,key):
        row=self.db.execute('SELECT payload FROM records WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self,key,record):
        self.db.execute('INSERT OR REPLACE INTO records VALUES (?,?)',(key,json.dumps(record,separators=(',',':'),allow_nan=False)))
        self.db.commit()
        return record

    def export(self):
        records={k:json.loads(v) for k,v in self.db.execute('SELECT key,payload FROM records')}
        write_table(self.path,records)
        return records

    def close(self):
        self.db.close()


def build_index(root,config):
    rows=[]
    for dataset in config['datasets']:
        for model in config['models']:
            for protocol in ('holdout','nested_cv'):
                path=table_path(root,dataset,model,protocol)
                if not path.exists(): continue
                records=read_table(path)
                rows.append(dict(dataset=dataset,model=model,protocol=protocol,file=path.relative_to(root).as_posix(),
                    records=len(records),outer_evaluations=sum(k.endswith('/evaluation') for k in records),
                    keys=sorted(records)))
    return dict(schema_version=1,unit='one row per fit or selection record',
                source_hash=config['source_hash'],groups=rows)
