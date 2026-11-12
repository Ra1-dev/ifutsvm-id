"""the independent audit must reject contaminated scaler provenance."""
import json
import pytest
from ifutsvm.data import load
from experiments.protocol import split_plan
from experiments.run_all import run_group,source_hash,write_json
from experiments.audit import audit
from experiments.storage import read_table,write_table,table_path


def test_audit_rejects_changed_scaler_statistics(tmp_path):
    dataset='ecoli4'; _,y=load(dataset)
    split_id,seed,train,test=next(split_plan(y,'holdout'))
    config=dict(output=str(tmp_path),budget=1,inner_folds=3,source_hash=source_hash(),versions={},
                datasets=[dataset],models=['svm_balanced'])
    write_json(tmp_path/'experiment_config.json',config)
    run_group((dataset,'holdout','svm_balanced',config))
    result=audit(tmp_path,partial=True)
    assert result['outer_records']==5 and result['inner_records']==15
    path=table_path(tmp_path,dataset,'svm_balanced','holdout')
    records=read_table(path); record=records[f'{split_id}/inner/candidate-0-fold-0']
    record['metadata']['scaler']['mean'][0]+=1
    write_table(path,records)
    with pytest.raises(AssertionError):
        audit(tmp_path,partial=True)
def test_reproduction_rkhs_record_identity(tmp_path):
    from experiments.reproduction_probes import run_probe
    from experiments.storage import read_table
    records={}
    job=('ecoli4','rkhs_norm',0,str(tmp_path),'storage-contract')
    assert run_probe(job,records)['error'] is None
    path=tmp_path/'reproduction_probes/ecoli4.parquet'
    assert list(read_table(path))==['rkhs_norm/seed-0']
    assert run_probe(job,records)['cached']


def test_bundle_preserves_every_byte_and_resume_keeps_newer_checkpoint(tmp_path):
    from experiments.evidence import pack,restore,read_table as read_evidence,table_exists,iter_tables
    path=table_path(tmp_path,'ecoli4','paper','holdout')
    records={'holdout-0/evaluation':dict(prediction=['positive','negative'],scores=[.7,-.4],
             train_indices=[0,1],test_indices=[2,3],metadata={'source_hash':'unchanged','seed':0})}
    write_table(path,records);original=path.read_bytes();expected=read_table(path)
    index=pack(tmp_path);assert index['tables']==1 and index['outer_evaluations']==1
    path.unlink()
    assert table_exists(path) and iter_tables(tmp_path/'runs')==[path]
    assert read_evidence(path)==expected
    restore(tmp_path);assert path.read_bytes()==original
    records['holdout-0/evaluation']['scores']=[.8,-.5]
    write_table(path,records);newer=path.read_bytes()
    restore(tmp_path);assert path.read_bytes()==newer


def test_completed_group_snapshot_never_reads_active_table(tmp_path):
    import hashlib
    from zipfile import ZipFile
    from experiments.evidence import pack
    active=table_path(tmp_path,'ecoli4','paper','holdout')
    record=dict(prediction=['positive'],scores=[.2],train_indices=[0],test_indices=[1],metadata={})
    write_table(active,{'holdout-0/evaluation':record});prior=active.read_bytes()
    pack(tmp_path)
    finished=table_path(tmp_path,'ecoli4','svm_balanced','holdout')
    write_table(finished,{f'holdout-{seed}/evaluation':record for seed in range(5)})
    stable=finished.read_bytes()
    write_json(tmp_path/'experiment_config.json',dict(source_hash='frozen'))
    write_json(tmp_path/'runs/index.json',dict(source_hash='frozen',groups=[dict(dataset='ecoli4',
        file=finished.relative_to(tmp_path).as_posix(),outer_evaluations=5,sha256=hashlib.sha256(stable).hexdigest())]))
    # This deliberately invalid live generation must never be opened by pack.
    active.write_bytes(b'writer currently owns this incomplete table')
    result=pack(tmp_path,completed_only=True)
    assert result['tables']==2 and result['outer_evaluations']==6
    assert result['run_index']['source_hash']=='frozen'
    assert sum(row['outer_evaluations'] for row in result['run_index']['groups'])==6
    with ZipFile(tmp_path/'runs/ecoli4.zip') as archive:
        assert archive.read(active.name)==prior and archive.read(finished.name)==stable

