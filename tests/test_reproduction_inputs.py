"""Synthetic input-contract tests; no real model fitting or output replacement."""
import argparse
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import check_setup
from src.submission_contract import submission_frame
from src.research.cache_inputs import add_cache_arguments, reference_run, housing_run, require_files


def test_absent_template_constructs_layout_without_writing(tmp_path):
    frame = submission_frame(tmp_path, 4)
    assert frame.columns.tolist() == ['Id','Predicted']
    assert frame.Id.tolist() == [0,1,2,3]
    assert list(tmp_path.iterdir()) == []
    template = tmp_path/'example-submission.csv'
    frame.to_csv(template,index=False)
    pd.testing.assert_frame_equal(frame,submission_frame(tmp_path,4))


@pytest.mark.parametrize('frame', [pd.DataFrame({'Id':[1,0],'Predicted':[0,0]}),
    pd.DataFrame({'Id':[0],'Predicted':[0]}),pd.DataFrame({'wrong':[0,1]})])
def test_present_bad_template_is_not_silently_replaced(tmp_path,frame):
    frame.to_csv(tmp_path/'example-submission.csv',index=False)
    with pytest.raises(ValueError,match='Invalid example'):
        submission_frame(tmp_path,2)


def test_setup_official_eight_file_layout_without_template(tmp_path,monkeypatch):
    payloads = {'train.csv':b'A,MONTHLY_RENT\n1,2000\n','test.csv':b'A\n1\n'}
    payloads.update({f'auxiliary/{n}': b'A\n1\n' for n in check_setup.AUXILIARY_FILES})
    raw=tmp_path/'data/raw'
    for name,data in payloads.items():
        p=raw/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}
    hashes['example-submission.csv']='unused-when-absent'
    monkeypatch.setattr(check_setup,'ROOT',tmp_path)
    monkeypatch.setattr(check_setup,'RAW_DIR',raw)
    monkeypatch.setattr(check_setup,'RAW_SHA256',hashes)
    assert check_setup.check_data()==[]
    (raw/'test.csv').write_bytes(b'A\n9\n')
    assert any('Data version mismatch' in s for s in check_setup.check_data())


def test_cache_override_and_complete_missing_message(tmp_path):
    parser=argparse.ArgumentParser();add_cache_arguments(parser)
    args=parser.parse_args(['--reference-run-dir',str(tmp_path/'new-id')])
    with pytest.raises(FileNotFoundError,match='short_recent_scoring_rows.csv') as exc:
        reference_run(args,tmp_path/'old-id')
    assert '--reference-run-dir' in str(exc.value) and 'compare_models' in str(exc.value)
    folder=tmp_path/'new-id';folder.mkdir()
    names=['short_recent_scoring_rows.csv','short_recent_local_trend_0.pkl','short_recent_reference_ensemble.npy']
    names += [f'short_recent_reference_{v}_{m}_42.npy' for v in ['local_trend_0','hierarchical_6'] for m in ['direct','residual']]
    for n in names:(folder/n).touch()
    assert reference_run(args,tmp_path/'old-id')==folder.resolve()
    with pytest.raises(FileNotFoundError,match='--housing-run-dir'):
        housing_run(args,tmp_path/'no-house')
    with pytest.raises(FileNotFoundError,match='long_recent'):
        reference_run(args,folder,['long_recent'])


def test_data_audit_missing_template_does_not_claim_course_file(tmp_path):
    from src.data_audit import audit
    source=Path(__file__).resolve().parents[1]/'data/raw'
    raw=tmp_path/'data/raw';(raw/'auxiliary').mkdir(parents=True)
    for name in ['train.csv','test.csv']:
        pd.read_csv(source/name,nrows=40).to_csv(raw/name,index=False)
    # Only a derived test fixture is written; never touch the real course inputs.
    pd.read_csv(source/'auxiliary/sg-hdb-block.csv').to_csv(raw/'auxiliary/sg-hdb-block.csv',index=False)
    result=audit(tmp_path)
    assert result['submission_contract_valid']
    assert 'in-memory' in result['submission_layout_source']
    assert 'example-submission.csv' not in result['raw_file_sha256']


def test_both_final_entry_paths_use_shared_submission_contract(monkeypatch):
    import sys
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'src'))
    import train_final_model as final
    assert 'submission_frame(RAW, len(test))' in inspect.getsource(final.main)
    assert 'submission_frame(RAW, len(test))' in inspect.getsource(final.create_reweighted_submission)


@pytest.mark.parametrize('module',['probe_rmse_structure','train_hdb_attributes'])
def test_cache_cli_fails_before_work_with_useful_message(tmp_path,module):
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/'must-not-be-created'
    result=subprocess.run([sys.executable,'-m','src.research.'+module,
        '--reference-run-dir',str(tmp_path/'missing-run'),'--output-dir',str(output)],
        cwd=root,capture_output=True,text=True)
    assert result.returncode != 0
    assert 'Incomplete cache' in result.stderr and '--reference-run-dir' in result.stderr
    assert 'short_recent_scoring_rows.csv' in result.stderr
    assert not output.exists()


def test_final_entry_smoke_without_template_or_real_training(tmp_path,monkeypatch):
    import json
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'src'))
    import train_final_model as final
    raw=tmp_path/'data/raw';raw.mkdir(parents=True)
    pd.DataFrame({'RENT_APPROVAL_DATE':['2024-01','2024-02'],'MONTHLY_RENT':[2000,2200]}).to_csv(raw/'train.csv',index=False)
    pd.DataFrame({'RENT_APPROVAL_DATE':['2025-01','2025-02']}).to_csv(raw/'test.csv',index=False)
    (tmp_path/'src').mkdir()
    for name in ['features.py','modeling.py','train_final_model.py']:
        (tmp_path/'src'/name).write_text('# synthetic smoke fixture\n')
    config=json.loads((root/'configs/model.json').read_text())
    config.update(iterations=1,seeds=[42])
    monkeypatch.setattr(final,'ROOT',tmp_path)
    monkeypatch.setattr(final,'RAW',raw)
    monkeypatch.setattr(final,'load_config',lambda *a,**kw:dict(config))
    monkeypatch.setattr('sys.argv',['train_final_model.py','--iterations','1','--seeds','42',
        '--model-prefix','smoke/model','--submission',str(tmp_path/'submissions/smoke.csv'),
        '--metadata',str(tmp_path/'reports/smoke.json'),'--task-type','CPU'])
    monkeypatch.setattr(final,'build_block_lookup',lambda *a:None)
    monkeypatch.setattr(final,'suspected_low_rent_mask',lambda a,*rest,**kw:pd.Series(False,index=a.index))
    monkeypatch.setattr(final,'build_features',lambda a,b,q,*rest:pd.DataFrame({'x':np.zeros(len(q))}))
    monkeypatch.setattr(final,'build_training_features',lambda a,b,*rest:(pd.DataFrame({'x':np.zeros(len(a))}),b,np.ones(len(a))))
    class FakeModel:
        tree_count_=1
        def save_model(self,path):Path(path).write_text('synthetic smoke stub, not a real model')
        def load_model(self,path):assert Path(path).is_file()
    monkeypatch.setattr(final,'fit_model',lambda *a,**kw:FakeModel())
    monkeypatch.setattr(final,'predict_model',lambda model,q,*a:np.full(len(q),2100.0))
    final.main()
    out=pd.read_csv(tmp_path/'submissions/smoke.csv')
    assert out.Id.tolist()==[0,1] and out.Predicted.tolist()==[2100.0,2100.0]
    assert len(json.loads((tmp_path/'reports/smoke.json').read_text())['components'])==4
    assert not (raw/'example-submission.csv').exists()
