"""Actual upstream sampler parity, EMA mapping and Slurm isolation for author I2SB."""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from mgm_i2i.cluster import official_i2sb as official


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def upstream():
    path=ROOT/'third_party/I2SB-official'
    if not path.exists():
        pytest.skip('Clone the pinned official repo to run real upstream parity tests')
    official.verify_upstream(path)
    return path


@pytest.mark.parametrize('nfe',[1,4,16,64])
def test_sampler_matches_original_ddpm_loop_and_restores_rng(upstream,nfe):
    schedule=dict(interval=1000,beta_max=.3,t0=1e-4,T=1.,clip_denoise=False)
    bridge=official.OfficialBridge(upstream,schedule,'cpu')
    _,_,spacing=official.reference_code(upstream)
    calls=[]
    def denoiser(x,index):
        calls.append(index.item())
        return .1*x+index.float().reshape(1,1,1,1)/1000
    image=torch.randn(1,3,4,4)
    before=torch.get_rng_state().clone()
    generator=torch.Generator().manual_seed(777)
    actual=bridge.sample(denoiser,image,nfe,generators=[generator])
    assert torch.equal(before,torch.get_rng_state())
    assert len(calls)==nfe
    with torch.random.fork_rng():
        torch.manual_seed(777)
        def predict(x,step):
            index=torch.full((1,),step,dtype=torch.long)
            return x-bridge.diffusion.get_std_fwd(index,xdim=x.shape[1:])*denoiser(x,index)
        result,_=bridge.diffusion.ddpm_sampling(spacing(1000,nfe+1),predict,image,
                                              log_steps=[0],verbose=False)
    torch.testing.assert_close(actual,result[:,0],rtol=0,atol=0)
    two=bridge.sample(denoiser,image.repeat(2,1,1,1),nfe,
                      generators=[torch.Generator().manual_seed(s) for s in (777,778)])
    torch.testing.assert_close(two[:1],actual,rtol=0,atol=0)
    if nfe>1:
        assert not torch.equal(two[0],two[1])


def test_original_adm_meta_count_and_shape_without_allocating_weights(upstream):
    cls,_,_=official.reference_code(upstream)
    with torch.device('meta'):
        model=cls(**official.ARCHITECTURE)
        assert sum(p.numel() for p in model.parameters())==official.PARAMETERS
        with torch.no_grad():
            for size in (128,256):
                assert model(torch.empty(1,3,size,size),torch.ones(1)).shape==(1,3,size,size)


def test_ema_shadow_mapping_uses_author_parameter_order_and_rejects_mismatch():
    model=nn.Sequential(nn.Linear(3,4),nn.Linear(4,2))
    state=dict(net={'diffusion_model.'+k:v.clone() for k,v in model.state_dict().items()},
               ema=dict(shadow_params=[torch.full_like(p,i+10) for i,p in enumerate(model.parameters())]))
    weights=official.ema_weights(model,state)
    for index,(key,_) in enumerate(model.named_parameters()):
        assert torch.all(weights[key]==index+10)
    wrong=copy.deepcopy(state);wrong['ema']['shadow_params'][0]=torch.ones(1)
    with pytest.raises(ValueError,match='EMA tensor'):
        official.ema_weights(model,wrong)
    wrong=copy.deepcopy(state);wrong['net'].pop(next(iter(wrong['net'])))
    with pytest.raises(ValueError,match='keys'):
        official.ema_weights(model,wrong)


def test_preparation_exports_only_ema_and_reuses_a_verified_bundle(tmp_path,monkeypatch):
    module=script('prepare_i2sb_official')
    upstream=tmp_path/'upstream';upstream.mkdir()
    weights=tmp_path/'weights';raw=weights/'author_download';raw.mkdir(parents=True)
    class Tiny(nn.Sequential):
        def __init__(self,**kwargs):super().__init__(nn.Linear(3,2))
    net=Tiny();count=sum(p.numel() for p in net.parameters())
    checkpoint=dict(net={'diffusion_model.'+k:v for k,v in net.state_dict().items()},
                    ema=dict(shadow_params=[torch.full_like(p,9) for p in net.parameters()]),
                    optimizer={'unused':'not exported'})
    torch.save(checkpoint,raw/'latest.pt')
    options=argparse.Namespace(corrupt='sr4x-pool',image_size=256,interval=1000,beta_max=.3,t0=1e-4,T=1.)
    (raw/'options.pkl').write_bytes(pickle.dumps(options))
    monkeypatch.setattr(module,'verify_upstream',lambda root:root)
    monkeypatch.setattr(module,'reference_code',lambda root:(Tiny,None,None))
    monkeypatch.setattr(module,'PARAMETERS',count)
    monkeypatch.setattr(official,'PARAMETERS',count)
    manifest=module.prepare(upstream,weights,False)
    exported=torch.load(weights/manifest['ema_file'],weights_only=True)
    assert set(exported)==set(net.state_dict()) and all(torch.all(t==9) for t in exported.values())
    assert manifest['source_checkpoint_sha256']==official.digest(raw/'latest.pt')
    assert module.prepare(upstream,weights,False)==manifest


def test_options_namespace_and_paths_are_read_without_arbitrary_pickle_globals(tmp_path):
    options=dict(corrupt='sr4x-pool',image_size=256,interval=1000,beta_max=.3,
                 t0=1e-4,T=1.,dataset_dir=Path('/dataset'))
    path=tmp_path/'options.pkl';path.write_bytes(pickle.dumps(argparse.Namespace(**options)))
    result=official.read_options(path)
    assert result==options
    assert official.validate_options(result)['clip_denoise'] is False
    with pytest.raises(ValueError,match='unconditioned'):
        official.validate_options(dict(result,cond_x1=True))
    with pytest.raises(pickle.UnpicklingError,match='Unexpected options class'):
        official.OptionsUnpickler(io.BytesIO(pickle.dumps(complex(1,2)))).load()


def test_bundle_checks_weights_architecture_and_pinning(tmp_path):
    path=tmp_path/'official_ema.pt';path.write_bytes(b'weights')
    manifest=dict(upstream_commit=official.UPSTREAM_COMMIT,parameters=official.PARAMETERS,
                  architecture=official.ARCHITECTURE,ema_file=path.name,ema_sha256=official.digest(path))
    meta=tmp_path/'manifest.json';meta.write_text(json.dumps(manifest))
    assert official.load_bundle(meta)[1]==path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        official.load_bundle(meta)
    manifest['upstream_commit']='changed';meta.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='version'):
        official.load_bundle(meta)


def test_official_plan_probe_dependencies_single_checkpoint_and_unknown_training_cost(tmp_path,monkeypatch):
    module=script('cluster_i2sb_official_submit')
    checkpoint=tmp_path/'official_ema.pt';checkpoint.write_bytes(b'test')
    bundle=tmp_path/'manifest.json';bundle.write_text('{}')
    monkeypatch.setattr(module,'load_multicrop_evaluation',lambda *a: ({'preparation':{'eval_crops_per_image':5}},'common-eval'))
    monkeypatch.setattr(module,'load_bundle',lambda *a: ({},checkpoint))
    monkeypatch.setattr(module,'verify_upstream',lambda p:p)
    real_digest=module.digest
    monkeypatch.setattr(module,'digest',lambda p:'7be5be79' if str(p).endswith('alexnet-owt-7be5be79.pth') else real_digest(p))
    args=SimpleNamespace(data_root=tmp_path,split='val',bundle=bundle,upstream=tmp_path,
                         weights_root=tmp_path,run_root=tmp_path/'run',cpus=4,nfe=[1,4,16,64],
                         partition='rocky',max_parallel=2,account='',constraint='',gpu_type='',
                         probe_minutes=20,eval_minutes=180)
    plan=module.make_plan(args)
    assert plan['phase_order']==['probe','eval'] and len(plan['probe'])==1 and len(plan['eval'])==4
    assert plan['external_pretrained'] and not plan['train'] and not plan['warmup']
    assert {x['checkpoint'] for x in plan['eval']}=={str(checkpoint)}
    assert {x['config']['seed'] for x in plan['eval']}=={0}
    assert plan['budget']['external_training_gpu_hours'] is None
    submit=script('cluster_submit')
    calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout=f'{900+len(calls)}\n')
    monkeypatch.setattr(submit.subprocess,'run',fake)
    submit.execute_plan(plan,args.run_root,True)
    assert len(calls)==3
    assert '--job-name=i2sb_ref_probe' in calls[0]
    assert '--dependency=afterok:901' in calls[1]
    assert '--dependency=afterany:902' in calls[2]
    assert all(not arg.startswith('--mem') for call in calls for arg in call)
    assert '--array=0-3%2' in calls[1]
    with pytest.raises(FileExistsError,match='Submission already'):
        submit.execute_plan(plan,args.run_root,True)


def test_official_evaluator_dispatch_and_collector_do_not_fabricate_training(tmp_path,monkeypatch):
    # Reuse the CPU stand-in used by the shared evaluator tests; the official
    # loader is replaced with a tiny model, while metrics and image I/O run.
    from mgm_i2i.cluster import metrics
    class FakeLPIPS(nn.Module):
        def __init__(self,**kwargs):super().__init__()
        def forward(self,a,b):return (a-b).square().mean((1,2,3),keepdim=True)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,**kwargs:self)
    monkeypatch.setattr(nn.Module,'cuda',lambda self:self)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(metrics,'seed_all',lambda seed:torch.manual_seed(seed))
    monkeypatch.setattr(metrics,'provenance',lambda:dict(gpu='CPU test fixture'))
    monkeypatch.setattr(torch.hub,'set_dir',lambda path:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda: 123)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda: 456)
    monkeypatch.setitem(sys.modules,'lpips',SimpleNamespace(LPIPS=FakeLPIPS))
    cfg=dict(method='i2sb_official',variant='i2sb_official_pretrained',seed=0,cpus=1,workers=0,
             scale=4,weights_root=str(tmp_path/'weights'),eval_crop=128,eval_batch_size=1)
    weight=tmp_path/'weights/hub/checkpoints/alexnet-owt-7be5be79.pth'
    weight.parent.mkdir(parents=True);weight.write_bytes(b'local')
    state=dict(complete=True,config=cfg,data_hash=None,step=None,successful_g=None,consumed=None,
               elapsed_training_seconds=None,training_stop='external_pretrained')
    class Model(nn.Module):
        def forward(self,*args):return args[0]
    class Bridge:
        def sample(self,model,c,nfe,**kwargs):return c
    monkeypatch.setattr(official,'load_for_evaluation',lambda *a:(state,Model(),Bridge(),dict(generator_parameters=official.PARAMETERS)))
    bank=[torch.full((3,128,128),100,dtype=torch.uint8) for _ in range(2)]
    monkeypatch.setattr(metrics,'evaluation_bank',lambda *a:(bank,[dict(image=f'{i}.png') for i in range(2)],'common','DIV2K',128))
    checkpoint=tmp_path/'ema.pt';checkpoint.write_bytes(b'fixture')
    output=tmp_path/'eval'
    result=metrics.evaluate(checkpoint,output,'val',1,4,evaluation_data_root=tmp_path,
                            evaluation_data_hash='common',expected_config=cfg)
    assert result['step'] is None and result['training_seconds'] is None and result['training_data_hash'] is None
    assert result['metrics']['model_lpips']==pytest.approx(0,abs=1e-12)
    assert result['generator_parameters']==official.PARAMETERS
    plan=dict(external_pretrained=True,train=[],eval=[dict(config=cfg,output=str(output),nfe=4)],evaluation_split='val')
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    script('cluster_collect').collect(path)
    report=(tmp_path/'val_summary.md').read_text()
    assert 'unknown, not zero' in report and 'I2SB trains from scratch' not in report


@pytest.mark.parametrize('seconds_per_nfe,complete',[(.001,True),(100,False)])
def test_probe_records_memory_and_blocks_insufficient_walltime(tmp_path,monkeypatch,seconds_per_nfe,complete):
    from mgm_i2i.cluster import engine,metrics
    worker=script('cluster_i2sb_official_run')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:123)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:456)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,**kwargs:self)
    monkeypatch.setattr(engine,'seed_all',lambda seed:None)
    monkeypatch.setattr(engine,'provenance',lambda:dict(gpu='CPU test fixture'))
    metadata=dict(official_bundle_sha256='bundle',precision='fp32',
                  inference_ms='must be replaced',generator_parameters=official.PARAMETERS)
    monkeypatch.setattr(official,'load_for_evaluation',lambda *a:({},None,None,metadata))
    bank=[torch.zeros(3,128,128,dtype=torch.uint8)]*250
    monkeypatch.setattr(metrics,'evaluation_bank',lambda *a:(bank,None,'common',None,128))
    monkeypatch.setattr(metrics,'predict_sr',lambda gen,c,*a,**k:torch.nn.functional.interpolate(c,scale_factor=4))
    ticks=iter([0,seconds_per_nfe,0,4*seconds_per_nfe])
    monkeypatch.setattr(worker.time,'perf_counter',lambda:next(ticks))
    item=dict(config=dict(seed=0,cpus=1,official_bundle_sha256='bundle'),checkpoint='unused',
              evaluation_data_root='unused',evaluation_data_hash='common')
    if complete:
        worker.probe(item,tmp_path,'val')
    else:
        with pytest.raises(RuntimeError,match='allocation is too short'):
            worker.probe(item,tmp_path,'val')
    result=json.loads((tmp_path/'probe.json').read_text())
    assert result['complete'] is complete and result['peak_allocated_bytes']==123
    assert result['official_bundle_sha256']=='bundle' and result['precision']=='fp32'
    assert result['inference_ms']=={'1':seconds_per_nfe*1000,'4':seconds_per_nfe*4000}
    assert result['generator_parameters']==official.PARAMETERS
