"""Field transfer, freeze semantics, effective-batch math and new Slurm phase dependencies."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mgm_i2i.cluster.models import Generator, Field
from mgm_i2i.cluster.engine import mgm_objective
from mgm_i2i.cluster.mgm_training import initialize_field, accumulated_update, mgm_step, validate_field_checkpoint


def script(name):
    spec = importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('version',[1,2])
def test_field_transfer_uses_learned_features_and_remaps_untrained_latent_channels(version):
    gen,field = Generator(8,architecture_version=version),Field(8,architecture_version=version)
    with torch.no_grad():
        gen.net.stem.weight[:,:3].fill_(.2)
        gen.net.stem.weight[:,3:].fill_(.9)
        gen.net.mid[0].net[2].weight.fill_(.123)
        gen.net.out.weight.fill_(.4)
    before = copy.deepcopy(gen.state_dict())
    report = initialize_field(field,gen)
    torch.testing.assert_close(field.net.stem.weight[:,:6],torch.full_like(field.net.stem.weight[:,:6],.1))
    if version == 1:
        assert field.net.stem.weight[:,6:].count_nonzero() == 0
    assert field.net.out.weight.count_nonzero() == 0
    torch.testing.assert_close(field.net.mid[0].net[2].weight,gen.net.mid[0].net[2].weight)
    assert field.net.mid[0].net[2].weight.data_ptr() != gen.net.mid[0].net[2].weight.data_ptr()
    for key,value in before.items():
        torch.testing.assert_close(gen.state_dict()[key],value)
    assert report['strategy']=='warmup_features' and report['copied']
    if version == 2:
        assert field.net.mid[0].time_projection.weight.count_nonzero()==0
    with pytest.raises(ValueError,match='versions must match'):
        initialize_field(Field(8,architecture_version=3-version),gen)


def test_accumulation_matches_one_full_batch_adam_update():
    torch.manual_seed(42)
    full = nn.Linear(3,2)
    micro = copy.deepcopy(full)
    x,y = torch.randn(8,3),torch.randn(8,2)
    a = torch.optim.Adam(full.parameters(),lr=1e-5,betas=(0.,.999))
    b = torch.optim.Adam(micro.parameters(),lr=1e-5,betas=(0.,.999))
    loss = F.mse_loss(full(x),y)
    loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(full.parameters(),100.)
    a.step()
    def chunks():
        for xx,yy in zip(x.split(2),y.split(2)):
            yield F.mse_loss(micro(xx),yy),{}
    result = accumulated_update(chunks,b,micro.parameters(),torch.amp.GradScaler('cuda',enabled=False),
                                count=4,grad_clip=100.,amp=False)
    assert result['success'] and result['grad_norm']==pytest.approx(float(expected_norm),rel=1e-6)
    torch.testing.assert_close(result['values']['loss'],loss)
    for p,q in zip(full.parameters(),micro.parameters()):
        torch.testing.assert_close(p,q)
        assert b.state[q]['step']==1
        torch.testing.assert_close(a.state[p]['exp_avg'],b.state[q]['exp_avg'])


@pytest.mark.parametrize('version',[1,2])
def test_field_only_warmup_learns_but_does_not_modify_generator(version):
    torch.manual_seed(9)
    gen,field = Generator(8,architecture_version=version),Field(8,architecture_version=version)
    nn.init.normal_(gen.net.out.weight,std=.01)
    initialize_field(field,gen)
    before_g,before_f = copy.deepcopy(gen.state_dict()),copy.deepcopy(field.state_dict())
    cfg = dict(batch_size=1,grad_accumulation=2,scale=4,amp=False,grad_clip=10.,field_only=True,
               midpoint=False,antithetic=False,noise_g=0.,noise_f=.01,reconstruction_weight=0.)
    gopt,fopt = torch.optim.Adam(gen.parameters(),lr=1e-5),torch.optim.Adam(field.parameters(),lr=1e-5)
    scaler = torch.amp.GradScaler('cuda',enabled=False)
    real = torch.rand(2,3,32,32)*2-1
    for _ in range(3):
        result = mgm_step(gen,field,real,cfg,gopt,fopt,scaler,mgm_objective)
    assert result['generator'] is None and not gopt.state
    for key,value in before_g.items():
        torch.testing.assert_close(gen.state_dict()[key],value)
    assert not torch.equal(before_f['net.out.weight'],field.state_dict()['net.out.weight'])
    assert all(p.grad is None for p in gen.parameters())
    cfg['field_only']=False
    result = mgm_step(gen,field,real,cfg,gopt,fopt,scaler,mgm_objective)
    assert result['generator']['success'] and result['field']['success']
    assert any(not torch.equal(value,gen.state_dict()[key]) for key,value in before_g.items())
    assert all(not p.requires_grad for p in field.parameters())
    assert all(state['step']==1 for state in gopt.state.values())


def test_field_checkpoint_rejects_other_generator_or_policy():
    cfg = dict(seed=0,width=120,noise_f=.01,adam_betas=[0.,.999])
    state = dict(complete=True,config=dict(cfg,field_only=True),data_hash='data',init_hash='warmup',step=20,skipped_aux=0)
    validate_field_checkpoint(state,cfg,'data','warmup')
    with pytest.raises(ValueError,match='different generator'):
        validate_field_checkpoint(state,cfg,'data','other')
    with pytest.raises(ValueError,match='noise_f'):
        validate_field_checkpoint(state,dict(cfg,noise_f=.2),'data','warmup')


@pytest.mark.parametrize('beta1,batch,accumulation',[(None,16,16),(.8,32,1)])
def test_authors_plan_reuses_old_warmup_and_has_field_phase(tmp_path,monkeypatch,beta1,batch,accumulation):
    submit = script('cluster_submit')
    source,dest = tmp_path/'old',tmp_path/'new'
    monkeypatch.setattr(sys,'argv',['submit','--run-root',str(source),'--variants','mgm',
                                    '--data-protocol','random128-v3'])
    submit.main()
    argv = ['submit','--run-root',str(dest),'--variants','mgm_critic_noise','--data-protocol','random128-v3',
                                    '--reuse-warmup-from',str(source),'--mgm-optimizer','authors',
                                    '--field-initialization','warmup_features','--field-warmup-minutes','15',
            '--mgm-microbatch-size',str(batch),'--grad-accumulation',str(accumulation)]
    if beta1 is not None:
        argv += ['--mgm-beta1',str(beta1)]
    monkeypatch.setattr(sys,'argv',argv)
    submit.main()
    plan = json.loads((dest/'plan.json').read_text())
    assert [len(plan[k]) for k in ('warmup','field_warmup','train','eval')]==[0,3,3,3]
    assert plan['budget']['training_gpu_hours']==6.75
    for field,main in zip(plan['field_warmup'],plan['train']):
        assert field['config']['field_only'] and field['config']['seconds']==900
        assert field['init']==main['init']
        assert main['field_init']==str(Path(field['output'])/'latest.pt')
        c=main['config']
        assert c['lr_g']==c['lr_aux']==1e-5 and c['adam_betas']==[beta1 or 0.,.999]
        assert c['batch_size']==batch and c.get('grad_accumulation',1)==accumulation
        assert field['config']['adam_betas']==c['adam_betas']
        assert field['config']['batch_size']==batch
        assert field['config'].get('grad_accumulation',1)==accumulation
        assert c['optimizer_profile']==('authors' if beta1 is None else 'authors_beta1_override')
    old = json.loads((source/'plan.json').read_text())
    assert [item['config'] for item in plan['reused_warmup']]==[item['config'] for item in old['warmup']]
    commands = submit.commands(plan,dest/'plan.json')
    assert '--job-name=mgm_field_warmup' in commands[0]
    assert '--dependency=afterok:<field_warmup_job_id>' in commands[1]
    assert '--dependency=afterok:<train_job_id>' in commands[2]


@pytest.mark.parametrize('beta1',['-0.1','1','nan'])
def test_mgm_beta1_rejects_invalid_values_before_writing_plan(tmp_path,monkeypatch,beta1):
    submit = script('cluster_submit')
    monkeypatch.setattr(sys,'argv',['submit','--run-root',str(tmp_path),'--variants','mgm',
                                    '--mgm-beta1',beta1])
    with pytest.raises(ValueError,match='beta1 must be'):
        submit.main()
    assert not (tmp_path/'plan.json').exists()


def test_latent_sweep_reuses_exactly_the_same_latent():
    diagnostic=script('cluster_diagnose_warmup')
    class Model(nn.Module):
        def forward(self,c,z):
            return F.interpolate(c,scale_factor=4,mode='nearest')+z
    c,z=torch.randn(1,3,8,8),torch.randn(1,3,32,32)
    results=diagnostic.latent_outputs(Model(),c,z,[0.,.1,.5,1.])
    for alpha,result in zip([0.,.1,.5,1.],results):
        torch.testing.assert_close(result-results[0],alpha*z)
