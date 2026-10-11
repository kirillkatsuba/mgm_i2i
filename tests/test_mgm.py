import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from mgm_i2i.losses import observation, variational_loss
from mgm_i2i.models import ImageGenerator, ImageField, project_average
from mgm_i2i.evaluate import ssim


def test_flip_preserves_observation_under_endpoint_exchange():
    torch.manual_seed(1)
    a,b,noise=[torch.randn(16,2) for _ in range(3)]
    t=torch.rand(16,1)*.5
    branch=torch.rand(16,1)>.5
    assert torch.allclose(observation(a,b,t,branch,noise),
                          observation(b,a,t,~branch,noise))


def test_exact_discrete_equal_distributions_have_zero_conditional_midpoint_field():
    ends=torch.cartesian_prod(torch.tensor([-1.,1.]),torch.tensor([-1.,1.]))
    midpoint=ends.mean(1)
    delta=ends[:,1]-ends[:,0]
    for value in midpoint.unique():
        assert delta[midpoint==value].mean()==0
    assert delta.square().mean()>0  # Regression error need not vanish.


def test_generator_gradient_keeps_field_input_derivative():
    x=torch.tensor([[.3,-.2]],dtype=torch.float64,requires_grad=True)
    real=torch.tensor([[.7,.4]],dtype=torch.float64)
    jac=torch.tensor([[.4,.2],[-.1,.6]],dtype=torch.float64)
    a=.7
    mixed=a*x+(1-a)*real
    pred=mixed@jac.T
    loss=variational_loss(pred,real-x)
    actual=torch.autograd.grad(loss,x)[0]
    expected=(-2*pred + 2*a*(real-x-pred)@jac)/2
    assert torch.allclose(actual,expected)
    assert not torch.allclose(actual,-pred)  # Detaching f would lose a term.


def test_variational_lower_bound_and_maximizer():
    means=torch.tensor([[1.,2.],[-.5,1.]])
    noisy_targets=torch.cat([means+.7,means-.7],0)
    f=means.repeat(2,1)
    assert torch.allclose(variational_loss(f,noisy_targets),means.square().mean())
    wrong=f+.4
    assert variational_loss(wrong,noisy_targets)<variational_loss(f,noisy_targets)


def test_image_projection_and_shapes():
    torch.manual_seed(0)
    c=torch.randn(2,3,16,16)
    z=torch.randn(2,1,32,32)
    g=ImageGenerator(width=8,blocks=1,project=True)
    output=g(c,z)
    assert output.shape==(2,3,32,32)
    assert torch.allclose(torch.nn.functional.avg_pool2d(output,2),c,atol=1e-6)
    f=ImageField(width=8,blocks=1)
    assert f(c,torch.rand(2,1,1,1),output).shape==output.shape


def test_ssim_identity():
    torch.manual_seed(2)
    x=torch.rand(2,3,32,32)
    assert torch.allclose(ssim(x,x),torch.ones(2),atol=1e-5)
