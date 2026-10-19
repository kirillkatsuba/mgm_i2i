"""Pinned upstream ADM/bridge adapter; no pretrained downloads on GPU nodes."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import sys
import types

import numpy as np
import torch
from torch import nn

UPSTREAM_COMMIT = '1ffdfaaf05495ef883ece2c1fe991b3049f814cc'
FOLDER_ID = '1VIA8haw3Oy0OqL-CeEM2Rekk92tL0gD0'
FILES = {'latest.pt': '1-PEPa4RxzC3ulAqx6TjoCAM-5B6Zus4G',
         'options.pkl': '1ZFM0Ebe1WeCrRQ9YbHOTQk8WXO3hmhjS'}
PARAMETERS = 552807171
SOURCE_HASHES = {
    'guided_diffusion/unet.py': '810d25e0bee5bbc1f857dc9ee886894281cc01ff08f8f15ae22ae907fc17e2cf',
    'guided_diffusion/nn.py': '9ce8e3bdb75de6bcd6ed181c425f12b87d6c19fa17d38843c8549c4946e9ec8b',
    'guided_diffusion/fp16_util.py': '6b8cb4ef3f1fb9d6bceb6598bf3f0c6abd7a2a396a92e98ceff9a6daeff0d9aa',
    'guided_diffusion/logger.py': '9addbaa62864bd13030c49b3afa7f8359dc4432a539278205f9dbc89ff71898f',
    'i2sb/diffusion.py': '8bf4b8b3e1ad45eeedf2078154e3567b5c05ee8498204ee1f644d4d0937ca92b',
    'i2sb/util.py': '3f1d70c1a1deb39c5454043d273832abff478c8f897257a5ebae886c440844d2',
    'i2sb/runner.py': 'ed015624d1449d47b720e2ccdf8aec382311087001d34dd525f6538f8170cda9',
    'i2sb/ckpt_util.py': 'a051327af6552fdb4ee2915054f15c47bcc7e9508d150b7542be4d080acc0969',
}
ARCHITECTURE = dict(image_size=256, in_channels=3, model_channels=256, out_channels=3,
                    num_res_blocks=2, attention_resolutions=[8, 16, 32], dropout=0.,
                    channel_mult=[1, 1, 2, 2, 4, 4], num_classes=None,
                    use_checkpoint=False, use_fp16=False, num_heads=4,
                    num_head_channels=64, num_heads_upsample=-1,
                    use_scale_shift_norm=True, resblock_updown=True,
                    use_new_attention_order=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_upstream(root):
    root = Path(root).resolve()
    for name, expected in SOURCE_HASHES.items():
        if digest(root/name) != expected:
            raise ValueError(f'Official I2SB source changed: {name}')
    return root


def _package(name, root):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(root)]
        sys.modules[name] = module
    elif sys.modules[name].__path__ != [str(root)]:
        raise ValueError('A different upstream directory is already loaded')


def reference_code(root):
    """Load original kernels without unused wandb/ImageNet/debug dependencies.

    No diffusion/network equations are rewritten. Utility functions are compiled
    directly from the pinned original AST. Only an unused ipdb import is omitted.
    """
    root = verify_upstream(root)
    _package('_mgm_i2sb_guided', root/'guided_diffusion')
    net_class = importlib.import_module('_mgm_i2sb_guided.unet').UNetModel
    _package('_mgm_i2sb_core', root/'i2sb')
    helper_name = '_mgm_i2sb_core.util'
    if helper_name not in sys.modules:
        helper = types.ModuleType(helper_name)
        tree = ast.parse((root/'i2sb/util.py').read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in ('space_indices', 'unsqueeze_xdim')]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(root/'i2sb/util.py'), 'exec'), helper.__dict__)
        tree = ast.parse((root/'i2sb/runner.py').read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_beta_schedule']
        helper.__dict__.update(torch=torch, np=np)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(root/'i2sb/runner.py'), 'exec'), helper.__dict__)
        sys.modules[helper_name] = helper
    diffusion_name = '_mgm_i2sb_core.diffusion'
    if diffusion_name not in sys.modules:
        module = types.ModuleType(diffusion_name)
        module.__package__ = '_mgm_i2sb_core'
        tree = ast.parse((root/'i2sb/diffusion.py').read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'ipdb')]
        exec(compile(tree, str(root/'i2sb/diffusion.py'), 'exec'), module.__dict__)
        sys.modules[diffusion_name] = module
    return net_class, sys.modules[diffusion_name].Diffusion, sys.modules[helper_name].space_indices


class OptionsUnpickler(pickle.Unpickler):
    """The official options.pkl is an argparse.Namespace of basic values."""
    def find_class(self, module, name):
        import pathlib
        allowed = {('argparse', 'Namespace'): argparse.Namespace,
                   ('pathlib', 'PosixPath'): pathlib.PosixPath,
                   ('pathlib', 'PurePosixPath'): pathlib.PurePosixPath,
                   ('pathlib._local', 'PosixPath'): pathlib.PosixPath,
                   ('pathlib._local', 'PurePosixPath'): pathlib.PurePosixPath,
                   ('torch', 'device'): torch.device}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unexpected options class: {module}.{name}')
        return allowed[module, name]


def read_options(path):
    with Path(path).open('rb') as stream:
        result = OptionsUnpickler(stream).load()
    if not isinstance(result, argparse.Namespace):
        raise ValueError('Expected official argparse.Namespace options')
    return vars(result)


def validate_options(options):
    if options.get('corrupt') != 'sr4x-pool' or options.get('image_size') != 256:
        raise ValueError('Expected the original sr4x-pool / ImageNet256 checkpoint')
    if any(options.get(k, False) for k in ('cond_x1', 'add_x1_noise', 'ot_ode')):
        raise ValueError('This adapter expects unconditioned stochastic sr4x-pool')
    if options.get('interval') != 1000 or not np.isclose(options.get('beta_max', 0), .3):
        raise ValueError('Unexpected checkpoint diffusion schedule')
    return dict(interval=options['interval'], beta_max=options['beta_max'],
                t0=options['t0'], T=options['T'],
                clip_denoise=bool(options.get('clip_denoise', False)))


def ema_weights(model, checkpoint):
    """Map torch-ema shadow tensors by the original network parameter order."""
    shadows = checkpoint['ema']['shadow_params']
    parameters = list(model.named_parameters())
    if len(shadows) != len(parameters):
        raise ValueError('EMA parameter count differs from author architecture')
    # The official runner saves Image256Net.state_dict, which prefixes the ADM.
    net = {k.removeprefix('diffusion_model.'): v for k, v in checkpoint['net'].items()}
    expected = model.state_dict()
    if set(net) != set(expected):
        raise ValueError('Checkpoint keys differ from original ADM')
    for key in expected:
        if net[key].shape != expected[key].shape:
            raise ValueError(f'Checkpoint shape mismatch: {key}')
    result = dict(net)
    for (key, param), tensor in zip(parameters, shadows):
        if tensor.shape != param.shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid EMA tensor: {key}')
        result[key] = tensor
    return result


class OfficialDenoiser(nn.Module):
    def __init__(self, net, schedule):
        super().__init__()
        self.diffusion_model = net
        self.register_buffer('noise_levels', torch.linspace(schedule['t0'], schedule['T'], schedule['interval'])
                             * schedule['interval'], persistent=False)

    def forward(self, image, steps):
        return self.diffusion_model(image, self.noise_levels[steps].detach())


class OfficialBridge:
    def __init__(self, root, schedule, device):
        _, diffusion, self.space_indices = reference_code(root)
        intervals = schedule['interval']
        betas = sys.modules['_mgm_i2sb_core.util'].make_beta_schedule(
            n_timestep=intervals, linear_end=schedule['beta_max']/intervals)
        betas = np.concatenate([betas[:intervals//2], np.flip(betas[:intervals//2])])
        self.diffusion = diffusion(betas, device)
        self.intervals = intervals
        self.clip_denoise = schedule['clip_denoise']

    @torch.no_grad()
    def sample(self, model, degraded, nfe, *, generators):
        if not 0 < nfe < self.intervals or len(generators) != len(degraded):
            raise ValueError('Invalid official sampler NFE/generators')
        steps = self.space_indices(self.intervals, nfe+1)[::-1]
        device = degraded.device
        # Upstream returns a CPU trajectory. Keep only its final state on GPU
        # so disk/trajectory transfers do not enter just this method's timer.
        outputs = []
        for image, generator in zip(degraded.split(1), generators):
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                if device.type == 'cuda':
                    torch.cuda.default_generators[devices[0]].manual_seed(generator.initial_seed())
                else:
                    torch.random.default_generator.manual_seed(generator.initial_seed())
                current = image
                for previous, step in zip(steps[1:], steps[:-1]):
                    index = torch.full((1,), step, device=device, dtype=torch.long)
                    clean = current-self.diffusion.get_std_fwd(index, xdim=current.shape[1:])*model(current, index)
                    if self.clip_denoise:
                        clean = clean.clamp(-1, 1)
                    current = self.diffusion.p_posterior(previous, step, current, clean, ot_ode=False)
                outputs.append(current)
        return torch.cat(outputs)


def load_bundle(path, *, verify_weights=True):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest['upstream_commit'] != UPSTREAM_COMMIT or manifest['parameters'] != PARAMETERS:
        raise ValueError('Unexpected official bundle architecture/version')
    if manifest['architecture'] != ARCHITECTURE:
        raise ValueError('Official ADM architecture changed')
    for key in ('ema_file',):
        name = Path(manifest[key])
        if name.is_absolute() or len(name.parts) != 1:
            raise ValueError('Bundle weights must be a local filename')
    weights = path.parent/manifest['ema_file']
    if verify_weights and digest(weights) != manifest['ema_sha256']:
        raise ValueError('Official EMA weights checksum mismatch')
    return manifest, weights


def load_for_evaluation(config, checkpoint, device='cuda'):
    manifest, weights = load_bundle(config['official_bundle'])
    if Path(checkpoint).resolve() != weights.resolve():
        raise ValueError('Plan checkpoint differs from official bundle')
    if digest(config['official_bundle']) != config['official_bundle_sha256']:
        raise ValueError('Official bundle changed after planning')
    cls, _, _ = reference_code(config['official_upstream'])
    with torch.device('meta'):
        net = cls(**manifest['architecture'])
    state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    net.load_state_dict(state, strict=True, assign=True)
    gen = OfficialDenoiser(net, manifest['schedule']).to(device).eval().requires_grad_(False)
    bridge = OfficialBridge(config['official_upstream'], manifest['schedule'], device)
    metadata = dict(complete=True, config=config, data_hash=None, step=None, successful_g=None,
                    consumed=None, elapsed_training_seconds=None, training_stop='external_pretrained')
    provenance = dict(upstream_commit=UPSTREAM_COMMIT, generator_parameters=PARAMETERS,
                      external_pretraining='Official I2SB sr4x-pool, ImageNet256 + ADM initialization',
                      source_checkpoint_sha256=manifest['source_checkpoint_sha256'],
                      official_bundle_sha256=config['official_bundle_sha256'],
                      author_schedule=manifest['schedule'], training_resolution=256,
                      resolution_note='Original ADM architecture evaluated on common crop128; no resize',
                      sampler_adapter='Upstream posterior/spacing; fixed per-crop RNG; no CPU trajectory storage')
    return metadata, gen, bridge, provenance
