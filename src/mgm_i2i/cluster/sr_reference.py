"""Author SinSR/VOSR networks and samplers on the common float pooling input.

No downloads here. Each model runs in its own process. The wrappers change I/O,
boundary handling and RNG ownership, not the learned architecture or sampler.
"""
import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F

from .official_i2sb import digest, _package

PINS = json.loads(Path(__file__).with_name('sr_reference_pins.json').read_text())
ASSETS = json.loads(Path(__file__).with_name('sr_reference_assets.json').read_text())
HF_REVISION = 'f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3'


def source_hash(root):
    root = Path(root)
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
             if p.is_file() and '.git' not in p.parts and p.suffix in ('.py', '.yaml', '.yml')}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def verify_sources(roots, method):
    for name in (['sinsr'] if method == 'sinsr' else ['vosr', 'dino']):
        if source_hash(roots[name]) != PINS[name]['source_tree_sha256']:
            raise ValueError(f'Pinned {name} source changed or missing')


def load_bundle(path, method=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('version') != 1 or manifest.get('method') not in ('sinsr', 'vosr'):
        raise ValueError('Unexpected SR reference bundle')
    if method is not None and manifest['method'] != method:
        raise ValueError('Bundle method mismatch')
    if manifest.get('hf_revision') != HF_REVISION:
        raise ValueError('Bundle model revision mismatch')
    specs = ASSETS[manifest['method']]
    if set(manifest['files']) != set(specs):
        raise ValueError('Reference bundle must include all original inference components')
    names = ['sinsr'] if manifest['method'] == 'sinsr' else ['vosr', 'dino']
    if manifest.get('source_pins') != {name: PINS[name] for name in names}:
        raise ValueError('Reference source pins mismatch')
    expected_checkpoint = 'SinSR_v1.pth' if manifest['method'] == 'sinsr' else 'VOSR_0.5B_os/checkpoints/ema_model.safetensors'
    if manifest.get('checkpoint') != expected_checkpoint:
        raise ValueError('Unexpected reference checkpoint')
    for name, sha in manifest['files'].items():
        local = Path(name)
        if (local.is_absolute() or '..' in local.parts or
                (path.parent/local).stat().st_size != specs[name]['size'] or digest(path.parent/local) != sha or
                (specs[name]['sha256'] and sha != specs[name]['sha256'])):
            raise ValueError(f'Invalid or changed reference asset: {name}')
    return manifest


def load_module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@contextmanager
def draw_rng(device, seed):
    """Fork the actual device RNG because upstream kernels use randn_like."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        else:
            torch.random.default_generator.manual_seed(seed)
        yield


def reflect_to_multiple(image, multiple=64):
    """Same right/bottom reflection convention as SinSR, with legal small pads.

    A single reflect pad of 32 on an input of 32 is illegal in PyTorch. Repeated
    legal pads (31 then 1) support our LR32. No real neighbouring pixels are used.
    """
    h, w = image.shape[-2:]
    bottom, right = (-h) % multiple, (-w) % multiple
    while bottom or right:
        dh, dw = min(bottom, image.shape[-2]-1), min(right, image.shape[-1]-1)
        if not (dh or dw):
            raise ValueError('Reflection requires spatial dimensions > 1')
        image = F.pad(image, (0, dw, 0, dh), mode='reflect')
        bottom -= dh
        right -= dw
    return image


def tensor_bicubic(image, scale=4):
    # PIL-style antialiasing, but keep the float LR rather than round to PNG.
    return F.interpolate((image+1)/2, scale_factor=scale, mode='bicubic',
                         align_corners=False, antialias=True).clamp(0, 1)*2-1


def strict_weights(model, path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in data:
        data = data['state_dict']
    state = {k.removeprefix('module.'): v for k, v in data.items()}
    model.load_state_dict(state, strict=True)


class SinSR(nn.Module):
    def __init__(self, root, assets, device):
        super().__init__()
        from omegaconf import OmegaConf
        root, assets = Path(root), Path(assets)
        _package('_mgm_sinsr_models', root/'models')
        _package('ldm', root/'ldm')
        cfg = OmegaConf.to_container(OmegaConf.load(root/'configs/SinSR.yaml'), resolve=True)
        self.net = importlib.import_module('_mgm_sinsr_models.unet').UNetModelSwin(**cfg['model']['params'])
        # This optional import ignores XFORMERS_DISABLED in the original code.
        # Select its own vanilla attention path even if the user's env has xformers.
        importlib.import_module('ldm.modules.diffusionmodules.model').XFORMERS_IS_AVAILBLE = False
        self.ae = importlib.import_module('ldm.models.autoencoder').VQModelTorch(**cfg['autoencoder']['params'])
        strict_weights(self.net, assets/'SinSR_v1.pth')
        strict_weights(self.ae, assets/'autoencoder_vq_f4.pth')
        params = dict(cfg['diffusion']['params'])
        params['timestep_respacing'] = 15  # Original schedule, one_step=True exits at its last timestep.
        self.diffusion = importlib.import_module('_mgm_sinsr_models.script_util').create_gaussian_diffusion(**params)
        self.to(device).float().eval().requires_grad_(False)

    def restore(self, c):
        h, w = c.shape[-2:]
        padded = reflect_to_multiple(c)
        value = self.diffusion.p_sample_loop(
            y=padded, model=self.net, first_stage_model=self.ae,
            clip_denoised=False, model_kwargs={'lq': padded}, progress=False,
            one_step=True, apply_decoder=True)
        return value[..., :h*4, :w*4].clamp(-1, 1)


def vosr_code(root):
    """Load original inference helpers without executing its CLI or hub loading."""
    root = Path(root)
    _package('_mgm_vosr_models', root/'models')
    dit = importlib.import_module('_mgm_vosr_models.lightningdit').LightningDiT
    decoder = importlib.import_module('_mgm_vosr_models.light_decoder').LightDecoder
    core = load_module('_mgm_vosr_core', root/'vosr.py')
    vae = load_module('_mgm_vosr_vae', root/'tiled_vae.py')
    from torchvision.transforms import Normalize
    import numpy as np
    scope = dict(torch=torch, F=F, Normalize=Normalize, np=np,
                 IMAGENET_DEFAULT_MEAN=(.485, .456, .406),
                 IMAGENET_DEFAULT_STD=(.229, .224, .225))
    tree = ast.parse((root/'inference_vosr_onestep.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                ('preprocess_raw_image', 'get_venc_features', 'wavelet_color_fix')]
    # Bind exactly the feature-extraction method defined inside load_dinov2.
    selected += [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'forward_with_features']
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(root/'inference_vosr_onestep.py'), 'exec'), scope)
    return dit, decoder, core.VOSR, vae, scope


def dit_arguments(args):
    return dict(input_size=args.resolution//8, patch_size=args.patch_size,
                in_channels=8, out_channels=4, hidden_size=args.dim, depth=args.depth,
                num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, z_dims=args.enc_dim,
                encdim_ratio=args.encdim_ratio, auxiliary_time_cond=getattr(args, 'auxiliary_time_cond', True),
                use_qknorm=args.use_qknorm, use_swiglu=args.use_swiglu, use_rope=args.use_rope,
                use_rmsnorm=args.use_rmsnorm, wo_shift=args.wo_shift,
                num_fused_layers=len(args.layer_dinov2b_list))


class VOSR(nn.Module):
    def __init__(self, root, dino_root, assets, device):
        super().__init__()
        from diffusers import AutoencoderKL
        from safetensors.torch import load_file
        assets = Path(assets)
        self.args = types.SimpleNamespace(**json.loads((assets/'VOSR_0.5B_os/args.json').read_text()),
                                         posterior_mode=True, vae_tile_size=0, vae_tile_overlap=0)
        if self.args.ae_type != 'sd2' or self.args.enc_type != 'dinov2b':
            raise ValueError('Expected author VOSR-0.5B-os with SD2 VAE / DINOv2-B')
        dit, decoder, core, self.vae_code, self.helpers = vosr_code(root)
        self.net = dit(**dit_arguments(self.args))
        # Unlike upstream strict=False, fail if architecture and weights disagree.
        self.net.load_state_dict(load_file(assets/'VOSR_0.5B_os/checkpoints/ema_model.safetensors'), strict=True)
        self.net.forward = self.net.forward_flexible
        self.ae = AutoencoderKL.from_pretrained(str(assets/'stable-diffusion-2-1-base/vae'),
                                               local_files_only=True, use_safetensors=True)
        checkpoint = torch.load(assets/'sd21_lwdecoder.pth', map_location='cpu', weights_only=True)
        decoder_config = checkpoint['config']
        self.decoder = decoder(in_channels=decoder_config['in_channels'],
                               out_channels=decoder_config['out_channels'],
                               block_out_channels=tuple(decoder_config['block_out_channels']),
                               layers_per_block=decoder_config['layers_per_block'])
        self.decoder.load_state_dict(checkpoint['model_state_dict'], strict=True)
        _package('dinov2', Path(dino_root)/'dinov2')
        self.dino = importlib.import_module('dinov2.hub.backbones').dinov2_vitb14(pretrained=False)
        strict_weights(self.dino, assets/'torch_cache/checkpoints/dinov2_vitb14_pretrain.pth')
        self.dino.head = nn.Identity()
        self.dino.forward_with_features = types.MethodType(self.helpers['forward_with_features'], self.dino)
        self.core = core(time_dist=self.args.time_dist, cfg_ratio=self.args.cfg_ratio,
                         cfg_scale=self.args.cfg_scale, interp_type=self.args.interp_type,
                         t_start=self.args.t_start, t_end=self.args.t_end,
                         accelerator=types.SimpleNamespace(device=torch.device(device)), args=self.args)
        self.to(device).float().eval().requires_grad_(False)

    def restore(self, c):
        from torchvision.transforms.functional import to_pil_image, to_tensor
        up = tensor_bicubic(c)
        latent, mean, std = self.vae_code.encode_dispatch(self.ae, up, self.args, c.device)
        features = self.helpers['get_venc_features'](self.dino, up, self.args)
        latent = self.core.sample_onestep(self.net, latent, venc_fea=features, n_steps=1)
        value = self.vae_code.decode_dispatch(self.ae, latent, self.args, mean, std, self.decoder)
        # Original default wavelet alignment, including original uint8 output.
        # Only postprocessing uses PIL; the pooling input to the network stays float.
        aligned = self.helpers['wavelet_color_fix'](
            to_pil_image((value[0].cpu()+1)/2), to_pil_image((up[0].cpu()+1)/2))
        return (to_tensor(aligned).unsqueeze(0).to(c.device)*2-1)


def load_for_evaluation(config, checkpoint, device='cuda'):
    method = config['reference_method']
    manifest = load_bundle(config['reference_bundle'], method)
    if digest(config['reference_bundle']) != config['reference_bundle_sha256']:
        raise ValueError('Reference bundle changed after planning')
    assets = Path(config['reference_bundle']).parent
    if Path(checkpoint).resolve() != (assets/manifest['checkpoint']).resolve():
        raise ValueError('Checkpoint differs from reference manifest')
    verify_sources(config['reference_sources'], method)
    if method == 'sinsr':
        model = SinSR(config['reference_sources']['sinsr'], assets, device)
        note = 'Author SinSR_v1 + VQ f4; legal repeated reflect-padding LR32 to64; crop SR256 to128'
    else:
        model = VOSR(config['reference_sources']['vosr'], config['reference_sources']['dino'], assets, device)
        note = ('Author VOSR-0.5B-os EMA + SD2 VAE + light decoder + DINOv2-B; '
                'float antialiased bicubic32to128 replaces uint8 PIL input resize; '
                'original flexible RoPE and wavelet color alignment with uint8 final output')
    counts = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
    unused = (sum(p.numel() for p in model.ae.decoder.parameters()) +
              sum(p.numel() for p in model.ae.post_quant_conv.parameters())) if method == 'vosr' else 0
    metadata = dict(complete=True, config=config, data_hash=None, step=None, successful_g=None,
                    consumed=None, elapsed_training_seconds=None, training_stop='external_pretrained')
    provenance = dict(reference_method=method, reference_bundle_sha256=config['reference_bundle_sha256'],
                      source_pins={k: PINS[k] for k in config['reference_sources']},
                      component_parameters=counts, inference_loaded_parameters=sum(counts.values()),
                      inference_active_parameters=sum(counts.values())-unused,
                      parameter_count_note='Loaded weights include unused original SD2 VAE decoder; active excludes it and its post_quant_conv' if unused else 'All loaded network components are used',
                      generator_parameters=counts['net'], external_pretraining=manifest['external_pretraining'],
                      inference_adapter=note, precision='fp32', sampling_draws='Fixed per-crop, no best-of-N')
    return metadata, model, None, provenance


@torch.no_grad()
def predict(model, c, offset=0, draw=0):
    outputs = []
    for index, image in enumerate(c.split(1)):
        with draw_rng(image.device, 20260101+(offset+index)*2+draw):
            result = model.restore(image)
        if result.shape != (1, 3, image.shape[-2]*4, image.shape[-1]*4) or not torch.isfinite(result).all():
            raise ValueError('Reference inference returned invalid shape or non-finite pixels')
        outputs.append(result)
    return torch.cat(outputs)
