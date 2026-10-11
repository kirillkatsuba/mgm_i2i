"""Download-only preparation. Training code never uses the network."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPERS = [
    ('flow-matching', '2210.02747v2'),
    ('stochastic-interpolants', '2303.08797v4'),
    ('zero-flow-encoders', '2602.00797v3'),
    ('pix2pix', '1611.07004v3'),
    ('perception-distortion', '1711.06077v4'),
    ('lpips', '1801.03924v2'),
    ('fid-ttur', '1706.08500'),
    ('edm', '2206.00364'),
    ('i2sb', '2302.05872'),
    ('sr3', '2104.07636'),
]
JOBS = [(f'materials/papers/{name}.pdf', f'https://arxiv.org/pdf/{ident}')
        for name, ident in PAPERS]
JOBS += [
    ('materials/code/edm.zip', 'https://codeload.github.com/NVlabs/edm/zip/refs/heads/main'),
    ('materials/code/sid.zip', 'https://codeload.github.com/mingyuanzhou/SiD/zip/refs/heads/main'),
    ('materials/code/pix2pix.zip', 'https://codeload.github.com/junyanz/pytorch-CycleGAN-and-pix2pix/zip/refs/heads/master'),
    ('materials/code/zero-flow.zip', 'https://codeload.github.com/probabilityFLOW/zfe/zip/refs/heads/main'),
    ('materials/code/cmgm.zip', 'https://codeload.github.com/daniil-shlenskii/cmgm/zip/refs/heads/main'),
    # An unchanged mirror; verified against the official archive's MD5.
    ('data/cifar-10-python.tar.gz', 'https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/main/cifar-10-python.tar.gz'),
    ('offline/torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth', 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'),
    ('offline/assets/katex.tar.gz', 'https://registry.npmjs.org/katex/-/katex-0.16.22.tgz'),
]


def fetch(job):
    relative, url = job
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(3):
        try:
            if not path.exists():
                temp = path.with_suffix(path.suffix + '.part')
                # curl uses the macOS trust store; the standalone Python on this
                # machine has no configured CA file. TLS verification stays on.
                subprocess.run(['curl', '-L', '--fail', '--silent', '--show-error',
                                '--connect-timeout', '30', '--max-time', '600',
                                url, '-o', str(temp)], check=True)
                temp.replace(path)
            if path.suffix == '.pdf' and not path.read_bytes().startswith(b'%PDF-'):
                path.unlink()
                raise ValueError('Response is not a PDF')
            if relative.startswith('data/'):
                md5 = hashlib.md5(path.read_bytes()).hexdigest()
                if md5 != 'c58f30108f718f92721af3b95e74349a':
                    path.unlink()
                    raise ValueError(f'CIFAR checksum mismatch: {md5}')
            if relative.endswith('alexnet-owt-7be5be79.pth'):
                if not hashlib.sha256(path.read_bytes()).hexdigest().startswith('7be5be79'):
                    path.unlink()
                    raise ValueError('AlexNet checksum mismatch')
            return {'path': relative, 'url': url, 'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'status': 'ok'}
        except Exception as exc:
            error = str(exc)
            time.sleep(attempt + 1)
    return {'path': relative, 'url': url, 'status': 'failed', 'error': error}


if __name__ == '__main__':
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, job) for job in JOBS]):
            result = future.result()
            results.append(result)
            print(result['status'], result['path'], result.get('bytes', result.get('error')), flush=True)
    (ROOT / 'offline/download-manifest.json').write_text(json.dumps(results, indent=2))
    if any(result['status'] != 'ok' for result in results):
        raise SystemExit('Some downloads failed; inspect offline/download-manifest.json')
