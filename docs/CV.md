# CV description

**Conditional Generative Models for Image Super-Resolution** — PyTorch, generative modeling, GANs, diffusion bridges, Slurm.

- Adapted Midpoint Generative Models to conditional ×4 super-resolution; implemented residual U-Nets, exact data consistency and a warm-started displacement field.
- Evaluated 10 MGM configurations across 3 seeds against regression, ESRGAN, I²SB-small and pretrained SinSR. Achieved LPIPS 0.139 at 9.13 ms/patch on V100, with ≈16× lower native latency than I²SB-small 16 NFE and 7.7% lower mean LPIPS, trading off PSNR/SSIM.

Project: https://github.com/kirillkatsuba/mgm_i2i
