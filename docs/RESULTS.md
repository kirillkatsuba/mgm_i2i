# Evaluation and results

## Protocol

- DIV2K: official 800 training photos; its 100 validation photos split deterministically into 50 project-validation and 50 project-test photos, seed 123.
- Training: four newly sampled 128×128 crops per photo/epoch, 3,200 patches, without flips/rotations. LR 32 is exact float average pooling ×4 after cropping.
- Each evaluation split: 50 photos ×5 fixed crops. Compute each metric per crop, average within photos, then across photos. Local models average three training seeds. SinSR uses one fixed author checkpoint.
- PSNR/SSIM: RGB, clipped output, no border shave. LPIPS: AlexNet v0.1. One fixed quality draw, no best-of-N. Final EMA checkpoints after fixed epoch budgets, not best-epoch selection.
- FP32 inference, batch 1, V100-SXM2-32GB, synchronized GPU timing, first batch excluded. Model loading, transfer, metrics and image export are excluded.

## What the evidence supports

Primary recipe: **MGM v2/no-interpolant-noise**, selected by validation LPIPS before interpreting test. v1 has a slightly lower test LPIPS, but test does not change recipe selection.

| Comparison with MGM v2 | Mean LPIPS difference, MGM − baseline | Paired 95% photo interval | Native latency ratio, baseline/MGM |
|---|---:|---:|---:|
| MSE + perceptual |−0.01344|[−0.02834,+0.00071]|≈1.05×|
| ESRGAN + P |+0.00192|[−0.00590,+0.00981]|≈4.14×|
| I²SB-small 16 NFE + P |−0.01160|[−0.02148,−0.00192]|≈16.09×|
| I²SB-small 64 NFE + P |−0.00376|[−0.01158,+0.00395]|≈64.07×|
| SinSR + P |−0.00693|[−0.01831,+0.00388]|≈15.36×|

MGM has lower mean LPIPS but lower PSNR/SSIM than the projected bridge. Projected ESRGAN has higher mean PSNR and slightly lower mean LPIPS than MGM. Intervals crossing zero do not establish superiority or equivalence. Native SinSR has LPIPS 0.16851, but the common-projection value is 0.14600; its transfer degradation and compute differ.

Intervals use 20,000 paired bootstrap resamples of 50 source photos after averaging crops and local seeds. Additional seed/photo intervals are included in CSVs. These analyses are exploratory, unadjusted for multiple comparisons, and conditional on available checkpoints/draws. Three seeds do not capture full training uncertainty.

## Baseline scope

MSE and MSE+VGG use the same projected U-Net as MGM v2, with zero latent. ESRGAN uses 23-block RRDBNet and our training recipe. I²SB-small is a 17.29M time-conditioned U-Net implementation, not the authors' ADM model. SinSR uses author SinSR_v1/VQ-f4 weights and external pretraining; its LR 32 input is padded to 64, reconstructed at 256 and cropped to 128. Both SinSR and MGM are one-step models.

The same final projection is reported for all methods; it is already inside MSE/perceptual/MGM. Added projection time for other baselines was not measured. Parameter count and latency do not isolate the effect of a training objective.

## Training diagnostics

The added sweep completed 48 field/joint phases and 60 validation/test evaluations. No G updates were skipped; two field updates were skipped. One v1 critic-only seed produced poor quality and was retained. Selected v2's logged joint objective fell 57.4%, while reconstruction MSE rose 2.9%; no intermediate validation curves establish quality convergence. Peak allocated training memory: 18.25 GiB.

Audited training subtotal: 89.68 GPU-hours on the HSE University supercomputer, excluding evaluation, earlier/development/failed attempts and external pretraining. Checkpoints/float banks remain outside Git. Saved hashes, configurations, metric aggregates and exported input pixels were checked; missing Git provenance in many jobs prevents a universal same-code claim.

## Downloadable evidence

- [All native/projected test means and seed SDs](results/test_metrics.csv): 18 configurations ×2 outputs.
- [Noise sweep, validation and test](results/noise_ablation.csv): 10 settings ×2 splits.
- [Paired comparisons with learned baselines](results/paired_baselines.csv).
- [Paired SinSR comparisons](results/paired_sinsr.csv).
- [Validation selection](results/validation_choice.json) and [provenance](results/provenance.json).

Visual examples show the first four test photos, crop 0, local seed 0/draw 1. They are illustrations rather than evidence of distributional realism. All images come from saved model exports, without generated or retouched content. VOSR has no accepted test result in this comparison.
