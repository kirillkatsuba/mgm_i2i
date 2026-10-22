# Reproduction

The repository contains implementations, planners and frozen result summaries. DIV2K, pretrained loss/reference weights and trained checkpoints must be obtained separately. No public trained-MGM checkpoint download is provided.

## Environment

The reported cluster runs used Python 3.11, PyTorch 2.6.0+cu118 and torchvision 0.21.0. On a compatible CUDA machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-cluster.txt
python -m pytest -q tests
```

## Prepare data

```bash
python scripts/cluster_prepare.py \
  --crop-mode multicrop --eval-crops-per-image 5 \
  --data-root data/div2k_multicrop128 --weights-root weights
```

This downloads official DIV2K HR archives and frozen AlexNet/VGG19 weights, saves full training photos, and creates fixed validation/test crop banks plus a checksummed manifest. Use `--archive-root` to reuse previously downloaded ZIPs. Training crops are sampled online; no fixed train-patch bank is used.

## Selected MGM recipe

Run on a Slurm login node with the Python environment accessible to GPU jobs:

```bash
python scripts/cluster_submit.py \
  --run-root cluster_runs/mgm_v2_no_noise \
  --data-root data/div2k_multicrop128 --weights-root weights \
  --data-protocol random128-multicrop-v4 \
  --train-crops-per-image 4 --eval-crops-per-image 5 \
  --architecture-version 2 --width 120 --variants mgm_no_noise \
  --seeds 0 1 2 --batch-size 16 \
  --warmup-epochs 50 --field-warmup-epochs 50 --train-epochs 80 \
  --field-initialization warmup_features \
  --mgm-optimizer authors --mgm-beta1 0.8 \
  --mgm-microbatch-size 32 --grad-accumulation 1 \
  --warmup-time-limit-minutes 60 --field-warmup-time-limit-minutes 90 \
  --train-time-limit-minutes 240 --eval-minutes 30 \
  --partition rocky --max-parallel 4
```

This is a dry run that writes the plan and prints sbatch commands. Repeat with `--submit` to schedule one GPU per model/seed through warmup, field-only, joint training and validation. Replace partition/account/constraint options for your cluster. Time limits are allocation bounds; training stops by epochs. Warmup batch 16 and main batch 32 are intentional. `--mgm-optimizer authors` selects learning-rate defaults, while explicit beta1=.8 overrides the paper's beta1=0.

For Conda job activation, export `CONDA_ENV` and `CONDA_SH` before submitting. For a venv, activate it and export `PYTHON_BIN="$VIRTUAL_ENV/bin/python"`. Batch jobs inherit these settings through `slurm/env.sh`.

## Baselines, sweep and held-out evaluation

- `bash slurm/launch_baselines.sh`: preview the MSE, MSE+perceptual, ESRGAN and I²SB-small plans; `--submit` schedules them. `BASELINE_MAX_PARALLEL` is a per-plan limit. Warmups/main are 50/80 epochs; I²SB trains 130 epochs, evaluated at 1/4/16/64 NFE.
- `bash slurm/launch_mgm.sh`: builds the original v1 matched-noise and v2 critic-only controls with 50/50/80 epochs. Their architecture-compatible MSE warmups are the sweep initializers.
- `bash slurm/launch_mgm_ablation.sh --suite paper --with-test`: previews the full five-regime sweep, reusing those warmups and original controls. Add `--retrain-controls` for fresh controls and `--max-parallel N` for the shared cap. Run only after the source plans/checkpoints exist; source roots can be overridden with `--source-v1/--source-v2`.

To evaluate a completed normal training plan on the fixed test bank:

```bash
python scripts/cluster_eval_submit.py \
  --source-runs cluster_runs/mgm_v2_no_noise \
  --run-root cluster_runs/mgm_v2_no_noise_test \
  --data-root data/div2k_multicrop128 --split test --max-parallel 4
```

Add `--submit` to run. Evaluate test after freezing validation decisions. Known test results should not become a new tuning target. References require separate pinned source/weight preparation; see `scripts/prepare_sr_references.py --help` and `slurm/launch_sr_references.sh`.

## Inspect the saved evidence

`docs/results/` can be read without CUDA or checkpoints. Regenerate the README plot with:

```bash
python docs/build_assets.py
```

The gallery uses the original archived PNG exports; its provenance lists exact source checksums and panel selections. Regenerate it with `python docs/build_assets.py --gallery` only when those local archives are available. The presentation PDF is a copy of the existing deck, not rebuilt by this workflow. Historical Russian research/setup notes remain in the repository and may describe older protocols; this page documents the reported multicrop experiment.
