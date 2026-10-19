# Оценка авторского I²SB sr4x-pool

Выбран следующий эксперимент: **готовый официальный checkpoint**, без обучения и без изменений авторской ADM-архитектуры. Сравнение на прежних DIV2K validation-парах128. Это отдельный внешний baseline с ImageNet/ADM pretraining, не продолжение I²SB-small и не полное воспроизведение ImageNet256 benchmark.

## Что запускаем

- [NVlabs/I2SB](https://github.com/NVlabs/I2SB), commit `1ffdfaaf05495ef883ece2c1fe991b3049f814cc`, оригинальная552 807 171-параметрическая ADM-сеть.
- Официальный `sr4x-pool` checkpoint, **EMA**, без fine-tuning. Один checkpoint; три независимых training seeds из него не создаются.
- Наши250 фиксированных val-crop128 из50 фото, pooling×4, без растягивания до256. Сеть исходно обучена на ImageNet256; это отличие input distribution раскрывается отдельно. Формы128 и256 проверены на meta-тензорах.
- NFE1/4/16/64; FP32 batch1, два фиксированных draws. PSNR/SSIM/LPIPS считаются для первого, второй нужен для diversity. Агрегация сначала по5 кропам фото, потом по50 фото.
- Нативный sampler output и отдельно одна финальная pooling-проекция. Внутри цепочки дополнительной проекции нет.
- Результаты не перезаписывают MGM/small/baselines. Новый run root: `cluster_runs/i2sb_official_sr4x_pool_multi5_val_r1`.

## 1. Подготовка на login node

```bash
cd ~/mgm_i2i
git pull --ff-only
conda activate mgm_i2i
export CONDA_ENV=mgm_i2i
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

python -m pip install -r requirements-i2sb-official.txt
python scripts/prepare_i2sb_official.py --download
```

Скрипт получает pinned upstream в `third_party/I2SB-official`, проверяет SHA256 всех используемых source files, скачивает два файла из [официальной папки sr4x-pool](https://drive.google.com/drive/folders/1VIA8haw3Oy0OqL-CeEM2Rekk92tL0gD0):

- `latest.pt`, file ID `1-PEPa4RxzC3ulAqx6TjoCAM-5B6Zus4G`;
- `options.pkl`, file ID `1ZFM0Ebe1WeCrRQ9YbHOTQk8WXO3hmhjS`.

Файлы помещаются в `weights/i2sb_official_sr4x_pool/author_download`. Затем экспортируются **только оригинальные EMA shadow tensors**, сопоставленные original named_parameters order. Архитектура и формы net/EMA проверяются; повторного усреднения или обучения нет. Для чтения исходного checkpoint используется CPU mmap; net/optimizer не копируются на GPU. На GPU загружается только `official_ema.pt` (около2.06GiB FP32 tensors). Авторский training checkpoint может содержать и net, и EMA, и optimizer, поэтому скачать и сохранить потребуется заметно больше2GiB. Кроме исходника сохраняется ещё EMA-export; до завершения подготовки оба нужны.

Подготовка записывает `manifest.json`: hashes исходного checkpoint/options, EMA-export, source files, architecture, sampler schedule и исходные параметры. Повторная подготовка проверяет уже готовый EMA-export. Ни к ImageNet, ни к ADM-initialization checkpoint дополнительного обращения нет: готовые SR-веса уже содержат обученный denoiser. AlexNet для LPIPS берётся из прежнего `weights/hub/checkpoints`.

Авторский `options.pkl` проверен локально: `sr4x-pool`, image_size256,1000 intervals, beta_max0.3,t0=1e-4,T=1, cond_x1=false, ot_ode=false, clip_denoise=false. Содержимое читается restricted unpickler; загружаются именно эти conventions, не значения из нашего small baseline по умолчанию.

Если Google Drive выдаёт quota/access error, скрипт завершится до создания bundle manifest. Исходные два файла можно скачать из официальной папки другим доступным способом и положить в `author_download`, затем повторить команду без `--download`. Это не повод подменять модель случайной инициализацией.

## 2. Поставить оценку в Slurm

```bash
bash slurm/launch_i2sb_official.sh
bash slurm/launch_i2sb_official.sh --submit
squeue -u "$USER"
```

Dry run проверяет bundle, source hashes, общий manifest/val bank, LPIPS weights и создаёт подписанный неизменяемый план. Повторный submit того же плана блокируется через `submission.json`.

Цепочка:

1. `i2sb_ref_probe`: одна GPU, FP32 batch1, реальный crop128 из validation, warmup и NFE1/4. Проверка finite outputs/shape, actual peak allocated/reserved и времени. Лимит20 минут.
2. `i2sb_ref_eval`: массив4 задач NFE1/4/16/64, одна GPU на задачу, максимум2 одновременно, `afterok` probe. Лимит каждой180 минут. Через каждые25 кропов в stdout выводится прогресс.
3. `i2sb_ref_collect`: CPU job после завершения/ошибки eval, отчёт о coverage и метриках.

Host RAM параметрами Slurm не задаётся: нет `--mem`, `--mem-per-cpu`, `--mem-per-gpu`. Ни веса, ни source code с GPU-узла не скачиваются. Между NFE общая immutable модель; различается только sampler budget. Test автоматически не запускается.

Это первый реальный GPU inference этой сети в нашем окружении, **измеренного времени пока нет**. Значения для I²SB-small переносить на553M нельзя.180 минут — начальный запас на одну evaluation, а не обещание такого расхода. Верхняя сумма allocation caps четырёх eval+probe —12.33 GPU·часа; задачи освобождают GPU по завершении. Probe экстраполирует время полного самого длинного NFE по measured NFE4; если estimate×1.5+5 минут превышает лимит, он сохраняет `complete=false` и завершится ошибкой, не открывая зависимые eval tasks. Это оценка по одному crop, а не гарантия walltime.

Для другого лимита — прямой CLI до submit:

```bash
python scripts/cluster_i2sb_official_submit.py \
  --run-root cluster_runs/i2sb_official_sr4x_pool_multi5_val_r2 \
  --eval-minutes 240 --max-parallel 2 --submit
```

Нельзя менять уже подписанный план; при неудачном probe сначала проверить его JSON/log и состояние зависимых jobs. Не отправлять новую серию параллельно ещё работающей старой. Worker допускает повтор отдельной eval entry после устранения причины ошибки; пример нового отдельного job без переотправки всего массива:

```bash
sbatch --partition=rocky --gpus=1 --cpus-per-task=4 --time=240 --array=3 \
  --chdir="$PWD" \
  --output=cluster_runs/i2sb_official_sr4x_pool_multi5_val_r1/logs/retry-%A_%a.out \
  slurm/i2sb_official.sbatch \
  "$PWD/cluster_runs/i2sb_official_sr4x_pool_multi5_val_r1/plan.json" eval
```

Это пример повтора NFE64 (index3 в default grid), только когда прежний writer остановлен и есть успешный matching probe. Collector после такого ручного retry запускается вручную.

Переменные launcher: `I2SB_OFFICIAL_RUN_ROOT`, `I2SB_OFFICIAL_BUNDLE`, `I2SB_OFFICIAL_UPSTREAM`, `I2SB_OFFICIAL_MAX_PARALLEL`, `MGM_DATA_ROOT`, `MGM_WEIGHTS_ROOT`, `PYTHON_BIN`. Прямой CLI также поддерживает partition/account/constraint/gpu-type. Для сопоставления latency нужна та же V10032 и FP32; фактический GPU сохраняется в каждом отчёте.

## 3. Проверить и забрать результат

```bash
cat cluster_runs/i2sb_official_sr4x_pool_multi5_val_r1/submission.json
cat cluster_runs/i2sb_official_sr4x_pool_multi5_val_r1/probe/probe.json
python scripts/cluster_collect.py \
  --plan cluster_runs/i2sb_official_sr4x_pool_multi5_val_r1/plan.json
```

Ожидается evaluation coverage4/4, один внешний checkpoint на NFE. `step`, `successful_g`, training time и training data hash неизвестны и записываются null, а не фиктивными нулями. Переданный author options `num_itr=1000000` не принимается за доказательство фактической длительности обучения опубликованных весов. Обучение этой модели в нашей серии отсутствует.

В `<run>/eval_nfeK/`: `val_metrics.json`, `val_images.json`, `val_patches.json`, `val_samples.png`, `samples/` для всех250 пар. В summary отдельно native/+projection. Hash исходных весов, bundle, original schedule,552.81M parameters, pretraining, resolution note и фактический GPU сохраняются. Время включает GPU sampler со всеми NFE/noise, исключает loading/metrics/transfer/дополнительную final projection; CPU trajectories upstream отключены, итоговое состояние остаётся GPU tensor.

Упаковать:

```bash
tar --exclude='*.pt' -czf i2sb_official_results.tar.gz -C cluster_runs \
  i2sb_official_sr4x_pool_multi5_val_r1
```

На локальном компьютере:

```bash
rsync -avP charisma-hse:/home/kkatsuba/mgm_i2i/i2sb_official_results.tar.gz ~/Downloads/
```

## Что проверено локально

- Pinned official code скачан, его hashes проверены. Публичная folder listing доступна; скачан и прочитан реальный664-byte `options.pkl`. Большой `latest.pt` локально не скачивался.
- Meta-forward оригинальной сети128/256 и точный count552807171 прошли.
- На CPU adapter sampler **побитово совпадает** с оригинальным `Diffusion.ddpm_sampling` для NFE1/4/16/64, включая final posterior, noise и порядок шагов; возвращается то же финальное состояние без CPU trajectory copies. Фиксированный RNG восстанавливается между crops/draws.
- Проверены EMA shadow mapping/shape rejection, ограниченный parser options, hashes bundle, экспорт/повторное использование EMA на synthetic checkpoint, shared evaluator dispatch, отсутствие выдуманного training budget, probe timing gate, Slurm dependency/duplicate guards и отсутствие memory flags.
- Настоящие author weights/качество, CUDA memory/latency и Slurm submission локально не проверялись. Первое такое измерение — cluster probe, а не CPU/meta-тест.
- Весь проектный suite: **88 passed** (`python -m pytest tests -q`), включая13 новых проверок official adapter/orchestration. Bash syntax, Python compilation и whitespace check прошли.

Детали исходной системы и альтернативы обучения — [research/I2SB_OFFICIAL_BASELINE.md](research/I2SB_OFFICIAL_BASELINE.md).
