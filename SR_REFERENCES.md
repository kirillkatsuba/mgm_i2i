# Авторские SinSR и VOSR на наших pooling-парах

Подготовлены две независимые цепочки Slurm: **probe → eval → collect**.
Обучения нет. Каждая GPU-задача использует одну GPU; обе модели могут выполняться
параллельно. В запросах ресурсов нет `--mem` или `--mem-per-*`.

## Что именно сравниваем

- **SinSR_v1**, авторские веса, UNetModelSwin 118 593 583 параметра плюс VQ f4
  autoencoder 55 322 782: всего **173 916 365** загруженных параметров.
- **VOSR-0.5B-os**, одношаговый дистиллированный вариант из статьи, **не VOSR 2.0**.
  По авторскому args.json DiT имеет **486 936 384** параметра. Дополнительно нужны
  DINOv2-B, SD2 VAE и light decoder. Полные loaded/active counts записывает GPU probe;
  название 0.5B относится к DiT, а не всему пайплайну. Неиспользуемые decoder и
  post_quant_conv исходного SD2 VAE исключаются из active count, но входят в loaded.

Код: [SinSR](https://github.com/wyf0912/SinSR),
[VOSR](https://github.com/cswry/VOSR), [DINOv2](https://github.com/facebookresearch/dinov2).
Точные Git-коммиты и хеши исходников: `src/mgm_i2i/cluster/sr_reference_pins.json`.
Веса VOSR зафиксированы на ревизии Hugging Face
`f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3`, включая все auxiliary assets.
Для VOSR проверяются опубликованные SHA256 LFS. GitHub assets SinSR имеют фиксированные
URL и размеры; SHA256 записывается при первом скачивании и проверяется затем.

Это оценка переноса **готовых систем** на нашу задачу. Модели отличаются обучающими
данными, деградацией, разрешением и размером. Этот запуск не воспроизводит их
исходные benchmark-таблицы и не изолирует эффект MGM-loss.

## Общие данные и метрики

Используем существующий `data/div2k_multicrop128`, протокол
`random128-multicrop-v4`. По умолчанию **validation: 50 фотографий × 5 фиксированных
кропов = 250 HR128**. Координаты и manifest hash совпадают с MGM. LR32 вычисляется
из HR float через avg_pool2d ×4, без сохранения LR в PNG и без округления.
Каждая модель получает только этот LR. Никаких соседних настоящих пикселей за
границами кропа, HR-based color matching или выбора лучшего random draw нет.

Общий оценщик считает RGB PSNR/SSIM, LPIPS AlexNet v0.1, pooling consistency,
разнообразие двух фиксированных draws. Сначала среднее пяти кропов фотографии,
затем среднее фотографий. Метрики вычисляются на тензорах до нашего PNG-экспорта.
Встроенное uint8-постпреобразование VOSR сохраняется и раскрывается ниже.
Сохраняются нативный выход и дополнительный выход после общей pooling-проекции.
В основной таблице это авторские baseline-пайплайны; проекция — отдельная абляция.

Один авторский checkpoint на метод не является тремя training seeds. Время и шаги
его внешнего обучения в результатах — unknown/null, а не ноль. Seed=0 в плане —
метка запуска; фактический seed шума зависит от crop index и draw.

## Адаптация к 32→128

1. **SinSR:** sampler и сети авторские, 15 исходных timestep, `one_step=True`:
   один вызов сети на последнем timestep и авторское декодирование VQ.
   Авторский `reflect`-padding делает размеры LR кратными 64. На LR32 исходный
   единственный pad32 недопустим в PyTorch. Используем правое/нижнее отражающее
   дополнение в два допустимых шага (31, затем 1), получаем LR64 → SR256 и берём
   исходную область SR128. Внутренние пиксели LR не изменяются. Это явно раскрываемый
   адаптер границ; он увеличивает вычисления и может влиять на края результата.
2. **VOSR:** увеличение нашего float LR32 до128 через antialiased tensor bicubic;
   VAE encode → DINOv2 features → исходный `sample_onestep(n_steps=1)` → light decoder.
   Используется авторский `forward_flexible` с динамическим RoPE. Семантический
   энкодер сохраняет авторский resize до448. В отличие от CLI, входная интерполяция
   выполняется на float tensor вместо PIL uint8, чтобы не округлять общий LR.
   Остаётся авторская default wavelet color alignment по **тому же LR**, включая
   её uint8-выход. Она включена во время инференса. Дополнительного resize до512
   и обратно нет. Это явно адаптированный float-вход, не побитовая копия CLI.

FP32 для всех компонентов, batch1, без xformers и без torch.compile, для сопоставления
с прежним MGM на V100. Загруженные checkpoint-сети проверяются `strict=True`,
включая VOSR (его CLI допускает `strict=False`). Пропуск несовпавших весов запрещён.

Локально проверены оригинальные архитектуры на meta tensors и численный CPU forward
уменьшенного авторского flexible DiT. Проверены совпадение SinSR sampler с upstream,
границы, RNG и Slurm. **Большие pretrained weights здесь не скачивались; настоящий
численный forward обоих pretrained checkpoints и совместимость на V100 проверит
кластерный probe.** Не считаем результаты или VRAM заранее измеренными.

## 1. Обновить код и установить зависимости на login-узле

```bash
cd ~/mgm_i2i
git pull --ff-only
conda activate mgm_i2i
python -m pip install -r requirements-sr-reference.txt
```

Дополнительные зависимости рассчитаны на уже установленный torch2.6 / torchvision0.21
из нашего CLUSTER.md. Не устанавливать целиком авторские requirements: они задают
другие torch/CUDA и множество ненужных training-пакетов.

## 2. Скачать только нужные веса и код

```bash
python scripts/prepare_sr_references.py --download
```

Около **3.4 GB весов суммарно**, плюс код. Нужен интернет только здесь.
Загрузки пишутся в `.part`, поддерживают Range-resume, завершённые файлы проверяются
и используются повторно. Результат — два `manifest.json` в
`weights/sr_references/sinsr_v1` и `weights/sr_references/vosr_05b_os`.
Общий AlexNet уже должен существовать в `weights/hub/checkpoints`, как для прежних
метрик. Подготовка не скачивает train data, ScreenSR, teacher или большие T2I UNet.

Можно подготовить одну модель: `--models sinsr` или `--models vosr`.

## 3. Отправить две независимые цепочки

```bash
export CONDA_ENV=mgm_i2i
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

# Проверить план; ещё ничего не отправляется
bash slurm/launch_sr_references.sh

# Отправить обе модели (максимум две GPU одновременно)
bash slurm/launch_sr_references.sh --submit

squeue -u "$USER"
```

Корень по умолчанию `cluster_runs/sr_refs_multi5_val_r1`, отдельные подпапки
`sinsr` и `vosr`. Каждый probe использует одну GPU, два прогревочных и три измеряемых
прохода реальных кропов. Записывает форму/finite check, параметры, GPU, FP32 latency
и peak memory. Только успешный probe разрешает eval **своей** модели (`afterok`).
Eval считает все250кропов, два draws, общие метрики и картинки. Collect запускается
после eval (`afterany`) и сообщает фактическое покрытие, в том числе пропуски.
Ошибка одной цепочки не является зависимостью для другой.

Лимиты: **20 минут probe + 90 минут eval на модель**. Это запас по wall time,
не прогноз фактической длительности. Верхняя граница резервирования обеих цепочек
3ч40мин GPU-времени; фактическое время ожидаемо меньше и будет измерено.
Probe оценивает полную оценку по самому медленному из трёх проходов, двум draws и
числу кропов, добавляет коэффициент2 и10мин; если90мин недостаточно, eval блокируется.
Это оценка с запасом, а не гарантия времени. При OOM, несовпадении весов или другой
ошибке полный eval не запускается. Сначала смотреть лог probe, не запускать дубликаты.

Одна модель: `bash slurm/launch_sr_references.sh --models sinsr --submit`.
Другой лимит/корень:
`bash slurm/launch_sr_references.sh --run-root cluster_runs/sr_refs_multi5_val_r2 --eval-minutes 180 --submit`.
Повторный `--submit` существующего плана запрещён, чтобы не задвоить очередь.

## 4. Проверить и забрать результаты

```bash
cat cluster_runs/sr_refs_multi5_val_r1/sinsr/probe/probe.json
cat cluster_runs/sr_refs_multi5_val_r1/vosr/probe/probe.json

python scripts/cluster_collect.py --plan cluster_runs/sr_refs_multi5_val_r1/sinsr/plan.json
python scripts/cluster_collect.py --plan cluster_runs/sr_refs_multi5_val_r1/vosr/plan.json

tar --exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' \
  -czf sr_references_results.tar.gz -C cluster_runs sr_refs_multi5_val_r1
```

На своём компьютере:

```bash
rsync -avP charisma-hse:/home/kkatsuba/mgm_i2i/sr_references_results.tar.gz ~/Downloads/
```

В каждой подпапке: `probe/probe.json`, `eval/val_metrics.json`, per-patch/per-photo JSON,
`eval/samples/*.png`, `val_runs.csv`, `val_summary.md`, plan/submission и Slurm logs.
Coverage каждого summary должно быть1/1. В gallery слева направо: nearest LR,
bicubic, projected bicubic, HR, draw1, draw2, projected draw1.

Latency включает полный адаптер и все model-компоненты, исключает загрузку моделей,
метрики, экспорт файлов и дополнительную финальную проекцию. Сравнивать с MGM только
при одинаковой GPU, FP32 и batch1. Память из probe относится к loaded pipeline;
пиковая память полного eval дополнительно включает LPIPS. Нет смешения этих оценок.

## Отдельный финальный test после фиксации настроек

```bash
bash slurm/launch_sr_references.sh \
  --split test --run-root cluster_runs/sr_refs_multi5_test_r1 --submit
```

Это отдельная оценка; прежние результаты MGM относятся к validation. Чтобы строить
общую финальную test-таблицу, надо также оценить MGM на этом же test manifest.
