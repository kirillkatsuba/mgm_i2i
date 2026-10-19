# Авторский I²SB: варианты следующего запуска

Проверено20 октября2026. Официальный [NVlabs/I2SB](https://github.com/NVlabs/I2SB), commit `1ffdfaaf05495ef883ece2c1fe991b3049f814cc`. Исходники скачаны в игнорируемый `third_party/I2SB-official`; большой checkpoint не скачивался, Slurm jobs локально не отправлялись. **Пользователь выбрал сначала готовый `sr4x-pool` checkpoint. Интеграция и команды подготовлены: [I2SB_OFFICIAL.md](../I2SB_OFFICIAL.md).** Ниже исходный аудит и варианты; состояние реализации уточнено в конце.

Текущий I²SB-small: наш time-conditioned U-Net120/240/480,17 289 963 параметра, training с нуля на DIV2K multicrop. Его результаты нельзя называть проверкой неизменённой авторской сети.

## Что означает авторская сеть

[network.py](https://github.com/NVlabs/I2SB/blob/1ffdfaaf05495ef883ece2c1fe991b3049f814cc/i2sb/network.py) использует ADM/guided-diffusion UNet. [ckpt_util.py](https://github.com/NVlabs/I2SB/blob/1ffdfaaf05495ef883ece2c1fe991b3049f814cc/i2sb/ckpt_util.py) задаёт image_size256, model_channels256,2 residual blocks, channel_mult(1,1,2,2,4,4), attention32/16/8,64 channels/head, scale-shift normalization, residual up/downsampling. Upstream сам удаляет sigma-head из исходного ADM checkpoint, оставляя3 канала.

На meta-тензорах построена именно эта конфигурация: **552 807 171 параметр**,2.059 GiB только FP32 weights, примерно31.97× наш I²SB-small. Forward shapes128→128 и256→256 проходят без изменения network config. Проверка форм не доказывает качество или точное воспроизведение исходного spatial/data distribution на128.

Повторить аудит:

```bash
.venv/bin/python research/inspect_i2sb_official_architecture.py
```

## Три разных эксперимента

| Вариант | Что сохраняется | Что отличается от авторского эксперимента |
|---|---|---|
| Готовый официальный `sr4x-pool` checkpoint, оценка на наших парах | Авторская архитектура, обученные веса, upstream sampler | Наш DIV2K crop128 вместо ImageNet256; другой домен/масштаб, другой evaluator |
| Авторская ADM-архитектура + ADM initialization, обучение на DIV2K | Backbone, bridge equations/sampler, исходный pretrained prior | Наши train crops, split и бюджет; возможные runtime compatibility changes |
| Полное воспроизведение | ImageNet256, авторская архитектура/init/data pipeline/recipe | Требует исходных данных и большого compute; отдельный benchmark от текущего DIV2K128 |

Полное воспроизведение не достигается одной заменой нашего `BridgeDenoiser` на большую сеть. Для практического baseline разумно начать с официального `sr4x-pool` checkpoint, проверить общий evaluator и затем отдельно решить вопрос fine-tuning на DIV2K.

## Авторские training settings по исходникам

В [train.py](https://github.com/NVlabs/I2SB/blob/1ffdfaaf05495ef883ece2c1fe991b3049f814cc/train.py): global batch256, microbatch2 с накоплением, lr5e-5, scheduler gamma0.99/1000 steps, EMA0.99, interval1000, beta_max0.3. `--num-itr=1000000` — CLI default, не доказательство фактического бюджета каждого опубликованного checkpoint. Training принудительно FP32 (`use_fp16=False`).

В [runner.py](https://github.com/NVlabs/I2SB/blob/1ffdfaaf05495ef883ece2c1fe991b3049f814cc/i2sb/runner.py) используется AdamW, normalized bridge MSE, EMA и reverse posterior sampler. Старый runner также зависит от ImageNet labels/ResNet50, сохраняет checkpoint каждые5000 iterations и вызывает дорогую built-in evaluation: напрямую подменить LMDB-путь нашим NPY-root нельзя. При одной GPU и global batch256/microbatch2 выполняются128 microbatches на optimizer step; это существенно медленнее одного batch32 forward/backward.

В [официальном README](https://github.com/NVlabs/I2SB) указаны ImageNet256, LMDB, обучение checkpoints на2 узлах×8 V10032 и default microbatch2 для32GB V100. Возможность entrypoint работать на одной GPU есть; то, что полный training в исходном окружении помещается, не гарантирует ту же память/скорость в нашем PyTorch2.6. Нужен реальный GPU probe. Обещать прежний batch32 или прежние64 минуты здесь нельзя.

## Порядок интеграции

1. Зафиксировать upstream commit и hashes скачанных weights/options. Официальный checkpoint `sr4x-pool` скачивается upstream-скриптом `scripts/download_checkpoint.sh sr4x-pool`; это ссылка на публичную Google Drive folder в исходниках. Доступность бинарных файлов и численная совместимость ещё не проверены.
2. Сделать adapter официального checkpoint/EMA/network/time conventions, сохранив sampler. Наш текущий evaluator жёстко строит `BridgeDenoiser`; официальный checkpoint напрямую он не поддерживает.
3. Smoke test на одной V100: batch1 inference на128/256, NFE1/4, затем реальные peak memory и latency. Meta shape-тест уже пройден; CUDA-часть не запускалась.
4. Оценка на тех же250 validation crop128, фиксированные draws, FP32 batch1, NFE1/4/16/64; native output основной, одна финальная проекция — дополнительная абляция. Не растягивать crop128 до256 молча. Для строгого авторского разрешения создать отдельную общую оценку256 для всех моделей, включая MGM, и не смешивать её числа с текущими128.
5. Обозначать результат как `I²SB-official-pretrained`, отдельно от `I²SB-small`. Раскрывать553M vs17M/16M и ImageNet/ADM prior. Это внешний сильный baseline с другими training resources, не matched-from-scratch эксперимент.
6. Если выбираем training на DIV2K, определить отдельный бюджет после probe, поддержку microbatch accumulation, checkpoint/resume и validation curve. Авторский global batch256 и наш прежний запрет накопления для MGM — разные рецепты; их не переносим друг на друга автоматически.

Обновление реализации: подготовлены pinned downloader/EMA export, official network/posterior adapter, общий evaluator, CPU parity tests и Slurm probe→NFE eval→collect. Публичная folder listing проверена, реальный `options.pkl` скачан/прочитан; большой checkpoint и GPU inference ещё не проверялись. Команды — [I2SB_OFFICIAL.md](../I2SB_OFFICIAL.md). `--mem`/`--mem-per-*` не добавлены; завершённые small/MGM run roots не меняются.
