# Сравнение SR ×4 на Slurm

**Актуально19 октября:** команды новых multicrop-данных,50/50/80 эпох и обучения — [MULTICROP_PROTOCOL.md](MULTICROP_PROTOCOL.md). Новый default CLI — `random128-multicrop-v4`. Ниже сохранены окружение и исторические команды `random128-v3`; для их воспроизведения явно задавай `--crop-mode random` при подготовке и `--data-protocol random128-v3 --data-root data/div2k_random128` в submit. Старые данные не перезаписывать.

**Актуальный MGM-рецепт:** [field warmup, latent diagnostic, author Adam и V100 batch probe](MGM_FIELD_WARMUP_DIAGNOSTICS.md). Он обновляет команды новых MGM-обучений ниже: сначала диагностика, затем запуск в новых run roots с явно заданными настройками.

**Новый этап (18 октября):** команды для wide v1 120/240/480, v2 critic-only и BSD100 всех готовых checkpoints — [EXPERIMENTS_V1_V2_BSD100.md](EXPERIMENTS_V1_V2_BSD100.md). При одновременной постановке нескольких планов общий лимит GPU равен сумме их max-parallel; в новой инструкции выделено 2+1+1 GPU.

Подготовлен запуск **MSE, MSE+perceptual, ESRGAN, MGM v2 и I²SB U-Net-120**. Одна Slurm-задача — одна модель и один seed на одной GPU. По умолчанию одновременно работают до **4 GPU-задач**; лимит меняется через `--max-parallel`. Это независимые запуски, без DDP и без распределения одной модели по нескольким GPU.

Код проверен офлайн на CPU. Реальные данные нового протокола, обучение на GPU и постановка в Slurm в этой сессии не выполнялись. Перед длинной серией нужен пилот ниже: он проверит CUDA/AMP, VRAM, локальные веса, checkpoints и evaluator. Для выбранных сетей ориентируемся на V100 32GB; фактическое потребление памяти ещё не измерено. Для сравнения времени выбирай одинаковый тип GPU для всех задач.

## Данные и модели

Общий `random128-v3`: 800 исходных DIV2K train-фото, онлайн равновероятный выбор фото и случайного HR crop128. Без resize, поворотов и отражений. LR32 = `avg_pool2d(HR,4)`. Validation/test — фиксированные center128 из 50/50 официальных validation-фото, split seed=123. Разделение выполняется по фотографиям. Flickr2K не добавлен.

Полные train RGB сохраняются в отдельных `train_images/NNNN.npy`, loader использует mmap. В RAM копируется выбранный патч; PNG не декодируется на каждом шаге. Фото и координаты зависят от seed и глобального sample index; последовательность сохраняется при prefetch/resume. Методы с одинаковым seed имеют одинаковую последовательность примеров по sample index, но при равном времени успевают обработать разное их число.

| Метод | Обучение | Параметры inference / trainable |
|---|---|---|
| `mse` | MSE; G при Z=0 | 16.02 / 16.02 млн |
| `mse_perceptual` | MSE + VGG19 conv5_4 до ReLU, L1 features | 16.02 / 16.02 млн |
| `esrgan` | RRDB23 L1 pretraining → pixel L1 + VGG + relativistic logistic GAN | 16.70 / 31.20 млн |
| `mgm` | G и условное поле v2; согласованный stochastic interpolant | 16.02 / 33.32 млн |
| `i2sb` | Bridge objective с нуля, U-Net-120, без pretrained ADM | 17.29 / 17.29 млн |

Наш U-Net имеет ширины 120/240/480, линейный residual без tanh; field получает sinusoidal time embedding. G MGM/MSE имеет hard pooling projection. ESRGAN работает в [0,1], остальные — в [-1,1]; evaluator приводит outputs к общему диапазону. ESRGAN/I²SB оцениваются с native output и с той же финальной pooling projection. Внутри reverse chain проекция не добавляется. Это адаптации на нашей pooling-задаче, а не воспроизведение исходных bicubic/ImageNet-256 метрик.

Инициализация: MSE/MSE+perceptual/MGM используют общие обычные веса MSE warmup для своего seed и свежие optimizer states. ESRGAN получает собственный RRDB L1 warmup. I²SB обучается с нуля. Внешние SR/diffusion/GAN checkpoints не используются; ImageNet VGG19 для loss и AlexNet для LPIPS раскрываются как frozen feature networks.

## Установка окружения на кластере

```bash
cd /path/to/mgm_i2i
conda create -n mgm_i2i python=3.11 -y
conda activate mgm_i2i
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-cluster.txt
export CONDA_ENV=mgm_i2i
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
```

CUDA wheels должны соответствовать драйверу GPU-узла. Можно использовать существующий Python: экспортировать абсолютный `PYTHON_BIN` вместо `CONDA_ENV`. Mac `.venv` и offline wheels не переносимы на Linux. Все data/run/weights paths должны быть доступны с GPU-узлов; скрипт задаёт checkout как `--chdir`.

## Подготовка данных и frozen weights

Выполнять один раз на узле с интернетом, где разрешена подготовка данных:

```bash
python scripts/cluster_prepare.py --crop-mode random --data-root data/div2k_random128 --weights-root weights
python -m pytest -q tests/test_cluster.py tests/test_cluster_baselines.py tests/test_baseline_preprocessing.py tests/test_mgm.py
```

Если оригинальные HR ZIP уже есть, повторно не скачивай:

```bash
python scripts/cluster_prepare.py --crop-mode random --data-root data/div2k_random128 \
  --archive-root data/div2k/downloads --weights-root weights
```

Подготовка скачивает официальные DIV2K HR train/valid ZIP, AlexNet и VGG19 (`vgg19-dcbb9e9d.pth`). Последний файл содержит стандартный checkpoint, но loss загружает только feature layers, без classifier. На GPU-узлах сеть не нужна: модели не скачивают веса автоматически. `--skip-vgg` допустим только для серии без ESRGAN/MSE+perceptual. Submit проверяет наличие/хеши нужных весов и всех prepared файлов до вызова sbatch.

Нужно место для архивов, полных uint8 NPY train images (примерно `3 × сумма(H×W)` байт плюс заголовки), frozen weights и checkpoints. Для ширины120 checkpoint с optimizer states занимает существенно больше места, чем прежняя компактная сеть; не используй прежнюю оценку объёма center128-bank. В manifest сохраняются split, координаты eval crop, источники и SHA файлов. Повторная подготовка проверяет данные; другая подготовка в тот же root запрещена.

Для прежнего all-center режима есть `--crop-mode center --data-root data/div2k_center128`, для v1 — `--crop-mode legacy --data-root data/div2k`. Старые планы/checkpoints читаются с исходными width, архитектурой и data protocol. Новые планы v4 требуют `random128-v3`, новый run root и warmup.

## Пилот всех пяти методов

```bash
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 --run-root cluster_runs/pilot_sr_v4 --pilot --max-parallel 4
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 --run-root cluster_runs/pilot_sr_v4 --pilot --max-parallel 4 --submit
```

Первая команда только пишет неизменяемый `plan.json`, SHA и печатает команды. Вторая ставит jobs. Пилот: один seed, MSE/RRDB warmup по 1 минуте, четыре main метода по 2 минуты, I²SB 3 минуты; eval I²SB NFE=1/4. Это 13 GPU-минут training плюс evaluation/загрузка/сохранение, не эксперимент для вывода о качестве.

Если GRES кластера различает GPU, добавь одинаковые аргументы к обеим командам, например `--gpu-type v100` или подходящий `--constraint`. Точное имя типа зависит от конфигурации Slurm. По умолчанию запрашиваются1 GPU и4 CPU. Параметры host RAM не передаются: используются настройки Slurm-кластера.

При OOM выбери общий `--batch-size 8` в **новом** pilot run root, затем используй его для всей серии. Engine поддерживает MGM gradient accumulation; согласованный новый MGM-рецепт использует batch32 без накопления. Изменение batch нужно раскрывать. `--fp32` отключает AMP для отдельного диагностического плана. Перед main проверь `complete: true`, успешные G updates, отсутствие массового FP16 skipping, `peak_allocated_bytes_this_session`, все metrics и PNG. Test при пилоте не используется.

## Основная серия и параллельность

```bash
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 --run-root cluster_runs/sr_v4 --max-parallel 4
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 --run-root cluster_runs/sr_v4 --max-parallel 4 --submit
```

Defaults: три seed (0/1/2), пять методов, train batch16, width120, FP16 AMP. Evaluation — FP32, batch1 для сопоставимой latency. I²SB: 1000 intervals, beta_max=.3, symmetric squared-linear schedule, AdamW 5e-5/EMA .99, StepLR(.99/1000 successful updates); NFE grid 1/4/16/64. MSE+perceptual weight=.01; ESRGAN weights pixel=.01, perceptual=1, GAN=.005, Adam1e-4. Это начальные настройки для validation-пилотов, не доказанно оптимальные. ESRGAN/RRDB в этой ограниченной по времени адаптации используют постоянный LR; upstream iteration milestones не перенесены механически.

План состоит из зависимых массивов:

1. **warmup** — 6 задач (MSE и RRDB × 3 seed), каждая 2 часа training.
2. **train** — 15 задач. MSE/MSE+perceptual/ESRGAN/MGM по 2 часа после warmup; I²SB по 4 часа с нуля, чтобы полный budget метода/seed был также 4 часа.
3. **eval** — 24 задачи: четыре one-pass метода × 3 seed и I²SB × 4 NFE × 3 seed, отдельный лимит 30 минут на задачу.
4. **collect** — CPU job, сводка validation после завершения eval.

`--array=…%4` ограничивает каждый GPU-массив четырьмя задачами. Фазы имеют `afterok` barrier, поэтому между ними нет перекрытия и суммарно эта серия использует не более 4 GPU одновременно. Выделение фактического количества GPU зависит от очереди и квот. Даже I²SB ждёт warmup barrier для простого общего ограничения ресурсов. Неудачная фаза блокирует следующую; автоматического продолжения незавершённого обучения нет.

Физический training budget основной серии — **48 GPU-часов**, с повторным использованием общих MSE warmup. Evaluation добавляет фактическое время отдельно. Поле `allocation_gpu_hours_upper_bound` — сумма всех time limits, не ожидаемая стоимость: main array получает общий лимит 250 минут из-за четырёхчасового I²SB, но двухчасовые методы завершаются и освобождают GPU раньше. Параллельность сокращает календарное время, не GPU-часы; phase barriers и очередь не позволяют обещать ровно четырёхкратное ускорение.

Для другого лимита используй, например, `--max-parallel 3` в обеих командах. Аргументы plan creation и submit должны совпадать. Повторный submit того же плана блокируется `submission.json`, в том числе при частичной ошибке постановки. Реальные job IDs записываются туда по мере submission. По умолчанию дополнительных абляций нет; можно явно задать `--variants mse mse_perceptual esrgan mgm i2sb mgm_antithetic`.

## MGM v2: critic-only noise после серии sr_4c20214

Следующая абляция сохраняет архитектуру v2 120/240/480 и весь протокол основной серии. Вариант `mgm_critic_noise` меняет только `noise_g: 0.01 → 0`; `noise_f=0.01`, random flip, интегрирование по t, LR projection, optimizer, batch16 и бюджет main120 минут остаются прежними. Gaussian latent Z остаётся случайным: выключается только шум интерполяции на шаге G. Это изменение режима обучения, число параметров и inference не меняются.

Чтобы начать с **тех же warmup-весов**, на кластере после получения изменений кода выполни сначала dry run:

```bash
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 \
  --run-root cluster_runs/sr_v2_critic_noise \
  --variants mgm_critic_noise \
  --seeds 0 1 2 \
  --reuse-warmup-from cluster_runs/sr_4c20214 \
  --max-parallel 3
```

Проверь в плане `noise_g=0`, `noise_f=0.01`, `architecture_version=2`, `width=120` и init paths в `sr_4c20214/warmup/seedN/latest.pt`. Затем поставь задачи:

```bash
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 \
  --run-root cluster_runs/sr_v2_critic_noise \
  --variants mgm_critic_noise \
  --seeds 0 1 2 \
  --reuse-warmup-from cluster_runs/sr_4c20214 \
  --max-parallel 3 \
  --submit
```

Нужны исходные `plan.json`, `plan.sha256` и три warmup `latest.pt` на кластере. Архив результатов без checkpoints для запуска недостаточен. Новый план проверяет подпись исходного плана и точное совпадение warmup config; до submit проверяет наличие checkpoint-файлов. Training engine проверяет завершённость warmup, manifest hash и архитектуру по содержимому checkpoint, записывает фактический `init_hash`. Эти hashes должны совпасть с исходной MGM для каждого seed; сравнение относится только к одинаковым данным и GPU.

В очереди будут **3 train-задачи → 3 validation eval-задачи → collect**, без новых warmup. Дополнительный training budget — **6 GPU-часов**, плюс eval. В `reused_warmup_training_gpu_hours` отдельно записаны прежние 6 часов shared warmup; полный бюджет одного метода остаётся 2 часа warmup + 2 часа main на seed. Это новый запуск от warmup, а не продолжение финального MGM checkpoint.

```bash
python scripts/cluster_collect.py --plan cluster_runs/sr_v2_critic_noise/plan.json
```

Сравниваем с `sr_4c20214/mgm`: mean±SD трёх seed, парные LPIPS/PSNR на тех же 50 validation center128, визуальные артефакты, разброс seed и training spikes. Held-out test пока не запускаем. Начальные поля/latent RNG и последовательности crop сохраняются при одинаковом seed/коде; ограничение по времени может дать разные числа обновлений. Прежние provenance gaps исходной серии учитываем при интерпретации; для строгого сравнения под одной ревизией можно указать `--variants mgm mgm_critic_noise` в **новом** run root (12 новых GPU-часов main).

На локальной машине GPU/Slurm-запуск не выполнялся. Передача кода на кластер и команды выше запускают эксперимент только после явного `--submit`.

## Контроль, продолжение и held-out test

```bash
squeue -u "$USER"
python scripts/cluster_collect.py --plan cluster_runs/sr_v4/plan.json
```

Логи лежат в `logs/warmup-…`, `train-…`, `eval-…`. Каждый training run содержит config, environment/provenance, progress, `train.jsonl`, atomic `latest.pt` и `status.json`. На SIGUSR1/SIGTERM Python завершает текущий шаг, сохраняется и выходит с кодом75; `consumed` и RNG сохраняются. Лимиты warmup/main оставляют 10 минут на startup/save сверх training budget. Evaluation отделена и не расходует этот запас.

Для продолжения конкретного элемента возьми его index и phase из plan и отправь отдельную задачу с `--resume` (пример: index0 train):

```bash
sbatch --array=0 --gpus=1 --cpus-per-task=4 --time=250 \
  --chdir="$PWD" --output=cluster_runs/sr_v4/logs/resume-%A_%a.out \
  slurm/cluster.sbatch "$PWD/cluster_runs/sr_v4/plan.json" train --resume
```

`--resume-if-exists` позволяет переотправить выбранные entries, продолжая существующие checkpoints и начиная отсутствующие. Завершённые checkpoints не переобучаются. Новые зависимости downstream jobs после отказа нужно выставлять на новые job IDs: изменение уже завершившегося job не исправляет старый afterok. Не редактируй подписанный plan и не запускай два процесса в один output directory одновременно.

После выбора гиперпараметров и NFE на validation можно вручную оценить весь eval array на test (для стандартных defaults 24 элемента):

```bash
sbatch --array=0-23%4 --gpus=1 --cpus-per-task=4 --time=30 \
  --chdir="$PWD" --output=cluster_runs/sr_v4/logs/test-%A_%a.out \
  slurm/cluster.sbatch "$PWD/cluster_runs/sr_v4/plan.json" eval --split test
```

После завершения: `python scripts/cluster_collect.py --plan cluster_runs/sr_v4/plan.json --split test`. Если variants/NFE/seeds изменены, границы массива берутся из `len(plan['eval'])`. Test сам по себе не запускается после training. Все NFE можно показать как заранее заявленную curve; выбирать лучший режим по test нельзя.

## Метрики и ограничения выводов

Оценка использует EMA и фиксированные draws на одинаковых 50 фотографиях. PSNR/SSIM — RGB [0,1] после clipping, без border shave; LPIPS AlexNet v0.1 на HR128. Отдельно записываются LR consistency до/после clipping, diversity и outputs с общей финальной projection. Hard projection гарантирует raw LR consistency конструктивно; это не преимущество MGM loss.

I²SB sampler начинается с nearest(LR), делает заявленное число вызовов сети и coarse posterior transitions. NFE1 детерминирован без дополнительного endpoint noise. У сети нет отдельного bicubic skip/extra LR concat. Последняя posterior mean возвращается на index0, как в upstream, без подмены на clean prediction или проекции внутри sampler. Это наш U-Net и training с нуля; upstream использует pretrained ADM.

Результаты находятся в `<variant>/seedN/eval_nfeK/val_metrics.json`, изображениях и PNG grid. Collector сохраняет `val_runs.csv`, `val_summary.md/json`; NFE — отдельные строки, mean±sample SD по seed. Время warmup записывается отдельно от main. Инференс timing включает весь sampler/NFE и генерацию bridge noise, исключает transfer/metrics/final projection и первый batch; FP32 batch1 одинаков для всех. Collector отмечает разные GPU или eval batch sizes: такие времена напрямую не ранжировать.

Результаты старой v1-серии не смешиваем с новой архитектурой/данными. Низкий LPIPS и высокий PSNR могут конфликтовать; преимущества оцениваем по quality–latency curve, визуальным примерам, устойчивости seed и фактическому training budget. По короткому пилоту нельзя заключать о сходимости или превосходстве.

Исходные формулы/архитектуры сверены с [I²SB diffusion](https://github.com/NVlabs/I2SB/blob/master/i2sb/diffusion.py), [runner](https://github.com/NVlabs/I2SB/blob/master/i2sb/runner.py), [RRDBNet](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py), [ESRGAN objective](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/esrgan_model.py). Это локальная реализация метода, не импорт upstream training stack. Native preprocessing helpers остаются справочными. Дополнительный старый full-frame utility из [FULL_IMAGES.md](FULL_IMAGES.md) поддерживает только наш U-Net G; основное сравнение всех моделей выполняет новый crop evaluator.

## Обновлённый MGM: остановка по эпохам

Следующий рецепт использует `--train-epochs 320 --field-warmup-epochs 100`, batch32 без накопления и Adam betas=(0.8,0.999). Одна условная эпоха —800 случайных train-патчей; main составляет8000 шагов, поле2500. Новый MSE warmup v1: `--warmup-epochs 1000`, batch16; v2 переиспользует старый warmup. Slurm wall-time лимит задаётся отдельно для каждой фазы. Прерванная задача сохраняется незавершённой и продолжается через `--resume-if-exists`; старые планы без epoch-флагов сохраняют остановку по времени. [Актуальные команды и продолжение](MGM_FIELD_WARMUP_DIAGNOSTICS.md).
