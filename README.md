# MGM для I2I: исследование и эксперименты

**Общий preprocessing:** для всех методов подтверждены случайные train-патчи128, фиксированный center128 на validation/test и average pooling ×4; оригинальные native transforms ESRGAN/I²SB остаются справочными. Подробности и варианты расширения данных — [research/PREPROCESSING_PROTOCOL.md](research/PREPROCESSING_PROTOCOL.md). ESRGAN, VGG perceptual loss и I²SB подключены к training/evaluation engine; CUDA/VRAM ещё нужно проверить пилотом.

**Data protocol всех выбранных моделей:** `random128-v3`: полные train-фото сохраняются в mmap NPY, loader выбирает случайные исходные HR crops128 без resize/поворотов/отражений. Validation/test — фиксированные center128; LR32 через pooling ×4. Новая подготовка — `data/div2k_random128`, нужны новый run root и warmup. Старые планы center128-v2 и legacy сохраняют исходные данные и loader. Подробности — [CLUSTER.md](CLUSTER.md).

**Решения и идеи:** [research/IDEAS.md](research/IDEAS.md). Новые кластерные планы используют ширины **120/240/480** (G — 16.02 млн параметров, MGM с полем — 33.32 млн) и архитектуру v2: RGB-residual без `tanh`, sinusoidal TimeEmbedding поля. Старые checkpoints оцениваются с прежней архитектурой; подробности совместимости — в [CLUSTER.md](CLUSTER.md).

**Запуск на Slurm:** [CLUSTER.md](CLUSTER.md). План v4 включает MSE/MSE+perceptual/ESRGAN/MGM/I²SB, 3 seed, отдельные warmup/train/eval arrays и collector. Каждая GPU-задача использует одну GPU; default `--max-parallel 4`. Основная серия: 48 GPU-часов training плюс evaluation; пилот всех пяти методов — 13 GPU-минут training плюс evaluation. Перед `--submit` создаётся reviewable plan; в этой сессии Slurm/GPU не запускались.

**Полные фотографии:** [FULL_IMAGES.md](FULL_IMAGES.md) — дополнительный инференс целых DIV2K-кадров готовыми EMA-checkpoints, PNG без уменьшения, сравнение методов в HTML и отдельные PSNR/SSIM. Новый режим и его тесты пока не запускались.

GitHub содержит исходники, конфигурации и текстовые инструкции. Датасеты, веса, PDF, сторонние репозитории, Mac-окружение и архив офлайн-пакета в Git **не включены**. Для Linux скачивание готовит `scripts/cluster_prepare.py`. Раздел ниже описывает ранее подготовленный локальный комплект на Mac; наличие этих файлов после `git clone` не предполагается.

## Локальный офлайн-пакет для Mac

Подробная содержательная подготовка без новых запусков: **[теория и проект исследования](offline/THEORY_AND_RESEARCH_PLAN.md)**, **[текст десяти слайдов на 20 минут](offline/PRESENTATION_20_MIN.md)** и **[34 вопроса к защите](offline/DEFENSE_QA.md)**. Эти новые текстовые документы находятся в папке проекта. Прежний ZIP не пересобирался в рамках запроса без запуска скриптов.

Начни с **[offline/guide.html](offline/guide.html)**: это самостоятельная инструкция на русском с формулами, локальными ссылками и планом на 20 часов. Интернет для чтения не нужен. Текстовая версия: [offline/GUIDE.md](offline/GUIDE.md).

Пакет подготовлен для этого Mac: Apple Silicon, Python 3.13. Установочные wheel-файлы в `offline/wheels/` относятся именно к этой платформе. Статьи и наш исходный код переносимы; `.venv` не переносима.

Всё уже находится в папке проекта и готово к работе. Резервная копия — `MGM_I2I_OFFLINE_MAC.zip` в корне проекта: статьи, данные, веса, установочные файлы, код и проверочные результаты. После распаковки при необходимости выполни `sh scripts/setup_offline.sh`. Начальная страница — `START_HERE.html`; её можно открыть в Safari/Chrome без локального сервера.

## Первые команды

В Terminal, из папки проекта:

```bash
cd /Users/kirill/edu/mgm_i2i
.venv/bin/python scripts/check_offline.py
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/train.py --config configs/toy_mse.json
.venv/bin/python scripts/train.py --config configs/toy_mgm.json
.venv/bin/python scripts/compare.py runs/toy_mse runs/toy_mgm
```

Для нового запуска укажи другое `--output`; существующие результаты не перезаписываются. `Ctrl+C` сохраняет checkpoint.

## Окружение и данные без сети

Окружение уже установлено. Если понадобится восстановить его:

```bash
sh scripts/setup_offline.sh
```

Все установочные файлы, изображения CIFAR-10, LPIPS/AlexNet-веса и JS для формул находятся локально. Скрипты обучения не скачивают данные. `scripts/download_materials.py` нужен только для повторной подготовки при наличии сети.

## Изображения

```bash
.venv/bin/python scripts/train.py --config configs/cifar_mse.json
.venv/bin/python scripts/train.py --config configs/cifar_mgm.json
.venv/bin/python scripts/train.py --config configs/cifar_gan.json
.venv/bin/python scripts/compare.py runs/cifar_mse runs/cifar_mgm runs/cifar_gan
```

Это конфигурации стартового исследования, а не подобранные лучшие гиперпараметры. На CPU сначала проверь 20–50 шагов, измерь скорость и реши, сколько обучать. Команды smoke-test и продолжения есть в руководстве.

## Где что находится

- `materials/` — основная статья и десять дополнительных PDF; карта чтения в руководстве.
- `third_party/` — распакованные публичные исходники EDM, SiD, pix2pix, Zero-Flow и categorical MGM. Это справочный код; наш прототип его не импортирует.
- `src/mgm_i2i/` — наша адаптация: модели, данные, MGM-loss и метрики.
- `scripts/` — обучение, оценка, сравнение, проверка офлайн-работы.
- `configs/` — настройки синтетических и CIFAR-экспериментов.
- `data/cifar10/` — 50 000 исходных train-изображений и 10 000 test-изображений. Train разбивается на 45 000 train / 5 000 val.
- `runs/verified_*` — проверочные запуски, полученные при подготовке; не итоговый результат задания.
- `offline/verification.json` — результат проверки с заблокированными сетевыми соединениями.
- `offline/inventory.json` — происхождение и SHA-256 скачанных файлов.
- `offline/AI_AND_COMPUTE.md` — шаблон раскрытия использования AI и вычислений по ТЗ.

Главная гипотеза: условный MGM сравнивает распределения HR-выходов для одного LR-входа. На синтетике проверяем две моды и неопределённость; на CIFAR — точность, perceptual-качество, согласованность и стоимость. Совпадение marginal-распределений само по себе не доказывает правильную I2I-связь.
