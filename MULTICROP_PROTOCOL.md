# DIV2K: несколько случайных crop128 для train и оценки

Согласовано 19 октября 2026. Новый протокол — **random128-multicrop-v4**. Размер моделей120/240/480, pooling×4 и image-level split800/50/50 сохраняются. Код и CPU-проверки подготовлены локально; реальные новые данные, GPU training/inference и Slurm submission здесь не запускались.

## Данные

- **Train:** четыре разных случайных crop128 на каждую из800 полных фотографий за эпоху. Эпоха состоит из четырёх независимо перемешанных проходов по всем800 фото. У каждого фото четыре разных положения crop; координаты меняются между эпохами. Нет resize, отражений и поворотов. Полные фото хранятся как uint8 NPY; банк фиксированных train-патчей не создаётся.
- **Validation:**50 прежних validation-фото ×5 фиксированных случайных crop128 =250 патчей.
- **Held-out test:**50 других прежних фото ×5 фиксированных случайных crop128 =250 патчей. Test не используется для выбора рецепта, checkpoints или NFE.
- Eval coordinates генерируются один раз при подготовке, из seed123 и image ID, без повторения положений внутри одного фото. Они записываются в manifest и одинаковы для всех моделей, training seeds и повторных оценок. Это не новые фотографии: пять crop одного фото статистически зависимы.
- Для каждого HR128 LR32 вычисляется как exact average pooling×4 после crop, в float без PNG-округления. HR/LR пары согласованы.
- Фото разделяются на train/val/test **до** нарезки. Координаты, порядок фотографий и train-патчей воспроизводятся по seed/index и сохраняются при resume, независимо от batch grouping и worker prefetch.

Предварительная подготовка не фиксирует train-кропы на всё обучение: четыре новых положения выбираются в каждой эпохе. Это отличается от старого банка6400 окон256. Старый `random128-v3` выбирал фото с возвращением и имел условную эпоху800 патчей; новый sampler гарантирует четыре посещения каждого фото за эпоху.

## Эпохи и объём обучения

Последнее решение пользователя: **50 эпох warmup →50 эпох поля →80 эпох MGM**. Одна эпоха =800×4=3200 патчей:

| Этап, на seed | Эпохи | Патчи | Physical batch | Шаги |
|---|---:|---:|---:|---:|
| Новый MSE warmup, v1 и v2 |50|160000|16|10000|
| Только поле |50|160000|32|5000|
| Совместный MGM |80|256000|32|8000|

Это обновляет предыдущий вариант250/25/80: warmup сокращён в5 раз, field-only увеличен в2 раза, объём MGM сохранён. Расписание является согласованным пилотным рецептом, а не воспроизведением длительности стадий из статьи. Учитываются обработанные примеры; successful/skipped optimizer updates раскрываются отдельно. Накопление отключено. MGM/поле: Adam lr_G=lr_field=1e-5, betas=(0.8,0.999), EMA G=.995. MSE warmup сохраняет batch16 и betas=(0.5,0.99).

Изменились sampler и evaluation manifest. Для нового протокола создаём **новые warmup обеих архитектур**; `sr_4c20214` warmup напрямую не переиспользуем, проверки provenance не ослабляем. Старые планы, данные и результаты остаются неизменными. Одинаковый объём патчей не гарантирует прежнюю скорость или качество.

## Время задач на V100 32GB

Ориентиры на один seed; начало отсчёта — работа на GPU, без ожидания в очереди:

| Фаза | Ожидаемое время | Slurm limit |
|---|---:|---:|
| Warmup,10000 шагов при batch16 |20–35 минут|60 минут|
| Поле,5000 шагов при batch32 |20–50 минут|90 минут|
| MGM,8000 пар обновлений при batch32 |100–180 минут|240 минут|
| Validation,250 crops |2–10 минут|30 минут|
| Collect,CPU |обычно меньше минуты|5 минут|

В прежних логах V100 warmup занимал0.145–0.151с/шаг (среднее0.148с), поэтому10000 шагов дают около25 минут. Старый MGM при batch16 занимал0.454–0.459с на пару обновлений; оценка для batch32 является экстраполяцией, field-only ещё не измерялся. Новые sampler/optimizer/AMP и дисковая нагрузка могут изменить скорость; лимиты выбраны с запасом. Память batch32 пока не измерена.

Все фазы завершаются по эпохам и освобождают GPU, даже если запрошено больше времени. Для двух архитектур ×3 seed ориентир примерно14–28 GPU-часов плюс очередь/диагностика; верхняя сумма запрошенных GPU limits —42 GPU-часа, не ожидаемый расход. До4 GPU работают параллельно, с отдельными барьерами фаз каждого плана.

Параметры host RAM не передаются в `sbatch`: используем настройки кластера. Это касается новых команд, evaluation-only, diagnostics и генерации команд из старых планов.

## Подготовка на кластере

Старый data root не перезаписывать. Если ZIP уже лежат в `data/div2k/downloads`:

```bash
python scripts/cluster_prepare.py \
  --crop-mode multicrop --eval-crops-per-image 5 \
  --data-root data/div2k_multicrop128 \
  --archive-root data/div2k/downloads --weights-root weights
```

Укажи фактический существующий каталог ZIP. Без `--archive-root` архивы будут скачаны в новый data root. Подготовка также проверяет/скачивает AlexNet и VGG19. На GPU-узлах скачивания нет.

Сохраняются800 full train NPY и banks `val.npy`/`test.npy`, каждый shape(250,128,128,3). В manifest есть `eval_crops_per_image=5`, policy, image IDs, crop_index/top/left и SHA256. Число train-кропов задаётся **в плане обучения** через `--train-crops-per-image 4`, а не `--patches-per-image` подготовки.

## Короткий запуск двух архитектур

После подготовки данных и активации окружения:

```bash
bash slurm/launch_mgm.sh
bash slurm/launch_mgm.sh --submit
```

Первая команда создаёт и показывает два подписанных плана; вторая ставит их в Slurm. Launcher фиксирует те же50/50/80 эпох и time limits, что подробные команды ниже. Одна модель/seed использует одну GPU, каждый план имеет max-parallel2, оба плана суммарно до4 GPU.

Перед submit на GPU-узлах должно активироваться установленное окружение:

```bash
conda activate mgm_i2i
export CONDA_ENV=mgm_i2i
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
```

Для другой директории prepared data задай `MGM_DATA_ROOT`. Для отдельной новой серии задай `MGM_RUNS_ROOT`; существующие планы не редактируются. Повторный `--submit` уже поставленного плана блокируется; продолжать partial jobs нужно через resume.

## Новый MGM v2 critic-only

```bash
python scripts/cluster_submit.py \
  --run-root cluster_runs/sr_v2_multi4_eval5_w50_f50_g80_b32_beta08 \
  --data-root data/div2k_multicrop128 --data-protocol random128-multicrop-v4 \
  --train-crops-per-image 4 --eval-crops-per-image 5 \
  --architecture-version 2 --width 120 --variants mgm_critic_noise --seeds 0 1 2 \
  --warmup-epochs 50 --field-warmup-epochs 50 --train-epochs 80 \
  --field-initialization warmup_features \
  --mgm-optimizer authors --mgm-beta1 0.8 \
  --mgm-microbatch-size 32 --grad-accumulation 1 \
  --warmup-time-limit-minutes 60 --field-warmup-time-limit-minutes 90 \
  --train-time-limit-minutes 240 --eval-minutes 30 --max-parallel 2
```

## Новый MGM v1 wide

```bash
python scripts/cluster_submit.py \
  --run-root cluster_runs/sr_v1_multi4_eval5_w50_f50_g80_b32_beta08 \
  --data-root data/div2k_multicrop128 --data-protocol random128-multicrop-v4 \
  --train-crops-per-image 4 --eval-crops-per-image 5 \
  --architecture-version 1 --width 120 --variants mgm --seeds 0 1 2 \
  --warmup-epochs 50 --field-warmup-epochs 50 --train-epochs 80 \
  --field-initialization warmup_features \
  --mgm-optimizer authors --mgm-beta1 0.8 \
  --mgm-microbatch-size 32 --grad-accumulation 1 \
  --warmup-time-limit-minutes 60 --field-warmup-time-limit-minutes 90 \
  --train-time-limit-minutes 240 --eval-minutes 30 --max-parallel 2
```

Обе команды — dry run. После просмотра плана повторить каждую с `--submit`. Два плана с max-parallel2 используют суммарно до4 GPU. Фазы каждого плана: warmup → field_warmup → train → validation → collect. Поле и генератор получают признаки собственного архитектурно совместимого warmup. Настройки inference и формулы выходов прежние; critic-only v2 имеет noise_g=0,noise_f=.01, v1 noise_g=noise_f=.01.

Фактические training GPU-часы неизвестны до запуска. В плане `training_gpu_hours=null`, а Slurm limits — ограничения одной allocation. При прерывании training сохраняется `complete=false`; повторная задача с `--resume-if-exists` продолжает счётчик эпох и оптимизаторы. Исходные зависимости от failed job не переназначаются автоматически; после resume надо заново запустить зависимые фазы. Не запускать параллельные задачи, пишущие один checkpoint.

Для smoke нужен отдельный run root и `--seeds 0 --warmup-epochs 1 --field-warmup-epochs 1 --train-epochs 1`; это проверка исполнения, не качества. Перед длинным main по-прежнему нужен V100 batch32 memory probe. Диагностика `G(c,alpha*z)` автоматически использует новый native validation bank, все250 патчей; выводит per_crop.csv и per_image.csv.

## Оценка и визуализации

Метрики считаются на каждом патче, затем усредняются по пяти патчам одной фотографии, затем по50 фотографиям. В JSON/CSV явно различаются **images=50** и **crops=250**. Разброс между патчами не выдаётся за разброс250 независимых фотографий.

`val_samples.png` показывает по одному crop первых8 разных фото. В `samples/` сохраняются grids **всех250 патчей**: nearest LR / bicubic / projected bicubic / HR / draw1 / draw2 / final projected draw1. В `val_patches.json` сохранены image ID, crop_index и координаты; `val_images.json` содержит средние по фото.

Оценка на этих crop не сопоставима напрямую со старым center128 score. Чтобы сравнить уже обученные baseline и новые MGM, все checkpoints нужно оценить на одном новом bank. Planner поддерживает frozen старые модели, не меняя их config/training hash:

```bash
python scripts/cluster_eval_submit.py \
  --source-runs cluster_runs/sr_4c20214 \
  --run-root cluster_runs/div2k_multi5_old_models_val \
  --data-root data/div2k_multicrop128 --split val \
  --max-parallel 2 --eval-minutes 30
```

Позже можно добавить completed новые run roots в `--source-runs`, используя новый общий evaluation run root. Это evaluation-only; источники обучения подписаны, новый evaluation manifest фиксируется отдельным hash. Сначала используем `--split val`. `--split test` — отдельная финальная оценка после фиксации всех решений.

Старые и новые модели всё равно имеют разные training sampler/рецепты; общая evaluation сравнивает итоговые системы, а не изолирует один эффект MGM. BSD100 остаётся отдельным ранее согласованным внешним center128 benchmark; его результаты не смешиваются с DIV2K multicrop.

## Локальная проверка

Пройдены75 тестов проекта (`python -m pytest tests -q`): distinct train positions и ровно4 посещения каждого фото за эпоху; смена координат между эпохами; одинаковые последовательности при resume/prefetch и разном batch grouping; подготовка250 eval crops с точным соответствием исходным пикселям и image-disjoint splits; фиксированные hashes; корректное усреднение по фото; общая оценка старых моделей; совместимость исторических протоколов; field initialization и epoch stopping. Обе команды обучения выше выполнены как dry run в временных каталогах: plan v6 содержит по3 warmup/field/main/eval задачи с целями160000/160000/256000 патчей. Новый launcher `slurm/launch_mgm.sh` также проверен dry run: оба плана без memory flags, лимиты60/90/240/30 минут. Bash syntax и Python compilation проверены. Реальные новые данные и CUDA training/inference не запускались.
