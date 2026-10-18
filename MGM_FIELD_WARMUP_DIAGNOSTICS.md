# MGM: перенос признаков поля, latent diagnostic и author optimizer

**Обновление19 октября:** актуальные данные и команды — [MULTICROP_PROTOCOL.md](MULTICROP_PROTOCOL.md):4 train-crop/фото/эпоху,5 фиксированных случайных val/test-crop/фото,50/50/80 эпох. Обе архитектуры получают новые warmup. Команды с center128 evaluation и reuse `sr_4c20214` ниже относятся к прежнему протоколу; для их воспроизведения нужны явные `--data-protocol random128-v3 --data-root data/div2k_random128`. Математика переноса поля, alpha sweep и optimizer остаётся применимой.

Согласовано 18 октября 2026. Этот рецепт обновляет ранее подготовленные wide v1 / v2 critic-only запуски. Старые планы и checkpoints не редактируем. Новые GPU-задачи запускает пользователь на кластере после получения кода; локально выполнены CPU-проверки, фактического V100 profiling/inference/training здесь нет.

## 1. Чем инициализируется поле

Флаг `--field-initialization warmup_features` переносит обученные conv/norm/skip блоки U-Net из **generator** весов MSE warmup в поле того же width и architecture version. Это наш адаптер SR warmup, а не точная копия авторского EDM-denoising pretraining.

У генератора вход `[bicubic(C), Z]`, у поля `[bicubic(C), mixed_HR]`. Latent-фильтры генератора при Z=0 не обучались, поэтому их НЕ копируем в mixed_HR branch. Обученные RGB stem-фильтры W делим между обеими ветвями: `[W/2, W/2]`; при двух близких изображениях это сохраняет разумный масштаб активаций. Дополнительные time maps v1 и time injections v2 начинаются с нулевого вклада. Внутренние признаки переносятся; выходная displacement-голова сбрасывается, поскольку RGB-residual и displacement имеют разный смысл.

Затем `--field-warmup-epochs 100` создаёт **отдельный field-only этап**, который обучает новую голову и адаптирует перенесённые признаки:

\[
\min_\psi E\|f_\psi(C,t,\widetilde X_t)-(Y-G_{warm}(C,Z))\|^2.
\]

G заморожен, его веса/EMA не обновляются, Gaussian Z включён, данные только train random128. Перед совместным MGM поле уже получило regression training. Передаём в main обученные веса поля, а Adam moments начинаем заново. Checkpoint guards требуют completed field-only этап, тот же dataset hash, **тот же warmup checkpoint hash G**, seed, width, architecture и noise/batch/optimizer policy. Сто условных эпох — стартовый пилотный бюджет, не число из статьи и не обещание сходимости поля. Одна эпоха означает800 случайно выбранных train-патчей.

Фазы: **MSE warmup (или reuse) → field_warmup → совместный MGM → validation → collect**. Warmup и field initialization costs сохраняются раздельно. Новые логи включают field regression / zero-field baseline на тех же samples, pre-clip gradient norms, факт gradient clipping, AMP scale и effective batch. Отношение field loss к zero-field baseline полезно для диагностики, но само по себе не доказывает совпадение распределений.

## 2. Сначала сравнить warmup при разных latent amplitudes

Для каждого сохранённого warmup запускается **FP32 inference** без изменения весов. Для всех четырёх alpha используется один и тот же фиксированный Z и те же 50 validation center128:

\[
G(C,0),\quad G(C,0.1Z),\quad G(C,0.5Z),\quad G(C,Z).
\]

По умолчанию выбираются `state['generator']`, а не EMA: engine именно эти веса загружает перед main. Опция `--weights ema` возможна как отдельная диагностика, но её нельзя смешивать с фактической initializer quality.

Команды на кластере:

```bash
mkdir -p cluster_runs/warmup_diagnostics/logs
sbatch --chdir="$PWD" --array=0-2%1 \
  --output=cluster_runs/warmup_diagnostics/logs/diagnostic-%A_%a.out \
  slurm/warmup_diagnostics.sbatch \
  "$PWD/cluster_runs/sr_4c20214" "$PWD/cluster_runs/warmup_diagnostics"
```

В каждой из трёх GPU-задач считается один training seed; seed0 дополнительно выполняет memory probe. Для нового v1 позже повтори с его run root и отдельным diagnostics root, когда его MSE warmup будет completed. Строки source root всегда означают каталог, внутри которого есть `warmup/seedN/latest.pt`.

Одна проверка вручную внутри GPU allocation:

```bash
python scripts/cluster_diagnose_warmup.py \
  --checkpoint cluster_runs/sr_4c20214/warmup/seed0/latest.pt \
  --output cluster_runs/warmup_diagnostics/seed0
```

Выходы `seedN/diagnostic.json`, `README.md`, `per_image.csv`, `alpha_samples.png` и отдельные grids **всех 50 фото** в `samples/`. Колонки: LR nearest / HR / alpha0 / alpha0.1 / alpha0.5 / alpha1. Сохраняются PSNR/SSIM/LPIPS, raw MSE, raw output change относительно alpha0, LR consistency и доля out-of-range компонентов. Не выбираем alpha по held-out test. Это диагностика чувствительности к Z; alpha0 не выдаём за исходный stochastic MGM результат.

## 3. Память V100 и effective batch

Согласованный рецепт: **physical batch32 = effective batch32, без накопления градиентов**, Adam betas=(0.8,0.999) для G и поля. В старой серии batch16, width120, HR128, AMP занимал peak PyTorch allocated **9.445 GiB** на Tesla V100-SXM2-32GB. Это не полная занятость VRAM; вместимость batch32 ещё не измерена.

`cluster_probe_mgm.py` проверяет физический batch в отдельном свежем CUDA subprocess. В памяти есть G, поле, EMA, gradients и оба Adam optimizer states. Требуются три успешных field+G updates; AMP skipped steps не считаются успешными. Можно передать несколько размеров через `--batch-sizes`; после OOM более крупные размеры не запускаются. Рекомендация учитывает reserved-memory headroom ≥15%. Согласованный запуск проверяет только32:

```bash
python scripts/cluster_probe_mgm.py \
  --checkpoint cluster_runs/sr_4c20214/warmup/seed0/latest.pt \
  --output cluster_runs/warmup_diagnostics/memory_probe.json \
  --batch-sizes 32 --mgm-beta1 0.8
```

Probe использует synthetic targets для train shapes/graphs/optimizer allocations; это не измерение качества или throughput. Проверяются architecture version checkpoint и AMP policy. После смены width/crop/backbone/precision необходимо новое измерение.

В выбранном запуске:

\[
B_{effective}=B_{physical}\times K =32\times1=32.
\]

Один field update обрабатывает32 примера, затем один G update — те же HR/LR-примеры с новыми случайными Z и переменными objective. EMA обновляется один раз на успешный G optimizer step. Ранее обсуждавшийся effective batch256 через накопление отменён. Код поддерживает накопление для других экспериментов, но оба запуска ниже задают `--grad-accumulation 1`. При OOM скрипт не уменьшает batch и не включает накопление автоматически; сначала выполняем probe.

При увеличении effective batch будет меньше Adam/EMA steps за те же часы; в новых логах `step` означает optimizer update pair, `consumed` — число реальных samples. EMA decay пока сохраняется0.995 на successful G step, поэтому его горизонт в samples тоже меняется. Это отдельное отличие от статьи, где EMA0.999; полного воспроизведения original recipe не заявляем.

## 4. Новый optimizer recipe и запуски

`--mgm-optimizer authors` задаёт **только MGM / field-only**:

- Adam lr_G=lr_field=1e-5;
- betas=(0,0.999);
- прежние MSE warmup configs сохраняются, поэтому можно reuse их checkpoint;
- EMA, gradient clipping, architecture, conditional SR objective и данные не становятся авторской CIFAR-конфигурацией автоматически.

Источник: [MGM, Appendix B](https://arxiv.org/html/2605.29920v2#A2). В согласованном рецепте дополнительно ставим **`--mgm-beta1 0.8`**: lr остаются1e-5, beta2 остаётся0.999, beta1 переопределяется для обоих оптимизаторов. В плане это `authors_beta1_override`, а не точное воспроизведение авторских betas. Legacy CLI defaults сохранены для прежних планов; новый запуск использует явные флаги ниже.

V2 critic-only, physical/effective batch32, без накопления, reuse исходного MSE warmup:

```bash
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 \
  --run-root cluster_runs/sr_v2_critic_epochs320_b32_beta08 \
  --architecture-version 2 --width 120 \
  --variants mgm_critic_noise --seeds 0 1 2 \
  --reuse-warmup-from cluster_runs/sr_4c20214 \
  --field-initialization warmup_features --field-warmup-epochs 100 \
  --train-epochs 320 \
  --field-warmup-time-limit-minutes 60 --train-time-limit-minutes 240 \
  --mgm-optimizer authors --mgm-beta1 0.8 \
  --mgm-microbatch-size 32 --grad-accumulation 1 \
  --max-parallel 1
```

Это dry run. Повтори те же аргументы с `--submit` для очереди. Фактическое training time теперь определяется скоростью выполнения100 эпох поля и320 эпох main. `training_gpu_hours=null`: фиксированные GPU-часы больше не задаются. `training_time_hint_gpu_hours` — только прежний ориентир, `allocation_gpu_hours_upper_bound` — сумма запрошенных Slurm лимитов, а не обещанный расход. Прежние6 GPU-часов MSE warmup раскрываются отдельно.

Обновлённый wide v1 с новым собственным MSE warmup и обычным noise0.01/0.01:

```bash
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 \
  --run-root cluster_runs/sr_v1_epochs320_b32_beta08 \
  --architecture-version 1 --width 120 \
  --warmup-epochs 1000 --warmup-time-limit-minutes 180 \
  --variants mgm --seeds 0 1 2 \
  --field-initialization warmup_features --field-warmup-epochs 100 \
  --train-epochs 320 \
  --field-warmup-time-limit-minutes 60 --train-time-limit-minutes 240 \
  --mgm-optimizer authors --mgm-beta1 0.8 \
  --mgm-microbatch-size 32 --grad-accumulation 1 \
  --max-parallel 2
```

MSE warmup сохраняет batch16 и Adam betas=(0.5,0.99); новые batch32/betas=(0.8,0.999) применяются к field-only и совместному MGM. V2 использует исходный time-budget warmup (в среднем972 условные эпохи). V1 создаёт собственный warmup на1000 условных эпохах; это близкий, но не идентичный объём. Новый main для обоих вариантов —320 эпох. Эти настройки являются стартовым экспериментом, оптимальное число эпох пока не установлено.

Если старые запуски уже поставлены в очередь, новые их не заменяют автоматически. Разные run roots обязательны. Для короткого smoke нового epoch-рецепта: `--seeds 0 --train-epochs 1 --field-warmup-epochs 1` и новый root; `--pilot` рассчитан на время и несовместим с epoch-флагами.

## 5. Остановка по эпохам и продолжение

Эпоха — **800 случайных train-патчей**, с выбором фото с возвращением; она не гарантирует посещение каждой фотографии ровно один раз. Sampling остаётся прежним random128-v3. Обработанные батчи с AMP-skipped updates входят в счётчик патчей; successful/skipped updates записываются отдельно.

| Этап нового запуска | Эпохи | Патчи | Batch | Шаги |
|---|---:|---:|---:|---:|
| MSE warmup v1 |1000|800000|16|50000|
| Поле, v1/v2 |100|80000|32|2500|
| MGM main, v1/v2 |320|256000|32|8000|

`--warmup-epochs`, `--field-warmup-epochs`, `--train-epochs` включают остановку по патчам отдельно для каждой фазы. Без соответствующего флага остаётся прежняя остановка по времени. При epoch-режиме `config.seconds` служит ориентиром для планирования; training loop не останавливается по этому значению. Runtime проверяет число train-фото по данным; цель должна делиться на effective batch без остатка.

Slurm limits задаются отдельно:180 минут на новый v1 warmup,60 на поле,240 на main. Это верхние ограничения одной allocation, не условия успешного завершения. При раннем USR1/SIGTERM сохраняются `complete=false`, processed patches, веса, Adam/scaler/RNG; процесс выходит75. Следующая фаза через `afterok` не запускается. Продолжение отдельной задачи, когда предыдущая уже завершилась:

```bash
sbatch --chdir="$PWD" --gpus=1 --cpus-per-task=4 \
  --partition=rocky --time=240 --array=0-2%1 \
  --output=cluster_runs/sr_v2_critic_epochs320_b32_beta08/logs/train-resume-%A_%a.out \
  slurm/cluster.sbatch \
  "$PWD/cluster_runs/sr_v2_critic_epochs320_b32_beta08/plan.json" train --resume-if-exists
```

`--resume-if-exists` продолжает partial checkpoints и пропускает уже completed seeds; не обнуляет epoch counter. Если прервана фаза поля, сначала повтори её с `field_warmup`, лимитом60 и отдельным log name. После завершения повторных training jobs заново запусти зависимые фазы: исходные `afterok` зависимости от failed job не переназначаются автоматически. Не запускай две задачи, пишущие один checkpoint, одновременно.

В `train.jsonl`, `status.json`, checkpoint и evaluation metadata сохраняются `epochs_completed`, `target_epochs`; дополнительно status/checkpoint содержат `target_samples`, `samples_per_epoch`. Progress показывает эпохи и патчи. Одинаковое число эпох означает одинаковый объём данных, но разные GPU-часы для разных методов; прежнее сравнение по равному времени сохраняется в старых планах.

Комбинация optimizer/field-init/batch меняет сразу несколько факторов. Это проверка улучшенного рецепта. Для определения причин выигрыша нужны дополнительные отдельные ablations. BSD100 eval planner поддерживает новые completed main checkpoints как обычные source runs; field-only checkpoints туда не включаются.

## Adam beta1

Adam хранит экспоненциальные средние градиента и его квадрата:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

После bias correction шаг равен \(\theta_{t+1}=\theta_t-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)\). Beta1 управляет памятью направления: при0.8 это80% прежнего среднего и20% текущего градиента; при0.5 половина приходится на текущий gradient и половина на накопленную историю; при0 — m_t=g_t. Это не SGD: нормировка через второй момент остаётся. В minimax-задаче направление меняется вслед за второй сетью; снижение памяти может уменьшать запаздывание, но превосходство beta1=0 не гарантируется для произвольной задачи. [PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).

Локально пройдены56 CPU-проверок: перенос v1/v2 без копирования latent filters, неизменность G при field-only, обучение поля, корректный G step после warmup, равенство accumulated Adam/full-batch update, source checkpoint guards, phase dependencies, fixed-Z alpha sweep и новый batch32/beta1=0.8 план с reuse прежнего warmup. Проверен реальный training loop на CPU: прерывание, сохранение незавершённого checkpoint и продолжение до точного epoch target дают те же веса и Adam moments, что непрерывный запуск. GPU quality и реальный batch fit ещё не измерены.
