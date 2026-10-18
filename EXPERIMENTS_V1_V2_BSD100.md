# Следующие эксперименты: wide v1, v2 critic-only и BSD100

**Обновление19 октября:** актуальные данные и команды — [MULTICROP_PROTOCOL.md](MULTICROP_PROTOCOL.md):4 train-crop/фото/эпоху,5 фиксированных случайных val/test-crop/фото,50/50/80 эпох. Обе архитектуры получают новые warmup. Команды с center128 evaluation и reuse `sr_4c20214` ниже относятся к прежнему протоколу; для их воспроизведения нужны явные `--data-protocol random128-v3 --data-root data/div2k_random128`. Математика переноса поля, alpha sweep и optimizer остаётся применимой.

**Обновление:** пользователь добавил перенос признаков поля, field-only warmup, Adam lr1e-5, betas=(0.8,0.999) и physical/effective batch32 без накопления; теперь100 условных эпох поля и320 эпох main. Новые рекомендуемые команды обучения и диагностики — [MGM_FIELD_WARMUP_DIAGNOSTICS.md](MGM_FIELD_WARMUP_DIAGNOSTICS.md). Команды BSD100 в этом документе остаются применимы; training-команды ниже описывают предыдущий рецепт.

Согласовано 18 октября 2026. Пользователь запускает команды на кластере после обновления кода. Локально подготовлены код, проверки и BSD100; GPU training/inference здесь не выполнялись.

## 1. Два обучения

| Run | Генератор | Поле | noise_g / noise_f | Новое обучение, 3 seed |
|---|---:|---:|---|---|
| `sr_v1_wide` | v1 120/240/480, 16 023 003 параметра | 16 027 323 | 0.01 / 0.01 | 6 GPU-ч warmup + 6 GPU-ч MGM |
| `sr_v2_critic_noise` | v2 120/240/480, 16 023 003 параметра | 17 293 203 | 0 / 0.01 | 6 GPU-ч MGM; прежний v2 warmup |

Общий дополнительный training budget — **18 GPU-часов**, плюс evaluation. Сохраняем DIV2K random128-v3, batch16, main120 минут на seed, projection и optimizer. У wide v1 **новый v1 warmup**, поскольку v2 checkpoint не является контролируемой инициализацией v1. V1 использует обычный MGM, чтобы сравнить его с уже обученным обычным v2 MGM. V2 critic-only сравниваем с тем же исходным v2 MGM. Сравнение только двух новых вариантов смешало бы архитектуру и noise policy.

Ниже команды с `--submit`. Для просмотра плана без очереди сначала выполни ту же команду без `--submit`, затем повтори с ним. При необходимости менять аргументы уже записанного плана используй новый run root. Планы создавать **на кластере**: они фиксируют абсолютные пути этого компьютера.

```bash
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 \
  --run-root cluster_runs/sr_v1_wide \
  --architecture-version 1 --width 120 \
  --variants mgm --seeds 0 1 2 \
  --max-parallel 2 --submit
```

```bash
python scripts/cluster_submit.py --data-protocol random128-v3 --data-root data/div2k_random128 \
  --run-root cluster_runs/sr_v2_critic_noise \
  --architecture-version 2 --width 120 \
  --variants mgm_critic_noise --seeds 0 1 2 \
  --reuse-warmup-from cluster_runs/sr_4c20214 \
  --max-parallel 1 --submit
```

Для второй команды нужны `sr_4c20214/plan.json`, `plan.sha256` и три `warmup/seedN/latest.pt`. Загрузка финального MGM вместо warmup не выполняется. Source plan и configs проверяются; engine проверяет completed/data hash/архитектуру реального checkpoint и записывает init_hash. При совпадении seed и кода сравниваются те же начальные веса G.

Каждый training plan автоматически завершится оценкой на прежних 50 DIV2K validation center128 и своим collect. Новые held-out test задачи не создаются.

## 2. BSD100 для всех уже обученных моделей

Пользователь выбрал отдельный BSD100, сохранив 50 held-out DIV2K. Источник — HR PNG из [SelfExSR](https://github.com/jbhuang0604/SelfExSR/tree/8f6dd8c1d20cb7e8792a7177b4f6fd677633f598/data/BSD100/image_SRF_4), commit `8f6dd8c1d20cb7e8792a7177b4f6fd677633f598`. Каталог всех 100 файлов, размеры и Git blob hashes сохранены в [research/bsd100_source.json](research/bsd100_source.json). Это HR-файлы SRF_4 benchmark, которые могут уже иметь обрезку границ под масштаб; дополнительные resize/аугментации не выполняем. Предоставленные авторами bicubic LR не используем.

Подготовка на login node скачивает только 100 HR PNG (~27 MB), проверяет каждое содержимое и создаёт uint8 банк `100×128×128×3` (~4.9 MB). RGB conversion, один центральный128 crop с каждого отдельного изображения; LR32 строится exact average pooling ×4. Готовый банк здесь имеет manifest SHA256 `d1bc4484dd448133944fb77e01ddac6d6c284d666e8bbd7ad71d02a6ff8bd53c`; результат повторной подготовки с тем же каталогом должен совпасть.

```bash
python scripts/cluster_prepare_bsd100.py --data-root data/bsd100_center128
```

Для текущих пяти методов, трёх seed и всех четырёх I²SB NFE (24 evaluation-задачи):

```bash
python scripts/cluster_eval_submit.py \
  --source-runs cluster_runs/sr_4c20214 \
  --run-root cluster_runs/bsd100_sr_4c20214 \
  --data-root data/bsd100_center128 \
  --max-parallel 1 --submit
```

Если сохранены checkpoints исторической v1-серии, включи **обе серии** сразу в отдельный plan (42 задачи: 24 текущих + 18 исторических). Это позволяет сравнить уже обученные системы на общем BSD100, хотя различия в training protocol остаются:

```bash
python scripts/cluster_eval_submit.py \
  --source-runs cluster_runs/sr_4c20214 cluster_runs/full_2h_20261013_202802 \
  --run-root cluster_runs/bsd100_existing_all \
  --data-root data/bsd100_center128 \
  --max-parallel 1 --submit
```

Выбери одну из двух команд выше, чтобы не считать текущую серию дважды. Требуются все checkpoints выбранных source plans; отсутствие файла блокирует submit до первой sbatch. Архив JSON/PNG без `.pt` недостаточен. Данные обучения не копируются и training jobs не создаются. Используются EMA из сохранённых checkpoints. Labels включают имя исходной серии, поэтому v1/v2 и разные NFE не смешиваются.

**Лимит параллельности относится к одному плану.** Приведённые настройки wide v1 `%2` + v2 critic `%1` + один BSD100 plan `%1` дают максимум4 одновременно занятых GPU при параллельном запуске всех трёх планов. `%4` в каждом отдельном плане не даёт общего лимита4. Для отдельного запуска BSD100 после обучения можно увеличить лимит в новом плане.

В BSD100 jobs используется FP32 batch1 для всех, включая историческую v1. Legacy config/checkpoint не редактируется; только внешний evaluation crop фиксируется как128. Для I²SB сохраняется исходная сетка NFE1/4/16/64. Manifest hash внешнего банка закреплён в плане и проверяется перед inference; training hash сохраняется отдельно. Метрики: RGB PSNR/SSIM после clipping, LPIPS AlexNet v0.1, native и final-projected outputs, LR consistency, diversity, latency. Те же seed draws для всех моделей; не выбираем лучший draw.

**Это наша оценка BSD100 center128/pooling, а не опубликованный full-frame bicubic benchmark.** Результаты этой серии нельзя подставлять в стандартные SOTA-таблицы BSD100. Если BSD100 используется для выбора следующих настроек, считаем его дополнительным validation; независимая финальная проверка остаётся DIV2K held-out50.

## 3. Результаты и повторная оценка новых моделей

```bash
python scripts/cluster_collect.py --plan cluster_runs/bsd100_sr_4c20214/plan.json
```

Для варианта со всеми историческими моделями подставь `cluster_runs/bsd100_existing_all/plan.json`. Collector также автоматически запускается в CPU job afterany и перечисляет отсутствующие результаты, если часть evaluation не завершилась.

Выходы:

- `external_runs.csv`, `external_summary.md/json` — метрики mean±SD по seed с отдельными строками NFE/серий;
- `<source>/<variant>/seedN/eval_nfeK/external_images.json` — метрики каждого из100 изображений;
- `external_samples.png` — общий grid первых8;
- `samples/` — отдельные PNG для **всех100** фотографий, в порядке: nearest LR / bicubic / projected bicubic / HR / draw1 / draw2 / projected draw1.

После завершения обоих новых training plans можно оценить их отдельно (ещё6 inference-задач):

```bash
python scripts/cluster_eval_submit.py \
  --source-runs cluster_runs/sr_v1_wide cluster_runs/sr_v2_critic_noise \
  --run-root cluster_runs/bsd100_new_mgm \
  --max-parallel 4 --submit
```

Этот последний план ставим после завершения двух новых обучений. Он не создаёт cross-plan dependency и проверяет наличие checkpoint-файлов; evaluator дополнительно требует completed=true. Во всех оценках используем одинаковый тип GPU для сравнения latency. Исторические gaps в Git provenance первой серии остаются ограничением сравнения.

## 4. Что проверено локально

Проверены исходные100 Git blob hashes, все100 center crops пиксель-в-пиксель с PNG, bank/manifest SHA256 и отсутствие обращения к DIV2K test. Выполнены dry-run планы wide v1 (3 warmup +3 train +3 eval) и общего BSD100 для текущей/исторической серии (42 eval). CPU-тесты проверяют различие noise-ветвей, совместимость warmup, immutable plans, внешние data hashes, сохранение100 PNG и агрегацию общих eval-данных при разных training hashes. GPU/Slurm и фактическое качество новых моделей проверяются пользовательским запуском на кластере.
