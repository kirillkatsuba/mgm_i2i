# Запуск бейзлайнов на текущем DIV2K multicrop-протоколе

Используются текущие реализации MSE U-Net v2, MSE+VGG perceptual, ESRGAN RRDBNet и I²SB с нашим time-conditioned U-Net. I²SB здесь — адаптация bridge-подхода к нашему масштабу сети и данным, а не исходный авторский DDPM++ backbone. MGM повторно не запускается.

Данные те же, что у завершённых MGM: `data/div2k_multicrop128`, протокол `random128-multicrop-v4`, 800/50/50 фото. Четыре новых случайных HR crop128 на фото за train-эпоху (3200 патчей), пять фиксированных validation/test-кропов на фото (250 патчей на split), LR32 = float average pooling ×4. Без resize/flips/rotations. Подготовку данных повторять не требуется, если этот root уже использовался для MGM. На submit проверяются manifest и hashes файлов. Подробности — [MULTICROP_PROTOCOL.md](MULTICROP_PROTOCOL.md).

## Рецепты

| Метод | Прогрев | Main | Патчи G/seed суммарно | Batch | Seeds |
|---|---|---|---:|---:|---|
| MSE | 50 эпох MSE | 80 эпох MSE | 416000 | 32 | 0/1/2 |
| MSE+perceptual | тот же MSE warmup | 80 эпох MSE+VGG | 416000 | 32 | 0/1/2 |
| ESRGAN | 50 эпох RRDB L1 | 80 эпох ESRGAN | 416000 | 32 | 0/1/2 |
| I²SB-adapted | нет | 130 эпох | 416000 | 32 | 0/1/2 |

Это сравнение по одинаковому числу обучающих патчей генераторов: у MGM тоже 50 warmup +80 main, но дополнительные 50 эпох обучения поля требуют отдельного учёта GPU-времени. Равные патчи не означают равные compute, число optimizer updates или одинаковую сходимость. У MGM warmup был batch16, у этих свежих baseline warmup batch32: одинакового warmup checkpoint и точного контроля начальной точки здесь нет. Прежние отменённые baseline warmup и завершённые MGM warmup не переиспользуются.

MSE/perceptual: U-Net120/240/480, около16.02M параметров; ESRGAN G около16.70M (+14.50M D при обучении); I²SB denoiser около17.29M. MSE U-Net использует ту же проекцию, что MGM. ESRGAN и I²SB оцениваются в нативном виде и с отдельно обозначенной финальной проекцией. Оптимизаторы и losses берутся из существующих baseline presets; MGM beta1=0.8 на них не переносится. VGG19 для loss — внешнее предобучение, генераторы обучаются с нуля/собственного warmup.

## Команды на кластере

```bash
cd ~/mgm_i2i
git pull --ff-only
conda activate mgm_i2i
export CONDA_ENV=mgm_i2i
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

# Проверить, что нет работающих старых baseline-задач.
squeue -u "$USER"

# Показать оба плана, затем отправить их в Slurm.
bash slurm/launch_baselines.sh
bash slurm/launch_baselines.sh --submit
```

Новые каталоги по умолчанию:

```text
cluster_runs/baselines_multi4_eval5_w50_g80_b32_r2
cluster_runs/i2sb_multi4_eval5_e130_b32_r2
```

Старые `baselines_multi4_eval5_w50_g80_b32` и `i2sb_multi4_eval5_e130_b32` могли уже иметь `submission.json` после отмены. Новый суффикс `r2` предотвращает конфликт файлов; это не отменяет старые jobs автоматически. Сначала проверить их состояние через `squeue`. Для самостоятельной новой серии можно задать `BASELINE_RUN_TAG=r3`; не использовать новый tag как способ повторного запуска уже работающей серии.

Launcher заранее проверяет наличие `submission.json` в обоих новых каталогах, а также формирует оба неизменяемых плана до первого submit. Повторное `--submit` после попытки отправки блокируется: смотреть сохранённые job IDs и продолжать partial jobs индивидуально. Это не атомарная транзакция Slurm: если второй submit завершится ошибкой, задания первого плана остаются в очереди. После такой ошибки не отправлять всё повторно вслепую.

Для нестандартных путей поддерживаются `MGM_RUNS_ROOT`, `MGM_DATA_ROOT`, `MGM_WEIGHTS_ROOT`, `PYTHON_BIN`. Для другой параллельности — `BASELINE_MAX_PARALLEL` (по умолчанию2 **на каждый план**, суммарно до4 GPU). Конфигурация scheduler и GPU request берётся из текущего `cluster_submit.py`: partition rocky, одна GPU на задачу, никаких `--mem`/`--mem-per-*`.

Если подготовка MGM выполнялась без VGG19, до submit на login node:

```bash
python scripts/cluster_prepare.py \
  --crop-mode multicrop --eval-crops-per-image 5 \
  --data-root data/div2k_multicrop128 \
  --archive-root data/div2k/downloads --weights-root weights
```

На GPU-узлах скачивания weights не требуется.

## Очередь и время

Первая серия: 6 warmup (MSE+RRDB ×3 seeds) →9 main (три метода ×3 seeds) →9 validation →CPU collect. MSE+perceptual и MSE используют общие три warmup. Каждый массив ограничен двумя одновременными задачами; между фазами барьер по успешному завершению всего массива.

Вторая серия независимо выполняет 3 I²SB main →12 validation (3 seeds ×NFE1/4/16/64) →CPU collect. Время I²SB-инференса оценивается отдельно для каждого NFE. Test автоматически не запускается.

Оценки времени на V10032 без очереди: первая серия около4–8 часов, вторая около2–4 часов параллельно; суммарно ориентировочно12–20 GPU·часов. Это экстраполяция предыдущих baseline-логов: новые ESRGAN/I²SB при batch32 ещё не измерены. Измеренные2 часа MGM main не следует напрямую переносить на эти архитектуры. Память batch32 у них тоже отдельно не подтверждена текущей MGM-серией.

Slurm limits: warmup60 минут, MSE/perceptual/ESRGAN main120, I²SB main180, каждая eval30, collect5 минут. Это запас времени, остановка обучения — по эпохам. Сумма всех GPU allocation limits двух планов **43.5 GPU·часа**; реальная работа должна завершаться раньше. При timeout/OOM проверить logs/status и возобновить отдельные jobs, не редактируя исходные планы и не создавая двух писателей одного checkpoint.

## Проверка результатов

```bash
squeue -u "$USER"
cat cluster_runs/baselines_multi4_eval5_w50_g80_b32_r2/submission.json
cat cluster_runs/i2sb_multi4_eval5_e130_b32_r2/submission.json

python scripts/cluster_collect.py \
  --plan cluster_runs/baselines_multi4_eval5_w50_g80_b32_r2/plan.json
python scripts/cluster_collect.py \
  --plan cluster_runs/i2sb_multi4_eval5_e130_b32_r2/plan.json
```

Ожидается validation coverage9/9 и12/12, `complete=true` всех training phases. Сравниваем с MGM по тем же validation-парам, clipped RGB PSNR/SSIM, LPIPS AlexNet, native crop128, агрегация сначала внутри фото, потом по50 фото. `val_summary.md/json`, `val_runs.csv` и per-run JSON/PNG сохраняются автоматически.
