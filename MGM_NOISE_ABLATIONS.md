# MGM v1/v2: недостающие варианты шума

21 октября 2026. Подготовлены код и CPU-проверки. Новые GPU-задачи здесь не запускались: пользователь выполняет команды на кластере.

**Обновление: обучение → validation → test одной командой.** По просьбе пользователя добавлен `--with-test`. Эпохи сохраняются: MSE50 из готовых weights → field50 → MGM80. Для полного выбранного набора после `git pull --ff-only`:

```bash
cd ~/mgm_i2i
conda activate mgm_i2i
export CONDA_ENV=mgm_i2i
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

bash slurm/launch_mgm_ablation.sh --suite paper --with-test --max-parallel 2
bash slurm/launch_mgm_ablation.sh --suite paper --with-test --max-parallel 2 --submit
```

Default run root: `cluster_runs/mgm_noise_paper_val_test_20261021`. Будет 24 field jobs → 24 main jobs → 30 val jobs → 30 test jobs → CPU collector двух splits. Test использует те же final EMA checkpoints с зафиксированными настройками. Test-картинки находятся в отдельном подкаталоге, не перезаписывая val. Итоги: `val_summary.md`/`test_summary.md`, соответствующие CSV/JSON. Allocation upper bound с двумя evaluation arrays —162 GPU-часа; ожидаемое обучение остаётся ~60.2 GPU-часа, evaluation оплачивается дополнительно. GPU concurrency cap2 относится ко всему плану. Если нужен только validation, флаг `--with-test` не указываем; остальная инструкция ниже описывает этот режим.

## Соответствие статье

[MGM, §5.1–5.2 и Table 1](https://arxiv.org/html/2605.29920v2#S5.T1) рассматривает шум интерполянта, critic-only noise и midpoint-only. Три математические конструкции статьи — midpoint, time-integrated с hidden flip, generalized со stochastic interpolant. Наши **v1/v2 — архитектуры реализации**, а не эти конструкции.

Смесь: Q = (1-w)G(C,Z) + wY + sqrt(lambda*t*(1-t))*epsilon, где w равен t или 1-t по скрытому flip. Поле получает базовое t. `noise_f` — lambda при обновлении поля; `noise_g` — lambda при обновлении генератора. Во всех режимах Gaussian latent Z сохраняется: «без шума» означает отсутствие **шума интерполянта**, а не детерминированный генератор.

| Режим | CLI variant | noise_f | noise_g | Время |
|---|---|---:|---:|---|
| Без шума | `mgm_no_noise` | 0 | 0 | U[0,.5] |
| Matched .01 | `mgm` | .01 | .01 | U[0,.5] |
| Critic-only .01 | `mgm_critic_noise` | .01 | 0 | U[0,.5] |
| Matched .05 | `mgm_matched_noise_005` | .05 | .05 | U[0,.5] |
| Midpoint, critic-only .01 | `mgm_midpoint_critic_noise` | .01 | 0 | только .5 |

Коэффициент под корнем; при t=.5 std=.05 для lambda=.01 и ~.1118 для lambda=.05. В статье lambda обозначена sigma; это не постоянный std sigma. Matched/no-noise имеют общую схему наблюдений двух игроков. Critic-only — эмпирическая модификация с разными схемами. В noisy midpoint при том же lambda также меняется средняя дисперсия шума: сравнение не изолирует только интегрирование времени. Для этого дополнительно нужен noiseless midpoint против noiseless time-integrated.

## Матрица и объём

Только текущий multicrop-v4 и 50/50/80; старые center128-серии не включаем.

| Архитектура | Без шума | Matched .01 | Critic .01 | Matched .05 | Midpoint critic .01 |
|---|---|---|---|---|---|
| v1 | новое | уже есть | новое | расширение | расширение |
| v2 | новое | новое | уже есть | расширение | расширение |

- **core:** первые три столбца, 12 новых обучений (4 настройки × 3 seeds). Переоценка шести старых controls, всего 18 validation evaluations.
- **paper:** также два последних столбца, 24 новых обучения и 30 evaluations. Выбранные режимы по мотивам Table 1, не полный авторский sweep.
- **`--retrain-controls`:** пересчитать также прежние v1 matched/v2 critic текущим кодом. Core тогда 18 обучений, paper 30. Это предпочтительно для строгого сравнения в одной ревизии: у старых compute-node запусков Git provenance был unavailable.

Core и paper — альтернативные планы: запуск обоих целиком дублирует core. Для расширения после core см. отдельные команды ниже.

MSE initializers берём из `cluster_runs/sr_v1_multi4_eval5_w50_f50_g80_b32_beta08` и соответствующего `sr_v2_...`. Каждый режим начинает с **MSE weights**, не с финального MGM. Проверяем подписи source plans и полное совпадение конфигураций. При submit отсутствие нужных checkpoint-файлов останавливает запуск до первого sbatch. При загрузке проверяются checkpoint metadata; данные и LPIPS-веса проверяются штатным verifier.

Фиксируем width120/240/480, seeds0/1/2, projection, split800/50/50, 4 новых train crop128/фото/эпоху, 5 фиксированных val crop128/фото, float pooling×4. MSE50 уже выполнен. На новый режим: перенос признаков G в поле, field50 (5000 шагов), main80 (8000 пар обновлений), batch32 без накопления, Adam lr_G/F=1e-5 и betas=(.8,.999), EMA .995, reconstruction_weight=0, AMP/clip10. Поле пересчитывается отдельно даже при совпадающем noise_f; это повторная стоимость, но самодостаточные запуски. Начальный G одинаков внутри architecture/seed.

Validation: EMA/FP32/batch1, одинаковые 250 кропов/50 фото, фиксированные draws, PSNR/SSIM/LPIPS/diversity/consistency/latency, усреднение кропов внутри фото, без best-of-N. С `--with-test` после validation оцениваем отдельный test: те же 50 фото ×5 фиксированных кропов и тот же evaluator; настройки по test не подбираем.

## Запуск

Сначала доставь новые изменения кода в `~/mgm_i2i`. Команды предполагают сохранённые прежние run roots.

```bash
cd ~/mgm_i2i
conda activate mgm_i2i
export CONDA_ENV=mgm_i2i
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

# Только план, без отправки.
bash slurm/launch_mgm_ablation.sh --suite core --max-parallel 2
# Отправка того же плана.
bash slurm/launch_mgm_ablation.sh --suite core --max-parallel 2 --submit
squeue -u "$USER"
```

Для расширенного набора **вместо core** замени `--suite core` на `--suite paper` в обоих вызовах. Для свежих controls добавь `--retrain-controls` к обоим вызовам. Разные suite/retrain получают отдельные default run roots. После dry run изменение параметров требует нового `--run-root`.

Пути к source можно переопределить `--source-v1 ПУТЬ --source-v2 ПУТЬ`. Data/weights/CPU должны соответствовать source warmup: несовместимые configs отклоняются.

Единый core-план `cluster_runs/mgm_noise_core_20261021/plan.json`: field array0–11 → train array0–11 → eval array0–17 → CPU collect. Следующая GPU-фаза ждёт успешного завершения **всего** предыдущего array. В каждой фазе максимум2 GPU-задачи **на обе архитектуры вместе**; другие отдельные планы этим лимитом не ограничены. Каждая задача на одной GPU. Параметров mem нет. Allocation limits: field90/main240/eval30 минут, это запас, не критерий остановки.

Соответствие indices/architecture/variant/seed сохранено в `ablation.matrix`, job IDs — `submission.json`. Повторный submit блокируется. После failure сначала продолжи задачу с `--resume-if-exists`, затем заново поставь зависимости; [инструкция](MGM_FIELD_WARMUP_DIAGNOSTICS.md#5-остановка-по-эпохам-и-продолжение). Indices бери из нового плана.

## Время

Экстраполяция прошлых V100-SXM2-32GB: field~32–33 мин и main~118–119 мин на настройку/seed.

| План | Новых обучений | Training GPU-часы | Идеальный training wall time при 2 GPU |
|---|---:|---:|---:|
| core | 12 | ~30.1 | ~15.1 ч |
| paper | 24 | ~60.2 | ~30.1 ч |
| core + fresh controls | 18 | ~45.2 | ~22.6 ч |
| paper + fresh controls | 30 | ~75.3 | ~37.7 ч |

Без evaluation, startup, очередей, повторов и уже оплаченного MSE warmup. Noise-режим может изменить скорость/устойчивость. `allocation_gpu_hours_upper_bound` — сумма заказанных лимитов. Унаследованное `reused_warmup_training_gpu_hours` использует прежние **minute hints**, не измеренную стоимость epoch-stopped warmup; план явно отмечает это. Фактическую стоимость восстанавливаем из status/checkpoints.

## Если core уже поставлен: только расширение

Не запускай целиком paper поверх core. Добавь только два режима для каждой архитектуры:

```bash
for V in 1 2; do
  python scripts/cluster_submit.py \
    --run-root "cluster_runs/mgm_noise_extra_v${V}_20261021" \
    --reuse-warmup-from "cluster_runs/sr_v${V}_multi4_eval5_w50_f50_g80_b32_beta08" \
    --architecture-version "$V" --width 120 \
    --variants mgm_matched_noise_005 mgm_midpoint_critic_noise --seeds 0 1 2 \
    --data-root data/div2k_multicrop128 --data-protocol random128-multicrop-v4 \
    --train-crops-per-image 4 --eval-crops-per-image 5 \
    --warmup-epochs 50 --field-warmup-epochs 50 --train-epochs 80 \
    --field-initialization warmup_features --mgm-optimizer authors --mgm-beta1 .8 \
    --mgm-microbatch-size 32 --grad-accumulation 1 \
    --field-warmup-time-limit-minutes 90 --train-time-limit-minutes 240 \
    --eval-minutes 30 --max-parallel 1
done
```

Dry run; после просмотра добавь `--submit` в каждый вызов. Здесь два отдельных плана с лимитом1: вместе до2 GPU **плюс** задачи уже запущенного core.

## Результаты

Collector создаёт `val_runs.csv`, `val_summary.json`, `val_summary.md`, отдельно группируя architecture/variant. Сравним режимы внутри каждой архитектуры и v1/v2 в одном режиме. Это не отдельная абляция tanh и time embedding: эти детали v1/v2 по-прежнему меняются вместе. Проверяем mean±SD трёх seeds, парные изменения по фото, field/zero baseline, gradients/skips и гладкие/текстурные области. Низкий MGM objective не доказывает хорошее качество.

Для передачи без весов (default core; для других планов замени root/имя архива):

```bash
tar --exclude='*.pt' --exclude='*.pth' -czf mgm_noise_core_results.tar.gz \
  cluster_runs/mgm_noise_core_20261021
```

Новый архив содержит также повторные evaluations старых controls. Их исходные результаты уже сохранены локально.
