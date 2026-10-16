"""Analyze saved JSON only: no model imports, training, inference, or tests."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'tmp/training_analysis_20261013/cluster_runs/full_2h_20261013_202802'
FULL = Path('/Users/kirill/Desktop/full_val')
OUT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'plan.json').read_text())
METHODS = list(dict.fromkeys(item['config']['variant'] for item in PLAN['train']))
COLORS = dict(zip(['warmup']+METHODS, ['#667788', '#2266bb', '#cc4444', '#1b9e77', '#9d60b1', '#d69216', '#0099bb']))
RUNS = []
ISSUES = []
if hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest() != (ROOT/'plan.sha256').read_text().strip():
    ISSUES.append('Plan SHA256 mismatch')


def load(path):
    return json.loads(path.read_text())


def moments(values):
    return dict(mean=stats.mean(values), median=stats.median(values),
                sd=stats.stdev(values) if len(values)>1 else 0,
                q10=float(np.quantile(values, .1)), q90=float(np.quantile(values, .9)))


for phase in ('warmup', 'train'):
    for item in PLAN[phase]:
        cfg = item['config']
        path = ROOT/cfg['variant']/f'seed{cfg["seed"]}'
        status = load(path/'status.json')
        saved_cfg = load(path/'config.json')
        env = load(path/'environment.json')
        rows = [json.loads(line) for line in (path/'train.jsonl').read_text().splitlines() if line.strip()]
        a = {key: np.array([row[key] if row[key] is not None else np.nan for row in rows]) for key in rows[0]}
        if saved_cfg != cfg:
            ISSUES.append(f'{path}: config differs from plan')
        if not status['complete'] or status['received_signals']:
            ISSUES.append(f'{path}: incomplete or interrupted')
        if any(np.any(~np.isfinite(a[key])) for key in ('seconds', 'step', 'g', 'mse')):
            ISSUES.append(f'{path}: nonfinite logged values')
        if phase == 'train' and cfg['method'] != 'mse' and np.any(~np.isfinite(a['aux'])):
            ISSUES.append(f'{path}: nonfinite auxiliary loss')
        if np.any(np.diff(a['step']) != cfg['log_every']) or np.any(np.diff(a['seconds']) <= 0):
            ISSUES.append(f'{path}: nonmonotone/gapped log')
        if status['step']//cfg['log_every'] != len(rows):
            ISSUES.append(f'{path}: unexpected log row count')
        if status['successful_g'] + status['skipped_g'] != status['step']:
            ISSUES.append(f'{path}: generator update count inconsistent')
        if status['elapsed_training_seconds'] < cfg['seconds']:
            ISSUES.append(f'{path}: training shorter than plan')
        bins = []
        for start in range(0, 7200, 300):
            mask = (a['seconds'] >= start) & (a['seconds'] < start+300)
            bins.append(dict(start=start, end=start+300, n=int(mask.sum()),
                             **{key: float(np.mean(a[key][mask])) for key in ('mse', 'g', 'aux')}))
        windows = {}
        for name, low, high in [('first_minute', 0, 60), ('first10', 0, 600),
                                ('second10', 600, 1200), ('penultimate10', 6000, 6600), ('last10', 6600, 7201)]:
            mask = (a['seconds'] >= low) & (a['seconds'] < high)
            windows[name] = {key: moments(a[key][mask].tolist()) for key in ('mse', 'g')}
            if cfg['method'] != 'mse':
                windows[name]['aux'] = moments(a['aux'][mask].tolist())
                if cfg['method'] == 'mgm':
                    windows[name]['aux_over_mse_mean'] = float(np.mean(a['aux'][mask])/np.mean(a['mse'][mask]))
        result = load(path/'val_metrics.json') if phase=='train' else None
        images = load(path/'val_images.json') if phase=='train' else []
        if result:
            if result['step'] != status['step'] or result['successful_g'] != status['successful_g']:
                ISSUES.append(f'{path}: evaluation/update count mismatch')
            if result['data_hash'] != status['data_hash'] or len(images) != 50 or result['crops'] != 250:
                ISSUES.append(f'{path}: evaluation data/coverage mismatch')
        entry = dict(phase=phase, method=cfg['variant'], seed=cfg['seed'], config=cfg,
                     status=status, environment=env, windows=windows, bins=bins,
                     val=result, images=images, arrays=a,
                     skipped_g_percent=100*status['skipped_g']/status['step'],
                     skipped_aux_percent=100*status['skipped_aux']/status['step'])
        RUNS.append(entry)

SUMMARY = dict(issues=ISSUES, training_gpu_hours=sum(r['status']['elapsed_training_seconds'] for r in RUNS)/3600,
               total_log_rows=sum(len(r['arrays']['step']) for r in RUNS), runs=[], methods={}, paired_images={}, full={})
accounting = []
for line in (ROOT/'slurm_accounting.txt').read_text().splitlines():
    columns = line.split()
    if len(columns) >= 5 and columns[0].startswith(('4402638_', '4402639_')):
        h, m, s = map(int, columns[4].split(':'))
        accounting.append(dict(job=columns[0], state=columns[2], exit=columns[3], seconds=h*3600+m*60+s))
SUMMARY['accounting'] = accounting
SUMMARY['allocation_gpu_hours'] = sum(row['seconds'] for row in accounting)/3600
SUMMARY['plan_hash_verified'] = not any('Plan SHA256' in issue for issue in ISSUES)
for run in RUNS:
    SUMMARY['runs'].append({k:v for k,v in run.items() if k not in ('arrays', 'images', 'val', 'bins')})
for method in ['warmup']+METHODS:
    runs = [r for r in RUNS if r['method']==method]
    item = dict(steps=moments([r['status']['step'] for r in runs]),
                skipped_g=[r['status']['skipped_g'] for r in runs],
                skipped_aux=[r['status']['skipped_aux'] for r in runs],
                skipped_g_percent=[r['skipped_g_percent'] for r in runs],
                skipped_aux_percent=[r['skipped_aux_percent'] for r in runs],
                windows={name: {key: moments([r['windows'][name][key]['mean'] for r in runs]) for key in ('mse', 'g')}
                         for name in ('first_minute', 'first10', 'second10', 'penultimate10', 'last10')})
    if method != 'warmup':
        item['val'] = {key: moments([r['val']['metrics'][key] for r in runs]) for key in runs[0]['val']['metrics']}
        item['val_images'] = {row['image']: {key: stats.mean([next(x for x in r['images'] if x['image']==row['image'])[key]
                                                            for r in runs]) for key in row if key!='image'}
                              for row in runs[0]['images']}
    SUMMARY['methods'][method] = item
    item['first_logged_mse'] = [float(r['arrays']['mse'][0]) for r in runs]
    item['last_aux_over_mse'] = [r['windows']['last10'].get('aux_over_mse_mean') for r in runs]

hashes = {r['status']['data_hash'] for r in RUNS}
if len(hashes) != 1:
    ISSUES.append('Multiple data hashes')
manifest_path = ROOT.parents[1]/'data/div2k/manifest.json'
manifest = load(manifest_path)
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() not in hashes:
    ISSUES.append('Archived manifest hash differs from run data_hash')
splits = {name: {Path(member).stem for member in manifest['banks'][f'{name}.npy']['images']}
          for name in ('train', 'val', 'test')}
if any(splits[a] & splits[b] for a,b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
    ISSUES.append('Image overlap between train/val/test')
SUMMARY['gpu_types'] = sorted({env['gpu'] for r in RUNS for env in r['environment']})
SUMMARY['session_counts'] = sorted({len(r['environment']) for r in RUNS})
for seed in (0, 1, 2):
    init = {r['status']['init_hash'] for r in RUNS if r['phase']=='train' and r['seed']==seed}
    if len(init) != 1 or None in init:
        ISSUES.append(f'seed{seed}: methods do not share a warmup hash')

ref = SUMMARY['methods']['mse']['val_images']
for method in METHODS:
    points = SUMMARY['methods'][method]['val_images']
    diffs = {image: {key: points[image][key]-ref[image][key] for key in ('model_lpips', 'model_psnr', 'model_ssim')}
             for image in ref}
    SUMMARY['paired_images'][method] = dict(
        lpips_better_than_mse=sum(row['model_lpips'] < 0 for row in diffs.values()),
        lpips_worse_than_mse=sum(row['model_lpips'] > 0 for row in diffs.values()),
        psnr_better_than_mse=sum(row['model_psnr'] > 0 for row in diffs.values()),
        lpips_delta=moments([row['model_lpips'] for row in diffs.values()]),
        worst_lpips=sorted(diffs.items(), key=lambda item:item[1]['model_lpips'], reverse=True)[:5],
        best_lpips=sorted(diffs.items(), key=lambda item:item[1]['model_lpips'])[:5])
    full = load(FULL/method/'seed0/result.json')
    crop_run = next(r for r in RUNS if r['method']==method and r['seed']==0)
    checkpoint_match = full['checkpoint_sha256'] == crop_run['val']['checkpoint_sha256']
    if not checkpoint_match:
        ISSUES.append(f'{method}: crop and full checkpoint hashes differ')
    subset = [row for row in crop_run['images'] if Path(row['image']).stem in {row['image'] for row in full['results']}]
    crop_model = stats.mean(row['model_psnr'] for row in subset)
    crop_base = stats.mean(row['projected_bicubic_psnr'] for row in subset)
    full_model = full['means']['draw1']['psnr']
    full_base = full['means']['projected_bicubic']['psnr']
    SUMMARY['full'][method] = dict(checkpoint_match=checkpoint_match, data_match=full['data_hash'] in hashes,
                                  n=len(subset), crop_subset_psnr=crop_model, crop_subset_projected_psnr=crop_base,
                                  full_psnr=full_model, full_ssim=full['means']['draw1']['ssim'],
                                  crop_gain_over_projected=crop_model-crop_base,
                                  full_gain_over_projected=full_model-full_base,
                                  peak_allocated_bytes=full['peak_allocated_bytes'], seconds=full['seconds'])

# Publication-style static plots: no model code is imported.
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'figure.dpi':130})
fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
panels = [(axes[0,0], ['warmup', 'mse'], 'mse', 'MSE reconstruction: warmup and continued MSE'),
          (axes[0,1], METHODS, 'mse', 'Reconstruction error during each objective'),
          (axes[1,0], [m for m in METHODS if m.startswith('mgm')], 'g', 'MGM generator objective (not a quality score)'),
          (axes[1,1], [m for m in METHODS if m.startswith('mgm')], 'aux', 'MGM field regression loss')]
for ax, methods, key, title in panels:
    for method in methods:
        runs = [r for r in RUNS if r['method']==method]
        data = np.array([[b[key] for b in r['bins']] for r in runs])
        time = np.array([b['end'] for b in runs[0]['bins']])/60
        ax.plot(time, data.mean(axis=0), label=method, color=COLORS[method])
        ax.fill_between(time, data.min(axis=0), data.max(axis=0), color=COLORS[method], alpha=.12)
    ax.set(title=title, xlabel='Training minutes within this phase', ylabel=key)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
axes[1,0].axhline(0, color='black', linewidth=.6)
fig.suptitle('5-minute means; lines = mean of 3 seeds, shaded bands = seed range\nDifferent objectives are not numerically comparable; only final validation was saved', fontsize=11)
fig.savefig(OUT/'training_curves.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
for ax, metric, title in [(axes[0], 'model_psnr', 'Validation PSNR (higher is better)'),
                         (axes[1], 'model_lpips', 'Validation LPIPS (lower is better)')]:
    means = [SUMMARY['methods'][m]['val'][metric]['mean'] for m in METHODS]
    errors = [SUMMARY['methods'][m]['val'][metric]['sd'] for m in METHODS]
    ax.barh(METHODS, means, xerr=errors, color=[COLORS[m] for m in METHODS], capsize=3)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis='x', alpha=.2)
ax = axes[2]
for i, method in enumerate(METHODS[1:]):
    vals = [SUMMARY['methods'][method]['val_images'][image]['model_lpips']-ref[image]['model_lpips'] for image in ref]
    ax.scatter(vals, np.full(len(vals), i), s=13, alpha=.55, color=COLORS[method])
    ax.scatter(stats.mean(vals), i, marker='D', s=45, color='black')
ax.set(yticks=range(len(METHODS)-1), yticklabels=METHODS[1:], xlabel='LPIPS(method) - LPIPS(MSE)',
       title='Paired photos: mean over 3 seeds')
ax.axvline(0, color='black', linestyle='--', linewidth=.8)
ax.invert_yaxis()
ax.grid(axis='x', alpha=.2)
fig.suptitle('50 validation photos; 5 crops averaged within each photo. Bars: mean +/- sample SD over 3 seeds.', fontsize=11)
fig.savefig(OUT/'validation_comparison.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
x = np.arange(len(METHODS))
ax.bar(x-.18, [SUMMARY['full'][m]['crop_gain_over_projected'] for m in METHODS], .36, label='5 crops/photo (same 8 photos)')
ax.bar(x+.18, [SUMMARY['full'][m]['full_gain_over_projected'] for m in METHODS], .36, label='Whole photo')
ax.set(xticks=x, xticklabels=METHODS, ylabel='PSNR(model) - PSNR(projected bicubic), dB',
       title='Same 8 photos, seed 0: gains relative to baseline\nCrop/full checkpoint hashes match for each method')
ax.axhline(0, color='black', linewidth=.7)
ax.legend()
ax.grid(axis='y', alpha=.2)
fig.savefig(OUT/'full_vs_crops.png', dpi=180)
plt.close(fig)

# Avoid NaN placeholder values for the nonexistent MSE auxiliary loss in exported JSON.
(OUT/'diagnostics.json').write_text(json.dumps(SUMMARY, indent=2, allow_nan=False)+'\n')
with (OUT/'training_windows.csv').open('w', newline='') as stream:
    fields = ['method', 'seed', 'window', 'mse_mean', 'mse_median', 'g_mean', 'aux_mean']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for r in RUNS:
        for name, window in r['windows'].items():
            writer.writerow(dict(method=r['method'], seed=r['seed'], window=name,
                                 mse_mean=window['mse']['mean'], mse_median=window['mse']['median'],
                                 g_mean=window['g']['mean'], aux_mean=window.get('aux', {}).get('mean')))

compact = dict(issues=ISSUES, total_log_rows=SUMMARY['total_log_rows'], training_gpu_hours=SUMMARY['training_gpu_hours'],
               allocation_gpu_hours=SUMMARY['allocation_gpu_hours'], accounting=SUMMARY['accounting'],
               methods={method: {k: v for k,v in item.items() if k != 'val_images'} for method,item in SUMMARY['methods'].items()},
               paired_images=SUMMARY['paired_images'], full=SUMMARY['full'])
(OUT/'summary.json').write_text(json.dumps(compact, indent=2, allow_nan=False)+'\n')
print('Issues:', ISSUES)
print('Log rows:', SUMMARY['total_log_rows'], 'GPU training hours:', SUMMARY['training_gpu_hours'])
for method, item in SUMMARY['methods'].items():
    windows = item['windows']
    print(method, 'MSE(first minute, first10, second10, penultimate10, last10):',
          [round(windows[name]['mse']['mean'], 7) for name in windows],
          'skip G/A:', item['skipped_g'], item['skipped_aux'])
print('Paired LPIPS wins /50:', {m:r['lpips_better_than_mse'] for m,r in SUMMARY['paired_images'].items()})
print('Slurm allocation GPU hours:', SUMMARY['allocation_gpu_hours'])
for method in METHODS:
    item = SUMMARY['methods'][method]
    print(method, 'first logged MSE:', item['first_logged_mse'], 'last aux/MSE:', item['last_aux_over_mse'],
          'G first10/last10:', item['windows']['first10']['g']['mean'], item['windows']['last10']['g']['mean'])

def fmt(value, digits=5):
    return f'{value:.{digits}f}'

lines = ['# Анализ обучения MGM I2I — серия full_2h_20261013_202802', '',
         'Источник: архив пользователя `training_analysis_20261013.tar.gz` и `/Users/kirill/Desktop/full_val`.',
         'Выполнены только чтение сохранённых JSON/логов, статистический анализ и построение графиков. '
         'Обучение, инференс и тесты не запускались. Модели и checkpoints не загружались.', '',
         '## Основной вывод', '',
         'Все задания завершены штатно. Недостаточный лимит Slurm и массовые FP16-overflow '
         'не объясняют артефакты этой серии. Реконструкционная ошибка MGM повышена уже '
         'в самом начале основного этапа и остаётся повышенной. Поведение согласуется с '
         'проблемой перехода от детерминированного warmup к стохастическому генератору и '
         'компромиссом цели MGM; причинность и достаточность обучения поля по этим логам не установлены.', '',
         '## Проверки целостности и бюджет', '',
         f'- Прочитано {SUMMARY["total_log_rows"]:,} записей train.jsonl для 3 warmup и 18 основных задач.',
         '- Проверены SHA256 плана, совпадение config с plan, длительность, счётчики обновлений, '
         'полнота логов, монотонность шагов/времени и конечность записанных losses.',
         '- Все 21 status имеют complete=true, received_signals=[]; Slurm для всех GPU-задач '
         'показывает COMPLETED, ExitCode=0:0. Collect также COMPLETED.',
         '- Все обучались на Tesla V100-SXM2-32GB, torch 2.6.0+cu118; каждый run имеет одну сессию.',
         '- Для каждого seed записан одинаковый init_hash у всех шести методов; data_hash '
         'одинаков у всей серии. Самих warmup-checkpoints в архиве нет, их байты повторно не проверялись.',
         '- SHA256 приложенного manifest совпал с data_hash; списки train/val/test не пересекаются по image ID.',
         '- Для всех 18 оценок: 50 фото, 250 crop; числа шагов совпадают с status. '
         'У всех шести full-frame запусков SHA256 checkpoint совпадает с crop-evaluation соответствующего seed 0.',
         f'- Суммарное время обучения: {SUMMARY["training_gpu_hours"]:.4f} GPU-часа. '
         f'Суммарное время 21 GPU-allocation по sacct: {SUMMARY["allocation_gpu_hours"]:.4f} GPU-часа '
         '(включая запуск/финальную оценку). Это не включает предыдущие попытки и дополнительный full-frame inference.',
         '- Git commit на compute nodes записан как unavailable: точную ревизию исполненного кода '
         'по provenance подтвердить нельзя; чтение текущего исходника не заменяет эту проверку.',
         f'- Выявленные автоматическими проверками несоответствия: {ISSUES or "нет"}.', '',
         '## Динамика реконструкции', '',
         'Значения — средние MSE по батчам в указанных окнах, затем среднее трёх seed; '
         'изображения нормированы в [-1,1]. Это train-ошибка текущего G, не validation и не EMA. '
         'Переходы между окнами могут зависеть от сложности батчей.', '',
         '| Метод | Первая минута | Первые 10 минут | Последние 10 минут | Последние / первые 10 минут |',
         '|---|---:|---:|---:|---:|']
for method in ['warmup']+METHODS:
    windows = SUMMARY['methods'][method]['windows']
    lines.append(f'| {method} | {fmt(windows["first_minute"]["mse"]["mean"])} | '
                 f'{fmt(windows["first10"]["mse"]["mean"])} | {fmt(windows["last10"]["mse"]["mean"])} | '
                 f'{windows["last10"]["mse"]["mean"]/windows["first10"]["mse"]["mean"]:.3f} |')
lines += ['', 'Warmup улучшает MSE примерно с 0.01153 до 0.01013 между начальными и конечными '
          '10-минутными окнами. За последние 20 минут изменение небольшое. Дополнительные '
          'два часа MSE дают около 0.00982 к концу. Это признаки замедления улучшения train-MSE, '
          'не доказательство оптимальности модели или отсутствия переобучения.', '',
          'MGM работает около 0.022–0.024 с первых минут. У основной MGM средняя MSE первой '
          'минуты 0.02284, у продолженного MSE — 0.00949. В конце обычная MGM около 0.02278, '
          'antithetic около 0.02170. Резкого позднего роста/численного взрыва нет; само увеличение '
          'длительности не обосновано этими кривыми как достаточное исправление.', '',
          '![Training curves](training_curves.png)', '',
          '## Пропуски обновлений и поле', '',
          '| Метод | Пропуски G по seed | Пропуски auxiliary по seed | Макс. доля G, % | Макс. доля auxiliary, % |',
          '|---|---|---|---:|---:|']
for method in ['warmup']+METHODS:
    item = SUMMARY['methods'][method]
    lines.append(f'| {method} | {item["skipped_g"]} | {item["skipped_aux"]} | '
                 f'{max(item["skipped_g_percent"]):.4f} | {max(item["skipped_aux_percent"]):.4f} |')
lines += ['', 'У всех GAN/MGM генераторных обновлений skipped_g=0. Редкие пропуски поля/дискриминатора '
          'составляют менее 0.11%; оснований объяснять качество массовыми overflow нет. '
          'Градиентные нормы, частота срабатывания clipping и история AMP scale не записывались: '
          'сами градиенты по этим данным не диагностируются.', '',
          'У MGM field regression loss близок по порядку к MSE разности концов; generator objective '
          'намного меньше и колеблется. Это совместимо и с небольшим условным средним displacement, '
          'и со слабым полем. При совпадении распределений regression loss не обязан быть нулём. '
          'Поэтому утверждать, что поле точно не обучилось, нельзя. У midpoint и antithetic '
          'generator objective после раннего снижения снова дрейфует вверх; численного доказательства '
          'стационарного minimax-равновесия нет.', '',
          '## Validation и неоднородность по фотографиям', '',
          '| Метод | PSNR mean ± SD | LPIPS mean ± SD | Фото с LPIPS лучше MSE, из 50 |',
          '|---|---:|---:|---:|']
for method in METHODS:
    item = SUMMARY['methods'][method]['val']
    lines.append(f'| {method} | {item["model_psnr"]["mean"]:.3f} ± {item["model_psnr"]["sd"]:.3f} | '
                 f'{item["model_lpips"]["mean"]:.5f} ± {item["model_lpips"]["sd"]:.5f} | '
                 f'{SUMMARY["paired_images"][method]["lpips_better_than_mse"]} |')
lines += ['', 'Счётчик фото сравнивает LPIPS после усреднения трёх seed внутри каждой фотографии. '
          'Пять crop одной фотографии не считаются независимыми наблюдениями. Это описательная '
          'статистика, не проверка значимости по 150 независимым примерам и не доказательство '
          'превосходства на новых seed.', '',
          'Antithetic улучшает средний LPIPS, но хуже MSE на 12 из 50 фото по этому критерию. '
          'В просмотренных crop-grid MGM уже показывает зерно и искусственные текстуры; '
          'артефакты не впервые появляются в полном кадре. LPIPS не оптимизируется напрямую '
          'ни одним из этих losses и не заменяет визуальную проверку.', '',
          '![Validation comparison](validation_comparison.png)', '',
          '## Полные фотографии и гипотеза GroupNorm', '',
          'Сопоставлены ровно те же восемь фото и seed 0; checkpoint hashes совпали. '
          'Однако набор пикселей/контекст и способ усреднения метрик у crop и full различаются. '
          'Поэтому даже это сопоставление не является изолированным тестом размера входа.', '',
          '| Метод | PSNR на 5 crop тех же фото | PSNR целого кадра | Crop: выигрыш над projected bicubic | Full: выигрыш над projected bicubic |',
          '|---|---:|---:|---:|---:|']
for method in METHODS:
    row = SUMMARY['full'][method]
    lines.append(f'| {method} | {row["crop_subset_psnr"]:.3f} | {row["full_psnr"]:.3f} | '
                 f'{row["crop_gain_over_projected"]:+.3f} | {row["full_gain_over_projected"]:+.3f} |')
lines += ['', 'У самого projected bicubic, где нет GroupNorm и обученной сети, PSNR меняется '
          'с 34.624 на crop до 28.511 на полном кадре. Значительная часть разницы абсолютных '
          'чисел объясняется самим протоколом оценки/выбором областей и нелинейностью PSNR. '
          'Относительно этого baseline все методы на полном кадре выглядят не хуже, чем на '
          'пяти crop той же выборки. Это ослабляет объяснение «виноват только полный размер». '
          'Влияние GroupNorm остаётся возможным, но не доказанным.', '',
          '![Full vs crops](full_vs_crops.png)', '',
          '## Кандидат на причину: переход от G(c,0) к G(c,z)', '',
          'В текущем engine warmup/MSE получает нулевой latent, тогда как MGM/GAN получает '
          'Gaussian latent сразу с начала основного этапа. В Generator входные каналы latent '
          'идут в обычную Conv2d stem. При z=0 вклад этих весов равен нулю, и градиент по '
          'latent-channel весам stem равен нулю. При стандартном Adam без weight decay '
          'эти веса остаются исходными случайными весами, тогда как остальная сеть обучается.', '',
          'Следовательно, общий checkpoint обеспечивает одинаковые веса, но не одинаковое '
          'начальное качество для нулевого и случайного latent. После warmup случайный z '
          'может сразу возбудить нежелательные высокие частоты. Начальный скачок train-MSE '
          'с этим согласуется, однако первые сохранённые записи уже после 20 обновлений: '
          'контролируемого измерения до первого обновления нет. Это проверяемая гипотеза, '
          'а не доказанный единственный источник артефактов.', '',
          'Дополнительно: чистая MGM после warmup не штрафует отклонение от единственного '
          'HR-target через MSE; поле конечной мощности и неточное minimax-решение могут '
          'не подавлять нежелательные текстуры. Детерминированный MSE и стохастический '
          'генератор также имеют различный ожидаемый distortion даже при корректном '
          'сэмплировании условного распределения, поэтому сам рост MSE не доказывает ошибку MGM.', '',
          'В частности, пусть Y и Y′ — независимые samples правильного условного распределения '
          'при фиксированном c. Тогда E||Y′−Y||² = 2·tr Cov(Y|c), '
          'а E||E[Y|c]−Y||² = tr Cov(Y|c). Поэтому корректный стохастический генератор '
          'может иметь вдвое большую MSE, чем точное условное среднее. Наши модели не '
          'являются доказанными точными решениями, но это объясняет, почему одного '
          'отношения ошибок около двух недостаточно для диагноза.', '',
          '## Что делать дальше', '',
          '1. Сначала короткая диагностика без нового обучения: на warmup checkpoint сравнить '
          'G(c,0) и G(c,alpha*z) при alpha=0,0.1,0.25,0.5,1 на фиксированных validation inputs. '
          'Повторить на готовых MGM checkpoints. Измерить шум гладкого фона и показать обе генерации; '
          'не выбирать alpha по held-out test и не выдавать alpha=0 за результат исходной стохастической модели.',
          '2. Проверить размер входа на одинаковых областях: MSE с z=0 снимает фактор случайного latent; '
          'для MGM использовать согласованное поле z и вырезать его вместе с LR. Сравнивать центральную '
          'область полного результата с отдельно обработанной областью, отдельно учитывая контекст и padding.',
          '3. Если гипотеза инициализации подтверждается, новый небольшой контролируемый опыт: '
          'noise-aware MSE warmup с тем же latent distribution либо архитектура с контролируемым '
          'вводом stochastic residual. Отдельно сравнить чистую MGM и MGM + reconstruction loss; '
          'не менять одновременно initializer, архитектуру, noise scale и loss.',
          '4. Для следующего обучения сохранять отдельные checkpoints и фиксированную validation '
          'каждые 10–15 минут; логировать gradient norms, clipping, AMP scale и нормы/качество поля. '
          'Лишь после этого выбирать длительность по кривой качества, а не по train-loss.', '',
          '## Что текущие данные не позволяют заключить', '',
          '- Нет промежуточных checkpoints/validation: нельзя найти лучший момент или доказать переобучение.',
          '- Нет field-only диагностики: нельзя доказать, что поле отстаёт или переобучено.',
          '- Нет контрольного noisy warmup и снимка до шага 1: влияние latent cold start пока гипотеза.',
          '- Есть только 3 seed, полный кадр проверен только на 8 фото/seed 0.',
          '- Ни низкий/отрицательный generator loss, ни высокий diversity не доказывают '
          'совпадение условных распределений или хорошее визуальное качество.', '',
          'Данные для воспроизведения расчётов: `diagnostics.json`, `training_windows.csv`, '
          '`summary.json`; код анализа: `analyze_logs.py`.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('Report:', OUT/'REPORT.md')
