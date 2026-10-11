# Карта кода: как прочитать прототип

Начинай с `src/mgm_i2i/losses.py`, затем `models.py`, затем `scripts/train.py`.
Все пути ниже заданы относительно корня проекта. Исходный код входит в пакет.

## Один шаг MGM

1. `data.batch` возвращает LR-условие `c` и его HR-target `y`.
2. `latent(y)` создаёт независимый шум для генератора. `g(c,z)` возвращает HR-изображение.
3. `random_bridge` выбирает базовое время, скрытый flip и шум интерполянта.
4. `observation` смешивает **HR fake и HR real**; размерности одинаковые.
5. `f(c,t,mix)` выдаёт вектор с размерностью HR. Target регрессии — `y-fake`.
6. На шаге поля используем `fake.detach()` и минимизируем MSE поля.
7. На шаге генератора параметры поля имеют `requires_grad=False`; градиент через его вход остаётся.
8. Генератор минимизирует вариационный loss. Обновляем EMA и сохраняем checkpoint.

Входное условие `c` для fake и real одинаковое. Для pix2pix-style conditional GAN используется scalar/patch discriminator; для MGM — векторное поле.

## Файлы и ответственность

| Файл | Назначение |
|---|---|
| `src/mgm_i2i/losses.py` | Небольшое математическое ядро: интерполянт, скрытый flip, две потери |
| `src/mgm_i2i/models.py` | MLP для toy, residual CNN для изображений, field/discriminator, точная проекция |
| `src/mgm_i2i/data.py` | Известное двухмодальное распределение, локальный CIFAR, разделение train/val/test |
| `src/mgm_i2i/evaluate.py` | Conditional W1, PSNR, SSIM, LPIPS, consistency, diversity и картинки |
| `scripts/train.py` | CLI/config, warmup, чередование игроков, EMA, логи, init/resume |
| `scripts/evaluate.py` | Независимая оценка сохранённой EMA на val или test |
| `scripts/toy_suite.py` | Три seeds и небольшая матрица абляций |
| `scripts/compare.py` | Сводный CSV и график для указанных run-папок |
| `scripts/prepare_data.py` | Проверка checksum CIFAR и преобразование в локальные NumPy arrays |
| `scripts/check_offline.py` | Проверка моделей и LPIPS при запрещённых socket-соединениях |
| `scripts/setup_offline.sh` | Восстановление окружения из wheel-файлов без PyPI |
| `scripts/build_guide.py` | Генерация HTML из Markdown, локальный JS/CSS/шрифты для формул |
| `tests/test_mgm.py` | Проверки симметрии, вариационного тождества, градиента и проекции |

## Что менять сначала

На toy меняй время/шум и число critic updates через CLI. На CIFAR сперва измерь скорость и выбери число шагов, `width` и `train_limit`. Для честного сравнения сохраняй одинаковую generator architecture и одинаковые данные. Каждый вариант — отдельная папка output, один config и один seed.

Не меняй знак target при flip, не передавай скрытую ветку полю, не ставь `no_grad()` вокруг поля на generator step и не заменяй generator loss регрессионной ошибкой поля. Эти изменения меняют метод.

## Как устроены результаты

`config.json` фиксирует настройки, `environment.json` — устройство и число параметров, `train.csv` — потери и время. `latest.pt` содержит generator, field/discriminator, EMA, optimizers и RNG. `metrics.json` относится к сохранённой EMA. `samples.png` или `toy_distribution.png` позволяют посмотреть результат.

MGM-loss и GAN-loss нельзя сравнивать по абсолютным значениям; у них разные смыслы. Сравнивай outputs на одних данных и стоимость обучения. Проверочные картинки/таблицы уже лежат в `runs/verified_*`.
