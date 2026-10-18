# Общий протокол сравнения SR ×4

**Обновление19 октября2026:** следующий протокол — [random128-multicrop-v4](../MULTICROP_PROTOCOL.md):4 разных random train-crop128 на фото в каждой эпохе,5 фиксированных случайных val/test-crop128 на фото. Split800/50/50, pooling×4 и отсутствие spatial augmentation сохраняются. Новый sampler сбалансирован по фото; одна эпоха3200 патчей.Согласованы50/50/80 эпох warmup/field/main:160000/160000/256000 патчей. Остальной текст ниже документирует предыдущий `random128-v3` с center128 evaluation.

Актуальное решение пользователя от 16 октября 2026 заменяет промежуточный выбор разных native preprocessing ESRGAN/I²SB. Для основной серии все пять методов используют одинаковые HR/LR пары. Исходные preprocessing baseline сохраняются только как справочные utilities и возможный отдельный эксперимент.

## Принятые условия

- **Train:** при каждом обращении равновероятно выбираем одну из 800 исходных train-фотографий и равновероятные допустимые координаты HR crop **128×128**, без resize. Используется вся площадь исходного фото, а не фиксированный банк ранее нарезанных patches.
- **Validation/test:** по одному фиксированному центральному HR crop 128×128 исходных пикселей: `top=(H-128)//2`, `left=(W-128)//2`.
- Повороты и отражения не добавляем. Seed и глобальный номер примера определяют фотографию и координаты; у методов с одним seed совпадает последовательность примеров при одинаковом sample index. При разной скорости обучения число просмотренных примеров может различаться.
- LR **32×32** получаем через `avg_pool2d(HR,4)` в float, без PNG-округления. Это общая pooling-деградация, включая ESRGAN.
- Validation/test используют идентичные пары для всех моделей и фиксированный порядок. Test не участвует в выборе checkpoint, loss weights, noise schedule и NFE.
- Архитектуры: наши U-Net **120/240/480**, ESRGAN — стандартный RRDB23. G MGM/MSE — 16 023 003 параметра, I²SB U-Net-120 — 17 289 963, ESRGAN G — 16 697 987.
- Готовые внешние SR/GAN/diffusion checkpoints не используем. Замороженные ImageNet VGG19 для perceptual loss и AlexNet для LPIPS раскрываем отдельно. Reconstruction pretraining/warmup на собственном train split допускается и учитывается в полном бюджете.

## Представление общей пары для каждого метода

| Метод | Диапазон SR-сети | Вход и цель |
|---|---|---|
| MSE / MSE+perceptual | [-1,1] | LR32 и нулевой latent; цель HR128 |
| MGM v2 | [-1,1] | LR32 и случайный latent; field сравнивает fake/real HR при том же LR |
| ESRGAN | [0,1] | LR32 непосредственно в RRDBNet; цель тот же HR128 |
| I²SB U-Net-120 | [-1,1] | degraded endpoint = nearest(LR32) до 128; bridge state и time в denoiser |

Преобразование [0,1] ↔ [-1,1] меняет представление значений, но не изображение, crop или оператор SR. Pixel loss weights нельзя механически сравнивать между диапазонами без учёта масштаба. VGG получает [0,1], затем ImageNet mean/std; для MSE+perceptual выполняется перевод из [-1,1]. Для метрик все результаты приводятся к одному диапазону и общим правилам clipping, границ и усреднения.

I²SB теперь работает **32→128**, без Resize256, train flip и ADM initialization. Сохраняются bridge objective/sampler; это адаптация метода к общей задаче, не воспроизведение опубликованного ImageNet-256 эксперимента. ESRGAN использует pooling-пары вместо исходных bicubic; это также явно раскрываем.

Основной MGM сохраняет hard pooling projection. Для ESRGAN/I²SB предусматриваем native выход и вариант с той же финальной pooling projection; проекцию внутри reverse chain не добавляем автоматически. Baseline loops и единый evaluator интегрированы; записываются native и projected метрики. GPU-пилот ещё не выполнялся.

## Разбиение данных: действующее решение и варианты увеличения

**Действующее разбиение пока не изменено:**
- Train: официальные DIV2K 0001–0800, 800 исходных фотографий, из которых онлайн выбираются разные patches.
- Validation: 50 фотографий из официальных 0801–0900.
- Held-out test: остальные 50 фотографий из 0801–0900.
- Сохраняем существующий image-level split seed=123. Этот test — наш held-out split, а не официальный DIV2K test.

[Стандартный DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) содержит 800 train, 100 validation и 100 test; HR официального test не предоставляются участникам. Поэтому 800/50/50 использует весь открытый официальный train, а не урезанную его часть.

**Предложение, ещё не принято:** расширить train до **DF2K = DIV2K train 800 + Flickr2K 2650 = 3450 фотографий**, оставить прежние DIV2K val50/test50. Это увеличит число исходных train-фотографий с 800 до 3450 (×4.3125); число возможных random crops не фиксируем. Такой состав train встречается в [официальном SwinIR](https://github.com/JingyunLiang/SwinIR) и [ESRGAN](https://github.com/xinntao/ESRGAN). Время на epoch возрастает; при фиксированных GPU-часах число просмотренных примеров/эпох будет другим, поэтому достаточность бюджета нужно проверять пилотом. Не обещаем пропорционального роста качества или wall time.

**Дополнительная независимая оценка, ещё не принята:** BSD100 и Urban100, обрабатываемые тем же center128/pooling для всех методов; метрики каждого датасета показывать отдельно. Это расширяет test, а не train. Их готовые bicubic LR в pooling-задаче не используем; опубликованные bicubic-метрики напрямую не сравниваем. Нужно проверить дубликаты/пересечения источников с train перед использованием.

Перенос 50–100 validation-фото в train даст небольшое увеличение и лишит прежней независимой оценки; без нового evaluation-набора не делаем. Пользователь отдельно утвердил random train-patches 128: это уже реализованный новый протокол; добавление Flickr2K пока не принято.

## Состояние реализации

- Новый protocol — **random128-v3**, default data root `data/div2k_random128`. Подготовка сохраняет полные RGB train images в `train_images/NNNN.npy` и фиксированные центральные val/test banks. LR формируется после crop при обучении. Нужны новый data root, run root и warmup.
- `RandomImageCrops` читает NPY через mmap и копирует только выбранный crop; PNG не декодируются на каждом шаге. Sample sequence определяется seed/index и не зависит от prefetch, порядка workers и resume.
- Старые center128-v2 и unversioned legacy plans продолжают читать прежние banks. Их нельзя заменять random128-v3 данными.
- `baseline_preprocessing.shared_sr_pair` принимает uint8 128×128 CHW/BCHW tensor, формирует HR/LR и нужный диапазон. Проверяются совпадение физических изображений между диапазонами и block means.
- `esrgan_pair` и `i2sb_transform` в том же модуле — **справочные native transforms**, не main protocol. Источники: [BasicSR dataset](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/paired_image_dataset.py), [BasicSR augmentation](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/transforms.py), [I²SB loader](https://github.com/NVlabs/I2SB/blob/master/dataset/imagenet.py).
- ESRGAN/I²SB/MSE+perceptual интегрированы в engine. CLI v4 по умолчанию запускает пять выбранных методов; исторический PatchGAN и MGM-абляции доступны явно. NFE grid I²SB 1/4/16/64 оценивается отдельными Slurm tasks. Инструкция — [CLUSTER.md](../CLUSTER.md).
- Реальные новые данные random128-v3, VGG19 weights, GPU smoke test и полный новый набор запусков в рамках этого изменения не готовились. Flickr2K и внешние test datasets не скачивались; dataset expansion требует отдельного решения.

## План запуска v4

Warmup: MSE и RRDB L1 по 2 часа/seed. Main: MSE/MSE+perceptual/ESRGAN/MGM по 2 часа, I²SB с нуля 4 часа для сопоставимого полного method budget. Физически 48 GPU-часов на стандартные 3 seed благодаря переиспользованию MSE warmup; evaluation считается отдельно. Веса loss и noise schedule заданы как начальные настройки для пилота; параметры CLI и ограничения сравнения — в [CLUSTER.md](../CLUSTER.md). Никаких GPU-запусков или скачивания весов в рамках интеграции не выполнено.
