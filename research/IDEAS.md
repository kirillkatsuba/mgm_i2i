# Идеи и решения по MGM для SR

Обновлено: 16 октября 2026. Пополняем по просьбе пользователя; предложения ниже не становятся решениями автоматически.

## Принятые решения

- Все выбранные методы используют **random128-v3**: онлайн случайные HR128 crops из полных train-фотографий; validation/test — фиксированный center128. Без resize, отражений и поворотов; LR32 через average pooling ×4. Прежние all-center и разные native baseline preprocessing отменены для новых планов.
- Сохраняем DIV2K и split 800/50/50 до отдельного решения об увеличении данных. Для основной серии одинаковы validation/test пары, размеры около 16–17 млн параметров и отсутствие внешнего generative pretraining; VGG/AlexNet раскрываем отдельно. Актуальная спецификация и предложение DF2K: [PREPROCESSING_PROTOCOL.md](PREPROCESSING_PROTOCOL.md).
- В новой кластерной архитектуре **v2** убираем `tanh` с residual: `raw = bicubic(LR) + UNet([bicubic(LR), Z])`.
- Поле v2 получает многомасштабный sinusoidal TimeEmbedding → MLP → добавление в residual-блоках. Четырёх карт `t, t², sin(2πt), cos(2πt)` больше нет. Время остаётся параметром MGM-смеси fake HR / real HR; генератор времени не получает.
- Проекцию и MSE warmup пока сохраняем. Решение об их изменении не принято.
- Старые кластерные checkpoints без `architecture_version` читаются как v1. Новые планы записывают v2 и требуют отдельного warmup; результаты версий нельзя смешивать как один эксперимент. Исторический Mac/CIFAR/toy-прототип в `src/mgm_i2i/models.py` в этом изменении не обновлялся.
- Следующим diffusion/bridge baseline для сравнения с MGM выбираем **I²SB-small**: time-conditioned U-Net 120/240/480 без pretrained ADM; точная спецификация и состав сравнения зафиксированы в разделе 7. SR3 в ближайший этап сравнения не включаем. Бюджет и noise schedule I²SB ещё предстоит определить по пилоту.
- Для следующей серии ширины U-Net **120/240/480** (`width=120`) приняты для MSE, MSE+perceptual, MGM G/field и I²SB-small. ESRGAN сохраняет стандартную конфигурацию RRDB23. Ширина 48 остаётся доступна только как явная настройка дополнительной серии.
- Основной набор сравнения: **MSE, MSE + VGG perceptual loss, ESRGAN, MGM v2, I²SB-small**. Используем одинаковый DIV2K random-train/center-eval протокол с pooling ×4; диапазон и интерфейс сети зависят от метода. Старый pure-GAN остаётся историческим objective baseline; основной GAN для новой серии — ESRGAN.

## 1. Варианты получения LR: отдельный эксперимент в будущем

В официальном [I²SB](https://github.com/NVlabs/I2SB) есть две задачи SR ×4:

- `sr4x-pool`: усреднение неперекрывающихся блоков 4×4. Оператор близок к нашему; известен точно и допускает простую проекцию.
- `sr4x-bicubic`: деградация bicubic-фильтром. Это другой оператор; формулу проекции для block averaging переносить на него нельзя.

Источник реализации операторов: [corruption/superresolution.py](https://github.com/NVlabs/I2SB/blob/master/corruption/superresolution.py).

Отдельно в [SR3, §3](https://arxiv.org/html/2104.07636v2) LR получают **bicubic downsampling с antialiasing**. Сглаживание перед уменьшением снижает наложение высоких частот (aliasing). Это характеристика деградации HR→LR, а не способ восстановления деталей.

`I²SB bicubic` и `bicubic + antialiasing` конкретной библиотеки не следует считать одним оператором без проверки ядра, границ и координат. Для будущего сравнения нужно зафиксировать реализацию, antialiasing и одинаковые пары для всех методов. Сейчас деградацию не меняем.

## 2. Upsampling и выходы SR-моделей

Единого требования увеличивать LR именно через bicubic нет:

- [SR3, §2.4](https://arxiv.org/html/2104.07636v2): bicubic до HR-размера, затем конкатенация с noisy HR.
- [ResShift, §2](https://arxiv.org/html/2307.12348v2): для выравнивания размеров допускается nearest-neighbor upsampling LR.
- [SwinIR, §3.1](https://arxiv.org/html/2108.10257v1): признаки извлекаются в LR; увеличение через sub-pixel convolution. В [официальном коде](https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py) также есть nearest+conv для real-world SR; финальный RGB получается свёрткой без `tanh` на поправке.
- [FlowSR, §4](https://arxiv.org/html/2605.12377v2): LR приводится к HR-размеру, используется flow/consistency; реализация опирается на Stable Diffusion и VAE.
- [OSEDiff, §3](https://arxiv.org/html/2406.08177v2): encoder → одношаговое преобразование latent → VAE decoder. Это не наша формула ограниченной RGB-поправки.

У нас сохраняется простой bicubic skip, но убирается ограничение residual через `tanh`. Его улучшение качества пока не установлено; обучение v2 ещё не проводилось.

## 3. Проекция: вопрос открыт

Проекция гарантирует, что уменьшение результата вернёт исходный LR. Например, если среднее HR-блока 0.4, а LR-пиксель 0.5, она добавляет 0.1 ко всем пикселям блока, сохраняя их взаимные различия.

Без проекции обучать conditional MGM можно. Возможные варианты: учить соответствие через conditional objective, добавить `||A(G(C,Z)) − C||²` или оставить hard projection. Первые два не гарантируют точное равенство. При шумном/неизвестном операторе hard constraint может быть неуместен.

Предложение для будущей абляции: projection / no projection / soft consistency при одинаковом остальном протоколе. Автоматически не запускаем и основную настройку пока не меняем.

## 4. TimeEmbedding v2

Для ширины w используются w sinusoidal features с логарифмически распределёнными частотами, затем MLP w→4w→4w с SiLU. Полученный embedding проектируется и добавляется в каждый residual-блок поля.

Базовое t∈[0, 0.5] перед кодированием умножается на 1000; это масштаб признаков, не изменение интерполянта и не раскрытие hidden flip. Выбор масштаба 1000 — инженерный параметр, оптимальность не проверена. В основной серии width=120 даёт 120 sinusoidal features и embedding размера 480.

## 5. Как устроен наш warmup

**Это supervised SR-регрессия, не обучение диффузии.** Вход: LR и нулевой latent; target: HR. `G(C, 0)` означает **Z=0**, а не t=0. Лосс: `MSE(G(C, Z=0), Y)`. Времени и предсказания шума нет.

После warmup веса генератора используются как общий старт MSE/GAN/MGM; поле инициализируется отдельно. У авторов MGM другой warmup: EDM-denoising зашумлённого изображения при разных уровнях шума.

Открытая проблема: переход от Z=0 на warmup к случайному Z в MGM. Альтернативы (например, постепенное введение latent или отдельный conditional denoising warmup) требуют отдельного решения и эксперимента. Warmup пока не меняем.

## 6. I²SB — выбранный baseline для следующего сравнения

**Решение от 16 октября 2026:** сосредоточиться на I²SB. Основа для воспроизведения — [официальный код NVlabs/I2SB](https://github.com/NVlabs/I2SB); подробный разбор обучения, inference и исходных сравнений — [SR3_I2SB_REPRODUCTION_PLAN.md](SR3_I2SB_REPRODUCTION_PLAN.md).

Почему выбран I²SB:

- Есть официальный PyTorch-код обучения и sampling, а также checkpoints.
- Постановка `sr4x-pool` близка нашей: block averaging ×4; официальный degraded endpoint получается через nearest upsampling LR до HR-размера.
- Модель восстанавливает HR, начиная с degraded image, и поддерживает разные inference budgets, включая один вызов сети. Это позволяет сравнить качество и время с однопроходным MGM.

**Протокол и детали дальнейшей подготовки:**

- Использовать общий DIV2K random128-v3 loader и pooling ×4. I²SB работает 32→128, со случайными train crops и без исходного Resize256/hflip. Validation/test пары и evaluator одинаковы для всех методов.
- Для основного сравнения выбран компактный denoiser сопоставимой ёмкости без внешнего generative pretraining (раздел 7). Официальный большой I²SB с pretrained ADM не входит в основной набор: именно такую initialization используют авторы, и её нельзя приравнивать к обучению с нуля на DIV2K.
- Построить quality–latency curve по NFE, включая NFE=1; измерять фактическое время при одинаковых GPU, precision и размере выхода. Один NFE разных архитектур не означает одинаковую стоимость.
- Сохранять native outputs I²SB и отдельно вариант с финальной LR-проекцией. Добавление проекции внутрь reverse chain считать отдельным изменением sampler.
- Сопоставлять PSNR/SSIM, LPIPS, LR consistency и визуальное качество; учитывать весь training budget, включая warmup MGM.

До реализации нужно зафиксировать upstream commit, noise schedule, бюджет обучения и сетку NFE. Архитектура и отсутствие pretrained initialization зафиксированы в разделе 7. Затем проверить data adapter, bridge target/sampler и провести короткий smoke test. Позднее выполнена локальная интеграция I²SB (раздел 13), GPU-обучение пока не запускалось; выбор baseline не означает, что его качество или преимущество MGM уже установлены.

## 7. Итоговый состав моделей и данные для следующей серии

**Последнее решение:** общие random train-crops128, center128 validation/test и pooling ×4 для всех методов. Диапазоны и интерфейсы сетей описаны в [PREPROCESSING_PROTOCOL.md](PREPROCESSING_PROTOCOL.md).

Зафиксировано 16 октября 2026 по запросу финализировать модели, размеры и данные. Это спецификация новой серии; обучение и замена существующих конфигураций этим документом не запускаются. Предыдущие результаты v1 сохраняются отдельно.

### Модели и число параметров

| Подход | Архитектура на inference | Параметры на inference | Дополнительная обучаемая сеть | Всего обучаемых параметров |
|---|---|---:|---|---:|
| MSE | Наш Generator v2, U-Net 120/240/480, Z=0 | 16 023 003 | Нет | 16 023 003 |
| MSE + perceptual | Тот же Generator v2, Z=0 | 16 023 003 | Нет; VGG19 заморожен | 16 023 003 |
| ESRGAN | RRDBNet ×4: 64 features, 23 RRDB, growth 32 | 16 697 987 | VGGStyleDiscriminator: 14 499 401 | 31 197 388 |
| MGM v2 | Наш Generator v2, U-Net 120/240/480, случайный Z | 16 023 003 | Conditional field v2: 17 293 203 | 33 316 206 |
| I²SB-small | Time-conditioned U-Net 120/240/480, 3→3 канала | 17 289 963 | Нет | 17 289 963 |

Числа воспроизводятся скриптом [count_selected_parameters.py](count_selected_parameters.py):

```bash
.venv/bin/python research/count_selected_parameters.py
```

Скрипт считает параметры actual local G/field/RRDB/D/BridgeDenoiser на meta device, без скачивания весов и обучения. Counts сверены с готовыми baseline implementations и совпадают с этой таблицей. Frozen VGG count не включает classifier.

Параметры означают веса и biases, включая trainable normalization parameters. Buffers, EMA-копии, optimizer states и evaluation networks в эти числа не входят. VGG19 feature extractor до conv5_4 имеет **20 024 384 замороженных параметра**: используется только в perceptual losses MSE+perceptual и ESRGAN и удаляется из inference. Classifier head VGG19 не используется. LPIPS/AlexNet — отдельная общая evaluation-сеть, не часть SR-моделей.

**Архитектурные детали и статус:**

- **MSE / MSE+perceptual / MGM:** существующая v2 без `tanh`, bicubic skip и hard LR projection. В MSE+perceptual используем pixel MSE и L1 в VGG19 features до ReLU conv5_4; вес perceptual term подбирается на validation. Для MGM основной вариант — согласованный stochastic interpolant; antithetic — дополнительная абляция той же архитектуры. Получается пять подходов и шесть training configurations при включённой antithetic-абляции.
- **ESRGAN:** стандартный [RRDBNet](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py) и [VGGStyleDiscriminator](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/discriminator_arch.py) с `input_size=128`, `num_feat=64`. D получает общий HR crop 128×128; оценка G на общих center128. Для SR-сети используем RGB [0,1]; train crop случайный из исходного фото, повороты и отражения отключены. Обучение: reconstruction pretraining на нашем train split, затем pixel L1 + VGG perceptual + relativistic adversarial loss по [рецепту BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/options/train/ESRGAN/train_ESRGAN_x4.yml). Для основной серии используем общий pooling ×4 вместо исходного bicubic, без скачивания готовых bicubic LR. Generator детерминированный. Сохраняем native output и вариант с той же финальной pooling-проекцией, что у MGM.
- **I²SB-small:** существующий core `UNet(cin=3, width=120, time_dim=480)` плюс `TimeEmbedding(120)`; нормированное bridge time t∈[0,1], sinusoidal features и MLP, embedding вводится в residual-блоки. Данные: общий random128-v3 (random train / center eval), без resize/поворотов/отражений, [-1,1]; задача 32→128. Input — bridge state X_t, output — 3-канальная оценка `(X_t − Y)/s_t`. Отдельного bicubic skip, `tanh` и конкатенации LR внутри denoiser нет; degraded endpoint — `nearest(LR)`. Bridge kernels и sampler сверяем с официальным I²SB. Интегрированы bridge objective и sampler; это **компактная адаптация метода**, не большой ADM-backbone авторов. Сохраняем native output и отдельно финальную LR projection.

Число inference parameters не равно стоимости: RRDBNet большую часть вычислений выполняет в LR, а наш U-Net — в HR. Скорость требуется измерять. Генераторы теперь близки по числу параметров: наш 16.02 млн против ESRGAN 16.70 млн; I²SB-small — 17.29 млн. MGM вместе с полем содержит 33.32 млн обучаемых параметров против 31.20 млн у ESRGAN вместе с D. Архитектуры и стоимость вычислений различаются: это сравнение систем, а не изоляция одного objective на одинаковой архитектуре.

### Общие данные всех методов

- **Датасет:** DIV2K HR, без FFHQ, ImageNet и дополнительных HR-коллекций для обучения SR-сетей.
- **Train:** 800 официальных train images, ID 0001–0800.
- **Validation / held-out test:** 100 официальных validation images, ID 0801–0900, разделены на 50/50 по исходным фотографиям; используем существующий manifest и split seed=123. Это наш held-out test, не закрытый официальный test DIV2K.
- **Train:** 800 полных исходных RGB-фотографий в отдельных mmap NPY; на каждый sample index случайно выбираем фото и координаты crop 128×128. Фиксированного банка train-патчей больше нет; повороты/отражения не применяем.
- **Train task:** LR 32×32 → HR 128×128. LR получаем точным `avg_pool2d(HR,4)` после выбранного HR crop, без промежуточного PNG-округления.
- **Validation/test:** по одному центральному HR crop 128×128 на фото, итого по 50 crops; LR 32×32. Метрики усредняются по 50 исходным фотографиям каждого split.
- **Диапазон:** RGB float [-1,1] у MSE/MGM/I²SB, [0,1] у ESRGAN; это обратимое представление одних физических HR/LR пар. VGG получает своё ImageNet preprocessing внутри loss; общий evaluator сохраняет нынешний clipping/нормализацию и отдельно измеряет LR consistency до/после clipping.
- **Pretraining:** готовые SR/GAN/diffusion checkpoints извне не используем. Reconstruction warmup выполняется на нашем train split. Замороженные ImageNet VGG19 для loss и AlexNet для LPIPS явно раскрываем как внешние feature networks.

### Что ещё определяется перед запуском

Состав моделей, их размеры и данные зафиксированы. Learning rates, loss weights, I²SB noise schedule, полный GPU-бюджет и NFE grid выбираем по коротким пилотам с одинаково обоснованным validation tuning budget. Время reconstruction pretraining/warmup включается в полную стоимость метода. Текущий MGM warmup сохраняется; вопрос перехода Z=0→случайный Z из раздела 5 остаётся открытым. Новые результаты нельзя смешивать с v1 или считать улучшением до оценки.

## 8. Переход на единый центральный crop 128×128

**История:** сначала был принят center128 для всех split; последнее решение раздела 12 заменяет только train на random128. Val/test остаются центральными.

Решение пользователя от 16 октября 2026 заменяет прежнюю двухэтапную random-bank схему. Центр исходного изображения размера W×H: `left=(W−128)//2`, `top=(H−128)//2`; берём ровно 128×128 исходных пикселей. Не уменьшаем фотографию и не вырезаем центр старого случайного patch.

Подготовка: `scripts/cluster_prepare.py --crop-mode center` (явное воспроизведение прежнего режима), новая папка `data/div2k_center128`. Существующие ZIP можно использовать через `--archive-root data/div2k/downloads`. Планы прежнего центрального режима содержат `data_protocol=center128-v2`, `crop=eval_crop=128`; старые планы без `data_protocol` сохраняют прежний loader для resume/evaluation. Подмена старого manifest новым блокируется; нужны новая папка run и новый warmup.

Данные не скачивались и настоящие DIV2K-банки в рамках этого изменения не пересоздавались. Проверка подготовки выполняется на синтетических локальных ZIP. ESRGAN/I²SB интегрированы в плане v4 (раздел 13), GPU-пилот остаётся следующим этапом.

## 9. Увеличение ширины основной серии до 120/240/480

Решение пользователя от 16 октября 2026: увеличить U-Net до `width=120` для сближения размера с ESRGAN. Обновлены defaults кластерных Generator/Field, `cluster_submit.py` и parameter inventory. В сохранённых планах/checkpoints ширина задана явно; они читаются с исходным размером. Warmup ширины 48 нельзя загружать в сеть ширины 120; нужна новая папка run и новый warmup. Тип архитектуры остаётся v2.

Команда `.venv/bin/python research/count_selected_parameters.py` считает основную серию; `--width 48` воспроизводит прежнюю компактную спецификацию. I²SB-small сохраняет название компактной адаптации относительно большого ADM авторов; на момент выбора ширины интеграции ещё отсутствовали. Сейчас они реализованы в плане v4 (раздел 13); сама смена ширины в старых планах не допускается. GPU-пилот, расход VRAM и качество увеличенной модели пока не проверены; обучение не запускалось.

## 10. Возврат preprocessing исходных baseline

**Промежуточное решение, отменено разделом 11:** ESRGAN и I²SB используют исходные crop/resize, augmentation и normalization; детали, источники, трактовка деградаций и статус реализации — в [PREPROCESSING_PROTOCOL.md](PREPROCESSING_PROTOCOL.md). Добавлены отдельные transforms и pixel-level проверки; на момент этого промежуточного решения baseline engine integrations отсутствовали; сейчас см. раздел 13. Нативные метрики разных задач не образуют общий рейтинг. Архитектуры, отсутствие внешнего generative pretraining и split не меняются.

## 11. Подтверждённый общий протокол честного сравнения

Пользователь подтвердил center128 для всех, average pooling ×4, одинаковые validation/test пары, близкий размер моделей и отсутствие готовых внешних generative weights. Это отменяет разные native geometry/degradation из раздела 10. Диапазоны сетей допускаются разные при общем evaluator. Детали — [PREPROCESSING_PROTOCOL.md](PREPROCESSING_PROTOCOL.md).

Текущее разбиение 800/50/50 сохраняется; вопрос расширения обсуждается отдельно. Предлагаемый DF2K (3450/50/50) и дополнительные BSD100/Urban100 — предложения, ещё не принятые и не подготовленные.

## 12. Онлайн случайные train-патчи 128×128

Принято пользователем: вместо одного центра train-фото каждый раз равновероятно выбираем исходную train-фотографию и случайный HR crop128 в ней. Validation/test сохраняют один center128 на фото; pooling ×4, split 800/50/50, архитектуры и диапазоны не меняются. Flickr2K не добавляем. Отражения и повороты не включаем.

Подготовка `--crop-mode random` теперь default, новый root `data/div2k_random128`, protocol `random128-v3`. Полные train RGB сохраняются в отдельных mmap NPY; это требует места для полных фото, но исключает PNG decoding в каждом sample. SHA каждого файла записан в manifest и проверяется перед submit. Seed + sample index определяют image/top/left, поэтому discarded prefetch и resume не меняют последовательность. Нужны новый run root и warmup; старые планы и loader modes сохранены. Детали — [PREPROCESSING_PROTOCOL.md](PREPROCESSING_PROTOCOL.md).

## 13. Полный Slurm-план и параллельные GPU-задачи

По запросу пользователя подготовлен plan v4: одна модель/seed на одной GPU, default max_parallel=4, фазы warmup → train → eval → collect с барьерами afterok и финальным afterany. Параллельность относится к независимым runs, без DDP. Реальные queue/GPU/VRAM здесь не проверялись.

Интегрированы RRDB23 и VGG-style D128, local frozen VGG19 pre-ReLU conv5_4, relativistic-average logistic objective и отдельный RRDB L1 warmup; I²SB symmetric schedule, bridge target и coarse posterior sampler с NFE grid. Подготовка теперь скачивает VGG19, train/eval не скачивают ничего. Local adaptations не импортируют upstream training stack и не претендуют на воспроизведение исходных published metrics.

Бюджет по default: warmup MSE/RRDB по 2 часа/seed; main четыре метода по 2 часа после warmup; I²SB с нуля 4 часа, чтобы полный method budget был сопоставим. Всего физически 48 GPU-часов training плюс evaluation, с общим MSE warmup для трёх подходов. GPU evaluation вынесена в отдельный array, 24 tasks для 3 seed и NFE1/4/16/64. FP32 batch1 latency, native и projected quality, warmup time отдельным полем. Все defaults — начальные конфигурации для validation-пилота, не утверждение об оптимальных гиперпараметрах или сходимости за 4 часа.

Готовые команды установки, подготовки, pilot/main, контроля, resume и test: [CLUSTER.md](../CLUSTER.md). Изменение числа GPU не меняет размер модели или training budget; перед длинной серией выполнить короткий пилот всех методов. Никакие sbatch, скачивание или обучение в этой сессии не запускались.
