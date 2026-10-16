# SR3 и I²SB: исходные пайплайны и воспроизведение на DIV2K

Проверено: 16 октября 2026 года. Статус: исследование статей и чтение публичного кода. Позднее локальные ESRGAN/I²SB baseline интегрированы в plan v4, см. [CLUSTER.md](../CLUSTER.md); во время первоначального разбора зависимости, checkpoints и GPU-запуски в рамках этого разбора не устанавливались/не запускались. Доступность ссылок на веса не равна проверке успешной загрузки и качества этих весов.

**Последнее обновление preprocessing:** для основной серии все методы используют общий random train128 / center eval128, pooling ×4 и задачу 32→128, без resize/поворотов/отражений (random128-v3). Это адаптация I²SB к общим данным; native ImageNet-256 preprocessing ниже описывает исходную работу. Актуальный протокол и предложение расширения до DF2K — [PREPROCESSING_PROTOCOL.md](PREPROCESSING_PROTOCOL.md).

## 1. Практический вывод

Обучить SR3-подобную conditional diffusion и I²SB на тех же DIV2K-парах технически возможно. Однако два утверждения нужно разделять:

1. **Воспроизведение опубликованной системы:** её архитектура, pretraining, данные, preprocessing и sampler. Получение исходных метрик требует исходного протокола.
2. **Сравнение методов на нашей задаче:** DIV2K, average pooling ×4, единый split и evaluator. Это адаптация с явно описанными отличиями, даже при использовании официальных diffusion kernels.

Рекомендованный основной эксперимент — компактные модели на одинаковых данных, без внешнего generative pretraining и с учётом полного бюджета обучения. Официальный pretrained I²SB можно оценивать отдельно как внешний ориентир.

## 2. Источники и доступный код

### SR3

- [Статья, arXiv 2104.07636v2](https://arxiv.org/abs/2104.07636v2), §2, §4, Appendix A; локальная [копия](../materials/papers/sr3.pdf).
- [Страница авторов](https://iterative-refinement.github.io/).
- [Janspiry/Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) — **неофициальная** PyTorch-реализация с training, validation, inference и ссылками на checkpoints.
- [novwaul/SR3](https://github.com/novwaul/SR3) — ещё одна сторонняя реализация; здесь рассмотрена как дополнительный кандидат, без полного аудита её training loop.

На проверенных официальных страницах и в поиске не обнаружен публичный исходный train/inference-репозиторий SR3 от авторов. Это результат проверки доступных источников, а не доказательство отсутствия любого другого релиза.

В README Janspiry прямо описаны отличия от статьи: DDPM-style residual blocks/concatenation, размещение attention, noise embedding и variance reverse transition. Поэтому обозначать этот репозиторий как «официальный SR3» или точное воспроизведение оригинала нельзя.

### I²SB

- [Статья, arXiv 2302.05872v3](https://arxiv.org/abs/2302.05872v3), §3, §5, Appendix C; локальная [копия](../materials/papers/i2sb.pdf).
- [Официальный PyTorch-репозиторий NVlabs/I2SB](https://github.com/NVlabs/I2SB).
- Предусмотрены checkpoints `sr4x-pool` и `sr4x-bicubic`; training — `train.py`, sampling — `sample.py`.

Чтение кода проводилось по текущим `master`/`main`; SHA релизов пока не зафиксированы. Перед переносом кода нужно сохранить конкретные commit SHA и изменения адаптера.

## 3. SR3: исходное обучение

Обозначения: $C$ — LR, $Y$ — HR, $C_\uparrow$ — bicubic-представление LR в HR-размере.

### Данные

- Faces: обучение FFHQ, оценка CelebA-HQ; в частности 16→128 и 64→512.
- Natural images: обучение ImageNet 1K, оценка на dev/validation; основной SR ×4 — 64→256.
- LR строится bicubic downsampling с antialiasing.
- Для ImageNet авторы отбрасывают слишком маленькие изображения, используют largest central crop и area resampling до HR-размера.

Это другой preprocessing, чем наш банк DIV2K patches и average pooling.

### Обучающий sample

1. Берём парные $(C,Y)$.
2. Выбираем уровень сохранённого сигнала $\gamma$ и $\varepsilon\sim N(0,I)$.
3. Получаем noisy HR:

$$Y_\gamma=\sqrt\gamma Y+\sqrt{1-\gamma}\varepsilon.$$

4. Denoiser получает $(C_\uparrow,Y_\gamma,\gamma)$ и предсказывает шум.

$$L_{\rm SR3}=\mathbb E\|f_\theta(C_\uparrow,Y_\gamma,\gamma)-\varepsilon\|_p^p,
\qquad p\in\{1,2\}.$$

Статья рассматривает обе нормы; в абляции cascaded ImageNet generation L1 слегка улучшает FID. Это loss к шуму, а не обычная MSE-реконструкция HR.

За один training sample не нужно проходить всю diffusion chain: noisy HR сэмплируется напрямую по формуле. Многошаговая цепочка нужна при inference.

### Архитектура и бюджет из статьи

U-Net на основе DDPM с BigGAN-style residual blocks, масштабированием skip connections на $1/\sqrt2$, расширенной глубиной и noise-level conditioning. LR конкатенируется с noisy HR.

Для 64→256 Appendix A указывает: base channels 128, multipliers {1,2,4,4,8,8}, 3 ResNet blocks, **625 млн параметров**. Для 16→128 — 550 млн. Наш G в следующей серии 120/240/480 имеет около 16.02 млн параметров (в прежней серии ширины 48 — 2.57 млн), поэтому прямое сравнение размеров/скорости не изолирует training objective.

Обучение SR3 и regression baselines: **1 млн optimizer steps**, batch 256, Adam, linear learning-rate warmup на 10 тыс. шагов, далее LR $10^{-4}$ для SR3 и $10^{-5}$ для regression. SR3 оценивают по последнему checkpoint; regression выбирают по held-out PSNR. В §2.4 указано 2000 training diffusion intervals с выбором непрерывного noise level между соседними уровнями.

Warmup learning rate здесь не означает предварительное supervised MSE-обучение генератора, которое используется у нас.

## 4. SR3: исходный inference и сравнения

### Inference

Начало — чистый Gaussian noise HR-размера. На каждом шаге denoiser получает текущее noisy image, LR condition и noise level. Reverse transition уменьшает шум и обычно добавляет случайность; последний шаг — без нового noise.

Авторы ограничивают inference **100 шагами** и подбирают inference noise schedule по held-out FID, пользуясь conditioning на continuous noise level. 2000 training intervals не означают обязательные 2000 inference calls.

Для корректного сокращения цепочки нужно корректное noise schedule и переходы. Просто заменить 2000 на 10 при неизменных endpoints beta — плохая замена: может измениться terminal SNR, и старт из Gaussian noise перестанет соответствовать обученному процессу.

### Сравнения

- Faces: PULSE, FSRGAN, regression той же архитектуры, bicubic в визуальных примерах.
- Natural SR: regression по FID, IS, PSNR, SSIM; дополнительная оценка через классификацию с DRCN, FSRCNN, PsyCo, ENet-E, RCAN.
- Human evaluation 2AFC: реалистичность и согласованность с LR.
- Cascaded generation — отдельный эксперимент; его FID не является чистой SR-метрикой на фиксированных настоящих LR.

Показательный результат Table 2, natural-image SR: regression PSNR 27.9 / FID 15.2; SR3 PSNR 26.4 / FID 5.2. Снижение PSNR сочетается с улучшением distributional quality. Эти числа получены на ImageNet и не сравнимы напрямую с нашими DIV2K-метриками.

## 5. SR3: что есть в Janspiry и что потребуется изменить

Проверены:

- [README](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/README.md): train/val/inference entrypoints, LMDB/PNG datasets, отличие реализации от статьи.
- [config/sr_sr3_16_128.json](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/config/sr_sr3_16_128.json): 6→3 каналов, width 64, два residual blocks; по умолчанию train и val schedules имеют 2000 шагов; EMA scheduler в конфиге помечен как неиспользуемый.
- [model/sr3_modules/diffusion.py](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/diffusion.py): continuous noise-level training, L1/L2, reverse sampler, LR-condition через `SR`.
- [data/LRHR_dataset.py](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/data/LRHR_dataset.py): формат пар/conditioning images.

Важные отличия кода от статьи: training выбирает continuous **sqrt cumulative alpha** между соседними значениями; статья формулирует sampling через gamma. Эту деталь тоже нужно фиксировать при выборе воспроизведения.

Upstream entrypoints:

```bash
python sr.py -p train -c config/sr_sr3_16_128.json
python sr.py -p val -c config/sr_sr3_16_128.json
python infer.py -c config/sr_sr3_16_128.json
```

Это существующие команды стороннего репозитория, **не готовые команды для нашего DIV2K**. Для нашей задачи нужно заменить loader/preprocessing, разрешения, sampling schedules и evaluator. Сторонний `prepare_data.py` нельзя запускать вслепую: он создаст свой LR вместо нашего average pooling.

Рекомендация: использовать код как проверочный reference, а основной baseline реализовать по статье в общем engine и называть **SR3-style conditional diffusion**. Точный режим gamma/sqrt-gamma, loss и sampler явно записать в config.

## 6. I²SB: исходное обучение

### Данные и initialization

ImageNet 256×256. Для SR — 64→256, с pooling или bicubic filters, заимствованными из DDRM. Clean и degraded endpoints должны иметь одинаковую размерность.

В исходных экспериментах denoiser инициализируется **pretrained unconditional ADM на ImageNet 256×256**. Это внешнее pretraining, не обучение I²SB с нуля на маленьком наборе. ADM служит инициализацией: отсюда не следует использование teacher-generated targets в последующем loss.

### Обучающий sample

Пусть clean endpoint $X_0=Y$, degraded endpoint $X_1=C_\uparrow$. При $f=0$ bridge имеет аналитическое условное распределение:

$$X_t\mid X_0,X_1\sim N(\mu_t,\sigma_{\rm bridge,t}^2I),$$

$$\mu_t=\frac{\bar s_t^2}{s_t^2+\bar s_t^2}X_0+
\frac{s_t^2}{s_t^2+\bar s_t^2}X_1,$$

$$\sigma_{\rm bridge,t}^2=\frac{s_t^2\bar s_t^2}{s_t^2+\bar s_t^2},
\quad s_t^2=\int_0^t\beta_\tau d\tau,
\quad\bar s_t^2=\int_t^1\beta_\tau d\tau.$$

Target сети:

$$L_{\rm I2SB}=\mathbb E\left\|\epsilon_\theta(X_t,t)-\frac{X_t-X_0}{s_t}\right\|^2.$$

Это **не просто добавленный Gaussian noise bridge**: numerator содержит displacement от clean endpoint. Использовать вместо target случайный $\varepsilon$ из bridge sampling было бы другим алгоритмом.

Восстановление clean prediction:

$$\widehat X_0=X_t-s_t\epsilon_\theta(X_t,t).$$

Sampling intermediate state не требует полной SDE-симуляции на каждом training step. В стандартном restoration-варианте сеть видит $(X_t,t)$, отдельная копия LR не конкатенируется. Флаг `--cond-x1` добавляет endpoint conditioning; в статье он полезен для более общих translation tasks. Его включение нужно отмечать как отдельный вариант.

## 7. I²SB: официальный код, настройки и inference

### Подтверждённые детали кода

- [train.py](https://github.com/NVlabs/I2SB/blob/master/train.py): ImageNet loader; defaults batch 256, microbatch 2, max iterations 1 млн, LR $5\cdot10^{-5}$, EMA 0.99, interval 1000. Training FP16 по умолчанию отключён.
- [runner.py](https://github.com/NVlabs/I2SB/blob/master/i2sb/runner.py): AdamW; MSE к `(xt-x0)/std_fwd`; StepLR; gradient accumulation; EMA sampling.
- [scripts/train.sh](https://github.com/NVlabs/I2SB/blob/master/scripts/train.sh): для `sr4x-pool` и `sr4x-bicubic` задан `beta-max=0.3`, без `--ot-ode`.
- [network.py](https://github.com/NVlabs/I2SB/blob/master/i2sb/network.py) и [ckpt_util.py](https://github.com/NVlabs/I2SB/blob/master/i2sb/ckpt_util.py): создание большой ADM U-Net из сохранённых kwargs; загрузка pretrained весов; удаление sigma-head. Одного CLI-параметра `--image-size` недостаточно, чтобы превратить этот путь в маленькую модель: Runner создаёт `Image256Net`.
- [corruption/superresolution.py](https://github.com/NVlabs/I2SB/blob/master/corruption/superresolution.py): block averaging и **nearest upsampling** для degraded endpoint.
- [diffusion.py](https://github.com/NVlabs/I2SB/blob/master/i2sb/diffusion.py): bridge marginals, coarse-step posterior и DDPM sampling.

Это defaults текущего кода. Их нельзя автоматически выдавать за подтверждённые логи длительности обучения каждого опубликованного checkpoint. README сообщает, что опубликованные checkpoints обучены на двух узлах по 8 V100 32GB.

### Inference

Старт — **degraded endpoint**, не чистый Gaussian noise. На выбранной временной сетке предсказываем clean endpoint и сэмплируем обратный posterior до следующего времени. Число вызовов сети задаётся `--nfe`.

В статье базовая дискретизация — 1000 времён. В текущем `runner.py` при `interval=1000` default sampler выполняет **999** network calls (`interval-1`); нужно считать реальные NFE, а не отождествлять их с длиной training schedule.

Однопроходный I²SB тоже нужно включить в сравнение. В проверенном SR-коде при NFE=1 стартовый endpoint детерминирован, а в финальном переходе шум не добавляется; в eval-режиме без дополнительных noise-модификаций результат детерминирован. Поэтому требование ненулевого diversity для этого режима было бы некорректным.

Upstream entrypoints, требующие ImageNet-compatible LMDB:

```bash
python train.py --name div2k_pool_reference --n-gpu-per-node 1 \
  --corrupt sr4x-pool --dataset-dir PATH_TO_COMPATIBLE_LMDB \
  --batch-size 16 --microbatch 1 --beta-max 0.3

python sample.py --ckpt div2k_pool_reference --n-gpu-per-node 1 \
  --dataset-dir PATH_TO_COMPATIBLE_LMDB --batch-size 1 --nfe 16
```

Показанные batch sizes — пример настройки entrypoint, не измеренная конфигурация для нашей GPU. Простая передача пути к нашим `.npy` в `--dataset-dir` не сработает.

Окружение upstream рассчитано на Python 3.8 / PyTorch 1.12 / torchvision 0.13 / torchmetrics 0.9.3. Совместимость с нашим PyTorch 2.6 нужно проверить отдельным smoke test. Это ещё не выполнено. Built-in evaluator зависит от ImageNet labels, ResNet50 и ImageNet FID statistics; для DIV2K его заменяем общим evaluator.

## 8. С чем сравнивали I²SB

Основные restoration baselines:

- conditional score models: ADM, Palette;
- inverse-problem samplers с известным оператором: DDRM, DDNM, ΠGDM;
- conditional Schrödinger bridge: CDSB.

В SR Tables 2 приведены ADM, DDRM, DDNM, ΠGDM, CDSB; Palette участвует в других restoration-сравнениях, в том числе анализе NFE. Называть эти таблицы прямым SR3-vs-I²SB сравнением нельзя.

Основные показатели — FID и ResNet50 classifier accuracy. Для SR используется полный ImageNet validation set и reference FID statistics ImageNet train. Table 2: I²SB Pool FID 2.7 / CA 71.0%; Bicubic FID 2.8 / CA 70.7%. Эти FID нельзя сопоставлять напрямую с SR3 FID 5.2: различаются системы и reference/evaluation protocols.

В дополнительных general I2I задачах сравнивают с pix2pix; в inpainting — также DeepFillv2 и HiFill. Вывод «один NFE уже конкурентоспособен» относится к указанным задачам; его нельзя без проверки переносить на наш SR.

## 9. Как сравнить на наших данных

### Общие данные

Использовать текущие `train.npy`, `val.npy`, `test.npy`, `manifest.json` и crop sampler из [data.py](../src/mgm_i2i/cluster/data.py):

- те же исходные image IDs и split;
- один исходный центральный crop 128×128 на фото, без resize и spatial augmentation;
- HR float в [-1,1], LR строго `avg_pool2d(HR,4)` после crop;
- train/val/test HR 128, LR 32;
- normalization, clipping и evaluator из [metrics.py](../src/mgm_i2i/cluster/metrics.py).

Не сохранять LR через PNG с дополнительным округлением и не генерировать его заново сторонними prepare scripts. Bicubic-представление condition и nearest-представление bridge endpoint — обратимые представления одного LR, но разные архитектурные решения; их надо фиксировать.

Для каждого метода должны совпадать распределение train-пар и точные evaluation pairs. При равном времени обучения различное число обновлений приведёт к различному числу просмотренных samples; это требуется записывать, а не обещать одинаковую последовательность batches.

### Два уровня сравнения

| Уровень | Что делаем | Допустимый вывод |
|---|---|---|
| Основной: matched backbone / data / budget | Компактный time-conditioned U-Net сопоставимой ёмкости; SR3 и I²SB kernels; без внешнего pretrained denoiser | Эффективность методов в ограниченном режиме |
| Дополнительный: исходные системы | Большой официальный I²SB с ADM initialization; выбранная SR3 reimplementation | Практический внешний ориентир с другим prior/compute |

Matched backbone означает общие ширины, глубину, normalization и сопоставимую ёмкость. Входы и targets сохраняют семантику метода: SR3 — LR+noisy HR и noise level, I²SB — bridge state+time. Использовать готовый `Generator` wrapper с bicubic skip как epsilon-predictor без обоснования не следует; общий core U-Net отделяем от output heads.

Новые baseline training objectives не получают автоматически наш MSE warmup: он обучает другой интерфейс. Для сравнения по compute считаем **всё** обучение, включая MGM warmup. Равные GPU-часы не гарантируют сходимости каждого метода. Сначала короткий пилот с кривыми validation, затем фиксируем бюджет.

### Проекция

Текущий MGM содержит hard LR projection. Исходные SR3/I²SB не обязаны иметь такую же операцию на каждом шаге.

Для проверки сохраняем native SR3/I²SB outputs и отдельно их вариант **с одной финальной проекцией**. Внутри reverse chain проекцию не добавляем без отдельной абляции: это меняет sampler. Отдельный projected результат нужен, чтобы не приписывать общий data-consistency prior преимуществу MGM-loss.

### Inference curve

- MGM: NFE=1.
- I²SB: NFE ∈ {1,2,4,8,16,32,64,100}; более длинный reference при доступном бюджете.
- SR3: для коротких budgets корректные inference schedules; оценка на тех же NFE, включая 100 как ориентир из статьи. Совсем малые NFE нужно интерпретировать с осторожностью и не использовать для искусственного ослабления baseline.

Для каждого NFE: PSNR, SSIM, LPIPS, при возможности DISTS, consistency до/после clipping, одинаковое число outputs без выбора по test target. FID на 50 исходных фото не используем как основной критерий воспроизведения image distribution.

Отдельно измеряем batch-1 latency и throughput при одинаковых GPU/precision/output shape. Timing включает весь sampler, preprocessing condition и sampling noise по объявленному протоколу; исключаем загрузку checkpoint, I/O, метрики, траектории для визуализации. В upstream I²SB logging переносит состояния на CPU: это не должно тайно входить только в его benchmark.

Большая разница latency при большом pretrained ADM и нашем маленьком U-Net отражает и размер сети. Главный график — качество–время с раскрытыми параметрами/pretraining, а не «NFE означает одинаковую стоимость».

## 10. Конкретный следующий этап реализации

1. Зафиксировать upstream commit SHA и список отличий от статей.
2. Добавить общий paired-data adapter к нашим банкам; проверить совпадение LR/HR batches и evaluation IDs.
3. Реализовать time/noise-conditioned core denoiser и отдельные SR3/I²SB parameterizations.
4. Проверить на маленьких тензорах bridge mean/variance, target→clean inversion, конечность endpoints и фактическое число NFE; сравнить с reference kernels.
5. Провести короткий CPU/CUDA smoke test train→checkpoint→resume→sample, без запуска полной серии заранее.
6. Измерить скорость/память и подобрать небольшой одинаково обоснованный validation tuning budget для обоих методов.
7. Зафиксировать основной протокол, обучить по нескольким seed, построить quality–latency curves; held-out test использовать после фиксации выбора.

Подготовленный разбор делает сравнение технически определённым. Он не является подтверждением, что два часа обучения достаточно для SR3/I²SB, что сторонние checkpoints воспроизводят цифры статей или что MGM их обгонит.
