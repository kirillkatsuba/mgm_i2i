# Выходы SR-сетей и возможность DDPM++ для MGM

## Наши v1 и v2

Обозначим C — LR32, U(C) — bicubic HR128, A — average pooling×4, R — nearest broadcast LR→HR. Общая жёсткая проекция:

\[
\Pi_C(Y)=Y+R(C-A(Y)),\qquad A(\Pi_C(Y))=C.
\]

В обеих версиях U-Net получает конкатенацию `[U(C), Z]` (6 каналов) и возвращает три канала RGB-поправки:

\[
G_{v1}(C,Z)=\Pi_C\left(U(C)+\tanh(F_\theta([U(C),Z]))\right),
\]
\[
G_{v2}(C,Z)=\Pi_C\left(U(C)+F_\theta([U(C),Z])\right).
\]

`tanh` v1 ограничивает именно поправку до сложения и projection. Ни одна формула не гарантирует диапазон финального RGB [-1,1]. Clipping применяется evaluator'ом при метриках/сохранении; в основном training objective его нет. Shape выхода G в обоих случаях `[B,3,128,128]`. Выход поля имеет такой же spatial shape, но означает signed displacement, а не RGB-картинку; у поля обеих версий финального tanh нет. У v1 время — четыре входные карты, у v2 — sinusoidal embedding + MLP в residual-блоках. Critic-only noise меняет training observations, формулу inference G не меняет.

Источники реализации: `src/mgm_i2i/cluster/models.py`, `losses.py`, `cluster/metrics.py`.

## Примеры распространённых SR-систем

Это представители разных семейств, а не заявление, что перечисленные модели являются лидерами всех SR-бенчмарков на текущую дату.

| Семейство | Непосредственный выход сети | Как получается итоговый HR |
|---|---|---|
| ESRGAN / RRDBNet | 3-channel RGB после последней Conv2d, без финального tanh/sigmoid | Learned upsampling + conv; обычного hard LR projection нет |
| SwinIR / HAT, classical SR | RGB reconstruction head с upsampling, без финального tanh | Возврат исходных mean/range; архитектурные residual connections преимущественно в features |
| DDPM/SR3-подобный epsilon predictor | Предсказание шума при заданном noisy state и времени | Reverse diffusion steps превращают состояния в HR |
| I²SB | Нормированный residual `(X_t-X_0)/std_fwd(t)` | Clean estimate `X_t-std_fwd*net_out` используется bridge sampler'ом |
| EDM denoiser | Backbone F плюс preconditioning | `D(x,sigma)=c_skip(sigma)*x+c_out(sigma)*F(c_in(sigma)*x,c_noise(sigma))` |

Отсутствие tanh само по себе не делает модель качественной: существенны objective, conditioning, инициализация и оптимизация. Наличие RGB output или skip connection не гарантирует consistency с LR.

Первичные источники: [BasicSR RRDBNet](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py), [SwinIR](https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py), [HAT](https://github.com/XPixelGroup/HAT/blob/main/hat/archs/hat_arch.py), [SR3](https://arxiv.org/abs/2104.07636), [I²SB runner](https://github.com/NVlabs/I2SB/blob/master/i2sb/runner.py), [EDM networks](https://github.com/NVlabs/edm/blob/main/training/networks.py).

## DDPM++ и обычная DDPM

**DDPM — способ задания и обучения диффузионной модели; DDPM++ — название семейства U-Net backbone.** Архитектуру U-Net можно использовать с разными losses и samplers. В исходной MGM авторы уже используют EDM ddpm++ для G и displacement field (приложение B); это естественный следующий backbone для нашей адаптации.

Возможная будущая реализация MGM с DDPM++:

1. Conditional generator получает upsampled LR и latent Z, возвращает RGB/residual. Time embedding генератора фиксируется либо исключается согласно выбранной параметризации; inference остаётся одним вызовом.
2. Displacement field получает upsampled LR, mixed HR и MGM time; возвращает displacement.
3. Сохраняется MGM objective. DDPM loss не появляется автоматически от замены backbone. Можно отдельно реализовать conditional denoising warmup для обеих сетей, учитывая соответствие входов и смысл выходной головы.
4. Ширины, число блоков, attention resolutions и нормализацию подбираем под целевое число параметров/VRAM. Числа120/240/480 нельзя механически перенести в любую upstream GroupNorm-конфигурацию; одинаковые widths у разных backbone не означают одинаковые параметры или compute.

DDPM-style U-Net также подходит. Если же обучать именно conditional DDPM с epsilon loss и reverse sampler, получится дополнительный diffusion baseline, обычно с несколькими вызовами сети. Само использование DDPM++ не превращает MGM в diffusion sampler.

Если под «обученную DDPM» подразумевать готовые pretrained weights, нужна отдельная адаптация к LR conditioning, разрешению, входным каналам и output parameterization. Unconditional checkpoint не становится SR или one-step генератором автоматически. Использование внешнего pretraining необходимо раскрывать отдельно от текущего сравнения с нуля.

На этом этапе DDPM++ не интегрируем одновременно с двумя согласованными абляциями; сначала измеряем wide v1 и v2 critic-only. Затем backbone можно проверить как отдельную версию с сохранением общего data/evaluation protocol.

## Формулы выбранных baselines и смысл projection

Дополнение от18 октября2026. Формулы сверены с нашими `cluster/baselines.py`, `engine.py`, `metrics.py` и официальным кодом ESRGAN/I²SB.

### MSE и MSE+perceptual

У этих двух baselines текущей серии одна и та же формула v2 с нулевым latent:

\[
\widehat Y_{MSE}(C)=\Pi_C(U(C)+F_{\theta_M}([U(C),0])),\qquad
\widehat Y_{MP}(C)=\Pi_C(U(C)+F_{\theta_P}([U(C),0])).
\]

Веса различаются, поскольку training losses различны. VGG применяется при обучении MP и не является частью inference. Обе модели имеют встроенную проекцию.

### ESRGAN

Обозначим \(q=(C+1)/2\) — LR в нормировке [0,1], \(\ell\) — LeakyReLU, \(N_2\) — nearest upsampling×2. Схема RRDBNet:

\[
h_0=C_{first}(q),\quad h=h_0+C_{body}(RRDB_{23}(h_0)),
\]
\[
h_1=\ell(C_{up1}(N_2h)),\qquad h_2=\ell(C_{up2}(N_2h_1)),
\]
\[
\widehat Y_E=2C_{last}(\ell(C_{hr}(h_2)))-1.
\]

Последняя свёртка возвращает RGB, без финального tanh, без нашего bicubic pixel skip и без hard LR projection. Внутренние residual connections есть в признаках/RRDB. Точное \(A\widehat Y_E=C\) исходной архитектурой не обеспечивается. На evaluation мы отдельно вычисляем \(\Pi_C(\widehat Y_E)\); это наш дополнительный вариант результата, не часть original ESRGAN training.

### I²SB

Наш denoiser получает bridge state и время, возвращает нормированный residual:

\[
r_\phi(X_k,t_k)\approx(X_k-Y)/s_k,
\qquad \widehat Y_{0|k}=X_k-s_k r_\phi(X_k,t_k).
\]

Здесь \(s_k^2=v_k\) — accumulated forward variance bridge. Inference начинается с \(X_K=R(C)\). Для coarse transition \(k\to j<k\) в нашей реализации:

\[
X_j=(1-v_j/v_k)\widehat Y_{0|k}+(v_j/v_k)X_k+\eta_{k,j},
\]

где \(\eta_{k,j}=\sqrt{v_j(1-v_j/v_k)}\,\xi\) при \(j>0\), а на последнем переходе к индексу0 шум отключён. Выход — последнее состояние sampler'а, не просто последняя clean estimate. Точное \(A X_{out}=C\) эта процедура для pooling SR не гарантирует. В official I²SB используются те же normalized-label/clean-estimate идеи и posterior sampler; финальная pooling-проекция в наших метриках добавлена отдельно. Специфическое mask enforcement в других задачах, например inpainting, не означает наличие такой projection в SR.

Первичные источники: [RRDBNet forward](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py), [I²SB clean estimate](https://github.com/NVlabs/I2SB/blob/master/i2sb/runner.py), [I²SB sampler](https://github.com/NVlabs/I2SB/blob/master/i2sb/diffusion.py).

### Куда проецируем

При нашей noiseless постановке \(C=AY\) допустимые HR образуют аффинное множество

\[
\mathcal S_C=\{X: AX=C\}=A^\dagger C+\ker A.
\]

Для full-row-rank A евклидова ортогональная проекция равна

\[
\Pi_C(V)=\arg\min_{X:AX=C}\|X-V\|_2^2
=V+A^T(AA^T)^{-1}(C-AV).
\]

Для неперекрывающегося pooling4×4 \(AA^T=I/16\), поэтому \(A^\dagger=16A^T=R\), где R — nearest replication. Получаем реализованную формулу

\[
\Pi_C(V)=V+R(C-AV)=RC+(I-RA)V.
\]

Bicubic U и nearest R здесь имеют разные роли: U — исходная плавная база сети, R — точный pseudoinverse оператора block averaging. Bicubic upsampling не является этим pseudoinverse.

В каждом RGB-канале одного4×4 блока к16 пикселям прибавляется одна и та же поправка: LR-пиксель минус среднее текущего блока. Среднее становится точно правильным, взаимные разности16 пикселей сохраняются. Внутри блока остаются15 свободных направлений. Для всего HR128 RGB множество имеет49 152−3 072=46 080 степеней свободы. При C≠0 это аффинное множество, а линейным является \(\ker A\). Равенство средних само по себе не ограничивает натуральность текстур.

Входной LR известен, и проекция встраивает этот факт в модель. Во время обучения Jacobian по raw prediction равен \(I-RA\): сеть влияет на направления с нулевым средним по блокам. Это не обязательный компонент MGM; это выбранное нами ограничение для конкретной SR-деградации.

### Что гарантировано, а что установлено экспериментом

Если истинный Y удовлетворяет AY=C, то по ортогональности

\[
\|V-Y\|_2^2=\|\Pi_C(V)-Y\|_2^2+\|V-\Pi_C(V)\|_2^2.
\]

Следовательно, projection не увеличивает MSE **того же фиксированного raw prediction до clipping**. Это не гарантия LPIPS/SSIM/FID и не сравнение результатов двух разных обучений. После clipping в [-1,1] точная consistency может нарушиться, и указанное тождество не является прямой гарантией порядка clipped PSNR. Нужны правильный A и noiseless C; при JPEG, шуме или неизвестном blur hard constraint может навязать неверное условие.

В сохранённой серии `sr_4c20214`, 50 validation фото, mean по3 seed:

| Метод | Native PSNR→projected | Native LPIPS→projected |
|---|---|---|
| ESRGAN | 25.524→26.035 | 0.1532→0.1491 |
| I²SB NFE64 | 24.837→25.826 | 0.2517→0.1938 |

Это положительный эффект финальной postprocessing projection в данной серии. Причинный эффект встроенной projection на MGM training ещё не измерен: для него требуется отдельная on/off абляция. У MSE/MP/MGM projection уже внутри G, повторная финальная projection почти тождественна. Native-vs-projected таблицы полезны для прозрачного сравнения с ESRGAN/I²SB, но одинаковое final postprocessing не делает training objectives/архитектуры одинаковыми.

### Прямые прецеденты

- **RND / GAN Prior Based Null-Space Learning for Consistent Super-resolution (AAAI2023)**: формула \(A^\dagger C+(I-A^\dagger A)V\), average pooling и его pseudoinverse. Для нашего pooling она математически совпадает с используемой projection. [Статья](https://ojs.aaai.org/index.php/AAAI/article/view/25372), [авторский код/формулы](https://github.com/wyhuai/RND).
- **Explorable Super Resolution (CVPR2020)**: Consistency Enforcing Module оборачивает SR-сеть и обеспечивает совпадение downsampled HR с LR. [CVF paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bahat_Explorable_Super_Resolution_CVPR_2020_paper.html).
- **DDNM (ICLR2023)**: range/null decomposition корректирует clean estimate внутри reverse diffusion, а затем sampler делает следующий переход. Это связано с нашей projection, но размещено внутри цепочки, а не только в конце. [Статья, eq.13](https://arxiv.org/html/2212.00490).
- **Geiss & Hardin, Downsampling Enforcement (2023; preprint2020)**: дифференцируемый выходной оператор сохраняет block averages в научных SR-задачах. [Авторский институт](https://www.pnnl.gov/publications/strictly-enforcing-invertibility-and-conservation-cnn-based-super-resolution).

Projection следует представлять как известный data-consistency prior нашей SR-адаптации. Исходная unconditional MGM не задаёт LR-оператор A и не требует такой проекции; novelty адаптации не может состоять в самой формуле projection.
