# Вопросы к защите MGM для I2I

Ответы относятся к [теоретическому разбору](THEORY_AND_RESEARCH_PLAN.md) и [сценарию презентации](PRESENTATION_20_MIN.md). Текущая подготовка не включает новые технические запуски.

## 1. Что именно делает метод one-step?

После обучения inference состоит из одного вызова $G(c,z)$. Поле, real target, interpolation time и ODE/SDE solver при получении изображения не используются. Обучение при этом содержит много итераций и auxiliary network.

## 2. Какое поле нулевое при совпадении распределений?

Условное среднее displacement при заданном наблюдении: $m_t(q)=\mathbb E[X_1-X_0\mid Q_t=q]$. Это не разность каждой пары и не обычное FM-поле во всех временах. Для исходной несмешанной по веткам линейной конструкции нужная симметрия возникает в середине.

## 3. Зачем скрывать flip?

Наблюдение должно симметрично смешивать две ветки. Если сообщить сети ветку или эффективное время, она сможет использовать orientation-specific posterior. Базовое время сохраняем, индекс ветки исключаем. При разных endpoint-распределениях ветку можно статистически угадывать по смеси; «скрытый» не означает полную information independence.

## 4. Почему при flip нельзя изменить знак target?

Target MGM всегда real минус fake. При перестановке endpoints и branch он меняет знак, а observation остаётся той же. Это создаёт нулевой conditional mean. Изменение знака по derivative ветки создаёт другое поле и разрушает данный argument.

## 5. Почему обычный интеграл квадрата FM velocity не подходит?

Одинаковые endpoint-распределения могут иметь меняющееся marginal-распределение интерполянта. Для независимых N(0,1) variance сжимается к середине. Соответствующее FM-поле вне середины ненулевое. Поэтому naive integral может штрафовать уже совпавшие endpoints.

## 6. Где нужна независимость концов?

В обменности при совпадении распределений и в factorization при доказательстве обратного направления. Равные marginals зависимой необменной пары не дают midpoint-свойство автоматически. В I2I требуем независимость generated и real HR при фиксированном C, обеспеченную независимым latent.

## 7. Чем предположения midpoint и интегральной теорем отличаются?

Обратная midpoint-теорема в статье предполагает почти наверное ограниченные независимые endpoints. Интегральная и generalized теоремы используют независимость, конечные вторые моменты и соответствующие symmetric/regular stochastic-interpolant conditions. Не следует переносить одну формулировку на другую без проверки.

## 8. Почему не учить generator тем же MSE, что field?

Для фиксированного generator квадрат displacement является константой по параметрам field. При изменении generator он меняется. Field regression loss равен squared displacement минус variational objective. Его минимизация по generator задаёт другую цель.

## 9. Как одновременно фиксировать поле и сохранять его градиент?

Фиксируем trainable параметры field. Его output остаётся функцией HR-mixture, которая зависит от generator. Autograd через input нужен. При `no_grad()` вокруг всего вызова field исчезает Jacobian term generator gradient.

## 10. Значит ли regression loss поля, равный нулю, успех?

Нет. При совпавших endpoint-распределениях оптимальный conditional mean равен нулю, но displacement независимых samples обычно ненулевой. Оптимальная regression error содержит conditional variance. Нулевой loss может также означать вырожденную пару, а не хорошее distribution matching.

## 11. Почему generator loss может быть отрицательным?

Для произвольного field variational objective равен истинному squared conditional mean минус approximation error field. Вторая часть может оказаться больше первой. Неотрицательна divergence после максимизации по полному классу, а не любое значение neural loss.

## 12. Что меняется в conditional адаптации?

Generator получает C и latent, field получает C, базовое время и HR-mixture. Endpoints относятся к одному C. Теорема применяется при фиксированном условии, затем divergence усредняется по C. Это популяционное утверждение, не гарантия оптимизации конечной сети по finite dataset.

## 13. Почему нельзя смешивать LR и HR?

Наш MGM сравнивает conditional распределения generated HR и real HR. LR служит входным условием. Upsampling LR и использование его как endpoint сравнило бы другое распределение и не было бы данной адаптацией MGM к generator outputs.

## 14. Откуда берётся неопределённость, если на вход один target?

В популяционной задаче оператор теряет информацию и разные HR могут соответствовать одному LR. В конечном dataset уникальный LR может иметь только один наблюдённый target. Learning conditional variability требует generalization и общих закономерностей. Это причина использовать toy с точно известными альтернативами и ограничивать выводы по SR.

## 15. Почему правильный sampler может проиграть MSE baseline по PSNR?

MSE минимизируется conditional mean. Самостоятельно sampled правильная мода может отличаться от единственной target-моды, хотя имеет правильную conditional probability. На toy paired MSE mean predictor равна 0.1225 на пиксель, у правильного независимого sampler 0.245.

## 16. Можно ли добиться большого diversity, добавляя шум?

Да. Поэтому diversity является diagnostic statistic. Правильность нужно проверять distributional quality, off-mode distance, consistency и perceptual metrics. В toy правильная high-frequency std равна 0.35, но совпадение одной std всё равно не определяет всё распределение.

## 17. Почему недостаточно off-mode distance или mode balance?

Sampler одной правильной моды имеет off-mode distance=0, но неверные probabilities. Output с малым шумом около нуля может иметь balanced signs, но не попадать в правильные modes. В совокупности эти метрики полезны. Conditional W1 сравнивает всю одномерную toy-меру.

## 18. Почему empirical W1 хорошего sampler не равен нулю?

За конечное число независимых draws число samples каждой моды случайно. Для exact sampler $\widehat W_1=2a|K/n-1/2|$. Analytical normal approximation даёт ожидаемое значение около 0.0175 при a=0.35 и n=256. Это sampling floor, а не bias самого exact distribution.

## 19. Не решила ли проекция всю задачу за generator?

Проекция фиксирует low-frequency часть. High-frequency modes и их probabilities остаются неизвестными. Все аналитические toy baselines согласованы с LR, но два из них имеют неправильное conditional distribution. Contribution projection надо раскрывать и применять одинаково к обучаемым baselines.

## 20. Можно ли восстановить C из смеси при проекции?

Из endpoints и безшумной смеси можно. С изотропным interpolant noise получается $AQ_t=C+\sigma_tA\varepsilon$. Поэтому из noisy observation точное восстановление больше не следует. Nullspace-noise предложен как отдельная идея, которая может сохранить constraint observation.

## 21. Зачем field получает C, если HR уже определяет LR?

В данной известной degradation с exact-consistent endpoints это может быть частично redundant information. Но noisy mixtures, imperfect consistency и другие paired задачи меняют ситуацию. Condition on/off рассматриваем вместе с projection и noise, а не делаем универсальный вывод из одной ablation.

## 22. Почему chosen SR использует average pooling?

Оператор известен, воспроизводим и допускает простую точную consistency projection. Это делает постановку объяснимой при небольшом бюджете. Ограничение: результаты не являются стандартным bicubic-SR benchmark и не устанавливают качество на real-world деградациях.

## 23. Достаточно ли одного Gaussian noise channel?

Это compact architectural choice для pilot, а не доказательство достаточной expressivity. Conditional uncertainty может иметь более высокую effective dimension, чем 32×32 scalar latent. Малый generator, ограниченный residual и latent dimension могут ограничивать match. После базового сравнения можно отдельно изучить latent channels и capacity с общим architecture budget для методов.

## 24. Почему GAN называется conditional GAN, а не pix2pix?

Он использует conditioning и небольшой patch discriminator с logistic objective. Полная архитектура, loss и training protocol pix2pix не воспроизводятся. Статья pix2pix даёт контекст выбора adversarial baseline.

## 25. Что именно измеряет LPIPS в данном pilot?

Feature distance к одному target с AlexNet v0.1. Inputs nearest-resized с 32×32 до 64×64. Это явный local comparative protocol. Более низкий score не доказывает правильные probabilities всех допустимых outputs или преимущество в любом human evaluation.

## 26. Почему здесь не основной FID?

FID относится к marginal feature statistics, не к individual input-output correspondence. Малый sample count затрудняет интерпретацию. Кроме того, implementation с fixed extractor/protocol пока отсутствует. В исходной статье FID относится к другой unconditional постановке с 50K generated samples.

## 27. Чем critic-only noise отличается теоретически?

Field обучается на noisy observations, а generator получает field на noiseless mixtures. Два шага не являются непосредственной максимизацией и минимизацией одной общей показанной variational objective. Это эмпирический design choice, который в статье дал лучший FID. Его перенос в I2I нужно проверять.

## 28. В чём математическое улучшение antithetic flip?

Вычисление обеих веток является conditional averaging по B. Оно сохраняет expectation и убирает variance, связанную с branch sampling, при finite-variance условиях. Но field computation возрастает. Нужно сравнение по полной стоимости, а не только по step count.

## 29. Можно ли доказать, что MGM побеждает GAN?

Представленная теория устанавливает критерий нуля divergence. Она не сравнивает practical optimization, capacity или score обученных сетей. В прежнем CIFAR pilot GAN имеет ниже LPIPS, чем MGM, а MSE выше PSNR. Для conclusion нужно зафиксированное достаточное исследование.

## 30. Что будет, если field слабое?

Оценка divergence может быть слишком низкой и generator может использовать ошибки field. Малая variational loss перестаёт быть свидетельством хорошего matching. Можно изучать field capacity и update ratio, но показатели качества outputs остаются основными.

## 31. Есть ли гарантия convergence alternating minimax?

В рассматриваемой статье правильность population divergence не равна гарантии convergence нейросетевых игроков. Возможны instability, sensitivity к initialization, learning rates и balance updates. Это прямо относится к ограничениям метода.

## 32. Какой результат можно защищать уже сейчас?

Корректную conditional формулировку, аналитический toy-вывод, explicit evaluation protocol и ранее выполненные ограниченные проверки. Нельзя утверждать завершённое superiority на SR или новое confirmed improved MGM. Более длинные runs и controlled ablations остаются вычислительной частью будущей работы.

## 33. Что раскрывать про AI и вычисления?

Какие инструменты использовались, какой объём помощи они дали, фактическую подписку и лимиты, что проверено самостоятельно, CPU/GPU и общее время вычислений. Важно учесть warmup, tuning и неудачные попытки. В текущей теоретической подготовке новых train/evaluation запусков не было.

## 34. Какой следующий эксперимент наиболее содержателен?

Для общей MGM-идеи: antithetic против обычного flip при одинаковой полной стоимости. Для conditional SR: isotropic против nullspace-noise при сохранении других настроек, после реализации и проверки observation-конструкции. При ограниченном бюджете сначала достаточно controlled toy comparison и хорошо описанного SR-pilot.
