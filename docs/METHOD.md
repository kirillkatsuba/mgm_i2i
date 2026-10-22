# Method

This project adapts MGM to paired super-resolution. It introduces a conditional implementation and evaluates architectural/training choices; it does not claim a new unconditional divergence or a novel projection operator.

## Conditional objective

Given HR $Y$ and LR $C=DY$, generate $X_0=G_\theta(C,Z)$ independently of $Y$ conditional on $C$. Both interpolated endpoints are HR; LR is their shared condition.

$$
t\sim\mathcal U[0,1/2],\quad B\sim\operatorname{Bernoulli}(1/2),\qquad
\widetilde X_t=\begin{cases}(1-t)X_0+tY,&B=0,\\tX_0+(1-t)Y,&B=1\end{cases}
+\sqrt{\lambda t(1-t)}\,\varepsilon.
$$

Here $Z$ and $\varepsilon$ are independent standard Gaussian maps. The field sees $(C,t,\widetilde X_t)$, not $B$. The displacement target is always $Y-X_0$.

$$
\min_\theta\max_\psi\ \mathbb E_C\int_0^{1/2}
\mathbb E\left[2\langle f_\psi(C,t,\widetilde X_t),Y-X_0\rangle
-\|f_\psi(C,t,\widetilde X_t)\|^2\right]dt.
$$

For field updates, detach $X_0$ and minimize displacement regression MSE. For generator updates, freeze field parameters while retaining gradients through its input and through $Y-X_0$. Our implementation reuses the HR batch with fresh latent/interpolation draws between the two updates. No pixel/perceptual penalty is added to the main MGM objective.

The population motivation is conditional distribution matching under the original MGM assumptions. Finite networks, imperfect minimax optimization and one observed HR per LR do not establish conditional coverage or convergence.

## Generator and field

$$
G_v(c,z)=\Pi_c\left(U(c)+h_v(F_\theta([U(c),z]))\right),\quad
h_1=\tanh,\quad h_2=\mathrm{id}.
$$

Both generators use a residual U-Net with widths 120/240/480, GroupNorm, SiLU and encoder–decoder skips. Generator input concatenates bicubic LR and a Gaussian map: six channels at 128×128. The field concatenates bicubic LR and mixed HR. v1 adds four spatial time maps; v2 injects a sinusoidal embedding/MLP into residual blocks. Generator parameters: 16.02M; field parameters: 16.03M (v1), 17.29M (v2). Only G is needed at inference.

For block averaging, nearest replication $R$ is the Moore–Penrose inverse of $D$:

$$
\Pi_c(v)=v+R(c-Dv)=Rc+(I-RD)v,\qquad D\Pi_c(v)=c.
$$

This is the Euclidean projection onto $\{x:Dx=c\}$. It fixes block means and leaves null-space details to the network. It follows existing [range–null space SR](https://ojs.aaai.org/index.php/AAAI/article/view/25372). Consistency holds before clipping; it does not guarantee correct textures. We did not train a projection-off MGM control.

## Training changes and ablations

1. **Warm-started field:** reuse compatible hidden features from the MSE-warmed generator, remap RGB input filters and reset displacement/time-injection heads.
2. **Staged training:** 50 epochs MSE warmup at $Z=0$, 50 field-only, 80 joint MGM. Warmup batch 16; field/joint batch 32 without accumulation. Main Adam learning rate $10^{-5}$, betas (0.8, 0.999), EMA 0.995, gradient clipping 10 and AMP.
3. **Noise sweep:** use separate coefficients $\lambda_f$ and $\lambda_G$ in the observation noise for field/G updates. Each architecture tests no noise, matched 0.01, critic-only 0.01, matched 0.05 and midpoint critic-only 0.01, with three seeds. The midpoint control fixes $t=1/2$.

Matched-noise recipes share one observation law. Critic-only noise uses different laws for the two updates and is an empirical modification of the shared minimax formulation. v1/v2 change both residual activation and field time conditioning, so their comparison does not isolate either change.

No interpolant noise had the lowest mean validation LPIPS within each architecture. v2/no-noise was selected at 0.14375±0.00056. Its test LPIPS is 0.13908±0.00238. Latent $Z$ remains random; no-noise does not mean deterministic generation.

Implementation: [models](../src/mgm_i2i/cluster/models.py), [MGM updates](../src/mgm_i2i/cluster/mgm_training.py), [losses](../src/mgm_i2i/losses.py).
