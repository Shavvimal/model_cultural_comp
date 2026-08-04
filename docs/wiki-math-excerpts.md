# Wiki math excerpts for the Method section

Harvested from Shav's own wiki at `/Users/shav/Code/ShavWiki/pages/` (all of
`artificial-intelligence/`, plus a sweep of `cloud-mlops/`) for the ORACLE @ EMNLP 2026 paper
(`/Users/shav/Code/next-mdx-blog/content/blog/cultural-bias-2026.mdx`), against the methods
critique in `/Users/shav/Code/model_cultural_comp/docs/statistical-review.md` §2.1, §2.3, §2.4.

**What this file is.** Material that already exists in Shav's notation and voice, debugged, and
worth lifting rather than re-deriving. Equations below are lightly adapted so they compile in
KaTeX: `$$` display blocks only, `aligned` / `cases` / `pmatrix` environments (KaTeX-safe), no
bare `align`, `\operatorname*{arg\,max}` instead of `\argmax`.

**What this file is not.** A survey of the wiki. The wiki is a taught-course maths/ML reference
(Year 1–2 engineering maths, Andrew Ng deep-learning notes, a Frank Kane–style applied-ML
track). It is strong on linear algebra, classical estimation theory and elementary statistics,
and has **nothing at all** on the specific machinery this paper's reviewers will press on. See
"Gaps" at the end — that list is longer than the finds, and it is the honest summary.

Ordered by where the material lands in the paper.

---

## 1. §3.2 — Standardization, and why the 2024 defect was a defect

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/real-world-data/data-cleaning-normalisation.mdx`

The Z-transform in his own notation. This is exactly the $(x-\mu)/\sigma$ step in the paper's
projection equation, and having it stated as a named, standard preprocessing operation lets §3.2
say "the standardization" rather than introducing it ad hoc.

> ### Types of Normalization
>
> **Z-score Normalization**: Focuses on how far each point is from the mean in terms of standard
> deviation.
>
> $$
> Z = \frac{X - \text{mean}}{\text{std. deviation}}
> $$
>
> **Min-Max Scaling**: Transforms data to fit within a specific range, usually 0 to 1.
>
> $$
> X' = \frac{X - \text{Min}}{\text{Max} - \text{Min}}
> $$
>
> ### When to Normalize?
>
> Always check the documentation for the specific model you're using. Some models need
> normalized data; others don't. In scikit-learn's PCA implementation, they have a `whiten`
> parameter that will automatically normalize your data for you.

**Adaptation for the paper.** Write the projection with $\mu,\sigma$ subscripted by item and
frozen at fit time, so the "fixed at fit time" claim is visible in the notation rather than only
in the prose:

$$
z_j = \frac{x_j - \mu_j}{\sigma_j}, \qquad
\text{score} = \mathbf{z}\, C R, \qquad j = 1,\dots,10
$$

with $\mu_j,\sigma_j$ the training means and standard deviations, $C$ the $10\times 2$ principal
axes and $R$ the stored $2\times2$ rotation. The 2024 defect is then legible as: the model path
computed `score` from $\mathbf{x}$ rather than $\mathbf{z}$.

**Bonus — the "why" argument, already written.**

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/feature-extraction/pca.mdx` §2

> Dimensional reduction using PCA consists of finding the features that maximize the variance.
> If one feature varies more than the others only because of their respective scales, PCA would
> determine that such feature dominates the direction of the principal components. […] we find
> that the "proline" feature dominates the direction of the first principal component without
> scaling, being about two orders of magnitude above the other features. This is contrasted when
> observing the first principal component for the scaled version of the data, where the orders
> of magnitude are roughly the same across all the features.

This is directly usable in §3.2's validation paragraph: the ten IVS items have wildly different
native ranges (A165 is 1–2, F063 and F118 are 1–10), so the scale-domination argument is not
hypothetical here — it is the mechanism by which the unstandardized 2024 model projection went
wrong, and it explains why `llama3:70b` (which sat where the high-range items pulled it) moved
by 2.05 units under the correction while the others barely moved.

---

## 2. §3.2 — PCA mechanics: eigendecomposition of the covariance matrix

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/feature-extraction/pca.mdx` §§1–5

Prose, not LaTeX, but a crisp five-step statement of the algorithm that the paper can compress
into two sentences:

> PCA creates the new variables by transforming the original (mean-centered) observations
> (records) in a dataset to a new set of variables (dimensions) using the eigenvectors and
> eigenvalues calculated from a covariance matrix of your original variables. Step-by-step:
>
> 1. Centering the values of all of the input variables
> 2. Potential scaling of the data, depending on the units of the variables
> 3. Calculating the covariance matrix of the data
> 4. Calculating the eigenvectors and eigenvalues of the covariance matrix
> 5. The principal components (eigenvectors) are sorted by descending eigenvalue.

> Following the identification and selection of principal components, original data observations
> are converted to these components through the creation of a projection matrix. This projection
> matrix is just the selected eigenvectors concatenated to a matrix.

That last sentence is the definition of $C$ in the paper's projection equation, in his own words.

**The linear-algebra guarantee underneath it.**

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/matrices/diagonalisation-of-symmetric-and-hermitian-matrices.mdx`

Worked and proved. Justifies, without hand-waving, that the covariance matrix admits an
orthonormal eigenbasis — the reason the two retained components are orthogonal *before* rotation
(and therefore the reason the review's §1c point about post-rotation correlation is a statement
about $R$, not about $C$).

> Recall that a matrix is symmetric iff it is equal to its transpose, i.e.
>
> $$
> A = A^{\top}
> $$
>
> The diagonalisation of symmetric matrices is particularly easy as all eigenvalues are real and
> corresponding eigenvectors are orthonormal (these are mutually orthogonal vectors of unit
> length).
>
> **Theorem.** For a symmetric matrix $A$, we may diagonalise $A$ with respect to an orthonormal
> set of eigenvectors.
>
> **Theorem.** For an $n \times n$ Hermitian matrix $A$ all eigenvalues are real.
>
> *Proof.* Consider an eigenvector $\boldsymbol{x}$ with a possibly complex eigenvalue $\lambda$.
> Starting from $A\boldsymbol{x} = \lambda\boldsymbol{x}$ and multiplying by the Hermitian
> conjugate of $\boldsymbol{x}$,
>
> $$
> \boldsymbol{x}^{\dagger} A \boldsymbol{x} = \lambda\, \boldsymbol{x}^{\dagger}\boldsymbol{x}
> $$
>
> Taking the complex conjugate of both sides and using $(AB)^{\dagger} = B^{\dagger}A^{\dagger}$
> together with $A^{\dagger}=A$,
>
> $$
> \boldsymbol{x}^{\dagger} A \boldsymbol{x} = \bar{\lambda}\, \boldsymbol{x}^{\dagger}\boldsymbol{x}
> $$
>
> Subtracting,
>
> $$
> \begin{aligned}
> (\lambda - \bar{\lambda})\, \boldsymbol{x}^{\dagger}\boldsymbol{x}
>   &= \boldsymbol{x}^{\dagger} A \boldsymbol{x} - \boldsymbol{x}^{\dagger} A \boldsymbol{x} \\
>   &= 0 .
> \end{aligned}
> $$
>
> Since $\boldsymbol{x}$ is an eigenvector (and therefore nonzero),
> $\boldsymbol{x}^{\dagger}\boldsymbol{x} \neq 0$, which implies $\lambda = \bar{\lambda}$ and the
> eigenvalues are real. $\square$
>
> [Eigenvectors of distinct eigenvalues are orthogonal by the same manipulation.] Recall the
> diagonalisation is $A = PDP^{-1}$, where $P$ is made up of the eigenvectors. Since $P$ is a
> matrix with orthonormal columns then it is orthogonal, which satisfies $P^{-1} = P^{\top}$, and
> $D$ is the diagonal matrix of eigenvalues. The biggest advantage of symmetric matrices is that
> we do not need to find the inverse of the matrix of eigenvectors using Gauss–Jordan — we can
> just take the transpose.

For the paper, one line suffices, but this is where it comes from:

$$
\Sigma = C \Lambda C^{\top}, \qquad C^{\top} C = I, \qquad
\Lambda = \operatorname{diag}(\lambda_1, \lambda_2), \quad \lambda_1 \ge \lambda_2
$$

---

## 3. §3.2 — The rotation: $R$ is orthogonal, and what that does to the score covariance

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/matrices/definitions-and-rules.mdx` (orthogonal matrices, Example 10.2)

This is the most directly load-bearing find for the review's §2.1, which needs the paper to (a)
state the rotation as an explicit orthogonal $R(\theta)$ with a reportable angle ($-41.76°$), and
(b) derive $\operatorname{cov}(SR) = R^{\top}\Lambda R$ and show it is *not* diagonal.

> For the product $AA^{\top}$ to be equal to the identity matrix, we deduce the following:
>
> $$
> \left(A A^{\top}\right)_{ij} = \sum_{k=1}^{n} a_{ik} a_{jk} =
> \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}
> $$
>
> Denoting the $i^{\text{th}}$ row vector of $A$ by $\boldsymbol{r}_i$, we have
>
> $$
> \boldsymbol{r}_i \cdot \boldsymbol{r}_j =
> \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}
> $$
>
> i.e. the row vectors of $A$ are mutually orthogonal unit vectors. Note that we also need
> $A^{\top}A = I$, which implies that the column vectors of $A$ must also be mutually orthogonal
> unit vectors.
>
> **Example.** Show that the following matrix is orthogonal:
>
> $$
> A = \begin{pmatrix}
> \cos\theta & -\sin\theta & 0 \\
> \sin\theta & \cos\theta & 0 \\
> 0 & 0 & 1
> \end{pmatrix}
> $$
>
> *Solution.* Define $\boldsymbol{r}_1 = (\cos\theta, -\sin\theta, 0)$,
> $\boldsymbol{r}_2 = (\sin\theta, \cos\theta, 0)$, $\boldsymbol{r}_3 = (0,0,1)$. Then
>
> $$
> \boldsymbol{r}_1 \cdot \boldsymbol{r}_1 = \cos^2\theta + \sin^2\theta = 1, \quad
> \boldsymbol{r}_2 \cdot \boldsymbol{r}_2 = 1, \quad
> \boldsymbol{r}_3 \cdot \boldsymbol{r}_3 = 1
> $$
>
> and
>
> $$
> \boldsymbol{r}_1 \cdot \boldsymbol{r}_2 = \cos\theta\sin\theta - \cos\theta\sin\theta = 0,
> \quad \boldsymbol{r}_1 \cdot \boldsymbol{r}_3 = 0, \quad \boldsymbol{r}_2 \cdot \boldsymbol{r}_3 = 0
> $$
>
> so $A$ is orthogonal, and it follows that $A^{-1} = A^{\top}$.

**Adaptation for the paper (§3.2, and the new sensitivity appendix the review asks for).** Drop
to two dimensions and give the rotation an angle:

$$
R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix},
\qquad R^{\top} R = I, \qquad \hat{\theta} = -41.76°
$$

Then the review's §1c point falls straight out of $\operatorname{cov}(S) = \Lambda$ and the
bilinearity of covariance (§6 below):

$$
\operatorname{cov}(S R) = R^{\top} \Lambda R
$$

which is diagonal only if $\Lambda \propto I$. With $\lambda = (2.484, 1.219)$ and
$\theta = -41.76°$ this gives rotated variances $(1.923, 1.780)$, covariance $0.628$, and

$$
r = \frac{0.628}{\sqrt{1.923 \times 1.780}} = 0.34
$$

Whitening first ($S \mapsto S\Lambda^{-1/2}$, so $\operatorname{cov} = I$) makes $R^{\top} I R = I$
and restores exact orthogonality — which is also what moves the estimator to the exact Rohe–Zeng
form. That one equation carries the whole "either whiten or report $r = 0.34$" decision.

---

## 4. §3.2 — MLE scaffolding for PPCA/EM

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/probability-and-statistics/sampling-and-estimation/estimators-and-point-estimation.mdx`

The wiki has **no EM derivation and no latent-variable model** (see Gaps). What it does have is a
clean, worked MLE framework that PPCA-by-EM can be introduced as an instance of — useful for one
or two sentences that make Tipping & Bishop feel like a special case of something standard rather
than a black box.

> Given some sample dataset $D = \{D_1, D_2, \ldots, D_n\}$, we can define a likelihood function
> $L(\theta)$ as the probability that the sample data $D$ was generated given some value of the
> model parameter $\theta$:
>
> $$
> L(\theta) = P(D \mid \theta)
> $$
>
> Since we assume our samples $D$ are i.i.d., this greatly simplifies to:
>
> $$
> L(\theta) = P(D \mid \theta) = \prod_{i=1}^{n} f_X\!\left(D_i \mid \theta\right)
> $$
>
> We define the Maximum-Likelihood Estimate of a parameter $\theta$ to be:
>
> $$
> \hat{\theta}_{\mathrm{MLE}} = \operatorname*{arg\,max}_{\theta} L(\theta)
> $$
>
> In practice it is advantageous to optimise the logarithm of this expression instead, i.e. the
> so-called log-likelihood function $\mathcal{L}(\theta) = \log L(\theta)$, as it transforms the
> series of multiplications into a summation, which results in a much easier function to optimise.
>
> **MLE procedure for an i.i.d. sample $D$:**
>
> 1. Write down the likelihood $L(\theta) = \prod_{i=1}^{n} f_X(D_i \mid \theta)$ and take
>    logarithms: $\mathcal{L}(\theta) = \log L(\theta)$
> 2. Maximise $\mathcal{L}(\theta)$ with respect to $\theta$ to obtain $\hat{\theta}$
> 3. Verify that the obtained $\hat{\theta}$ is indeed the maximum and within the correct range.

**Adaptation for §3.2.** The one sentence this buys: *we fit by maximum likelihood, maximising the
observed-data log-likelihood $\mathcal{L}(\theta) = \sum_i \log f(\mathbf{x}_i^{\mathrm{obs}} \mid
\theta)$ over the observed entries only, which under a Gaussian latent model has no closed form
and is maximised by EM.* Note his step 1–3 discipline does not transfer: EM has no closed-form
step 2, which is the whole point, and the paper should say so.

**Also here — the bias/variance decomposition, stated properly.** Better than the informal
`Error = Bias² + Variance` in `real-world-data/bias-variance-trade-off.mdx`; use this version if
the paper needs to justify EM imputation over listwise deletion as a bias-for-variance trade:

> $$
> \operatorname{MSE}(\hat{\theta}) = E\!\left[(\hat{\theta} - \theta)^2\right]
> $$
>
> Starting from the definition of MSE, we can expand and contract the quadratic terms to derive a
> useful expression:
>
> $$
> \begin{aligned}
> \operatorname{MSE}(\hat{\theta})
>   &= \left(E\!\left[\hat{\theta}^2\right] - E[\hat{\theta}]^2\right) + \left(E[\hat{\theta}] - \theta\right)^2 \\
>   &= \operatorname{Var}(\hat{\theta}) + \operatorname{Bias}(\hat{\theta})^2
> \end{aligned}
> $$
>
> where $\operatorname{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta = 0$ if 'unbiased', and
> $\operatorname{Var}(\hat{\theta})$ is how sensitive an estimate is to randomness inherent in the
> data. An estimator is **consistent** if $\operatorname{MSE}(\hat{\theta}) \to 0$ as
> $n \to \infty$.

---

## 5. §3.2 — Why not listwise deletion (the missingness argument, already written)

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/feature-engineering.mdx` — "Imputing Missing Data"

No maths, but the argument the paper makes in one clause is here in full, and in a voice that
transfers. Both halves are load-bearing: the first is why mean-imputation is not an option, the
second is why dropping is not either.

> ### Mean Replacement
>
> Replace missing values with the mean value from the rest of the column […] Fast & easy, won't
> affect mean or sample size of overall data set. But it's generally pretty terrible. Only works
> on column level, **misses correlations between features** — if there is a relationship between
> age and income, you may miss this. E.g. could say a 10 year old is earning 50K a year. It is a
> very naïve approach.
>
> ### Dropping
>
> If not many rows contain missing data, and dropping those rows doesn't **bias** your data […]
> it could be a reasonable thing to do. What if there's an actual relationship between which rows
> are missing data and some other attribute of those observations? E.g. we're looking at income:
> there might be a situation where people that have very high or very low incomes are more likely
> to not report it. By removing, or dropping, all of those observations, you're actually removing
> a lot of people that have very high or low incomes from your model and that might have a very
> bad effect on the accuracy of the model you end up with. […] But it's never going to be the
> right answer for the "best" approach. Almost anything is better.

**How it fits.** §3.2's justification sentence currently reads "Discarding incomplete rows would
lose 28.4% of the data non-randomly; imputation would bias the component estimates." The second
clause is the weaker half and the review pushes back on it (§2.3: "'missing by design is the
benign case' is a stronger sentence"). The wiki's framing is sharper and survives the pushback:
*mean/marginal imputation destroys exactly the cross-item covariance structure the components are
estimated from; listwise deletion selects on the design.* PPCA-EM is the estimator that does
neither, because it imputes from the joint model rather than from the margin.

---

## 6. §3.4 — The bootstrap variance identity (the single most useful find)

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/probability-and-statistics/random-variables-and-probability-distributions/multivariate-distributions.mdx` — "Covariance"

Statistical review §2.4(a) is the most consequential methods criticism in the paper, and its
central equation, $\operatorname{Var}(p) = \sum_j \sum_k w_j w_k \operatorname{Cov}(\bar{x}_j,
\bar{x}_k)$, is exactly the $n$-term generalisation of the bilinearity result Shav has already
worked through here for two variables. Lift the properties, then generalise in one line.

> The covariance of two random variables $X$ and $Y$ is defined as
>
> $$
> \operatorname{Cov}[X, Y] = E\big[(X - E[X])(Y - E[Y])\big] = E[XY] - E[X]E[Y]
> $$
>
> Covariance is a measure of linear association between two random variables.
>
> Consider the case where $X$ and $Y$ are independent random variables:
>
> $$
> \begin{aligned}
> E[XY] &= \int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty} xy\, f_{X,Y}(x,y)\, dx\, dy
>        = \int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty} xy\, f_X(x) f_Y(y)\, dx\, dy \\
>       &= \int_{-\infty}^{\infty} x f_X(x)\, dx \int_{-\infty}^{\infty} y f_Y(y)\, dy = E[X]E[Y]
> \end{aligned}
> $$
>
> which results in $\operatorname{Cov}[X,Y] = 0$. However, we must be careful to note that the
> converse does not hold! That is, if $\operatorname{Cov}[X,Y] = 0$ we cannot conclude that $X$
> and $Y$ are necessarily independent.
>
> For random variables $X, Y, Z$ and constants $a, b, c, d$:
>
> $$
> \begin{aligned}
> &\text{1.} \quad \operatorname{Cov}[X, a] = 0 \\
> &\text{2.} \quad \operatorname{Cov}[aX + b,\; cY + d] = ac \operatorname{Cov}[X, Y] \\
> &\text{3.} \quad \operatorname{Cov}[X + Y,\; Z] = \operatorname{Cov}[X, Z] + \operatorname{Cov}[Y, Z]
> \end{aligned}
> $$
>
> We say that covariance is a **bilinear operator**, in the sense that it is linear in both its
> inputs. Notice that $\operatorname{Cov}[X, X] = \operatorname{Var}[X]$, so the second property
> above explains why $\operatorname{Var}[aX + b] = a^2 \operatorname{Var}[X]$. The third property
> implies that
>
> $$
> \begin{aligned}
> \operatorname{Var}[X + Y]
>   &= \operatorname{Cov}[X + Y,\, Y + X] \\
>   &= \operatorname{Cov}[X,X] + \operatorname{Cov}[X,Y] + \operatorname{Cov}[Y,X] + \operatorname{Cov}[Y,Y] \\
>   &= \operatorname{Var}[X] + 2\operatorname{Cov}[X,Y] + \operatorname{Var}[Y]
> \end{aligned}
> $$

**Adaptation for §3.4.** Apply bilinearity to the affine map. A model's map position is
$p = w_0 + \sum_{j=1}^{10} w_j \bar{x}_j$ with weights $w_j$ fixed by the stored pipeline
($\mu, \sigma, C, R$ and the rescaling constants), so by properties 2 and 3 above, iterated:

$$
\operatorname{Var}(p)
= \sum_{j=1}^{10}\sum_{k=1}^{10} w_j w_k \operatorname{Cov}\!\left(\bar{x}_j, \bar{x}_k\right)
= \sum_{j=1}^{10} w_j^2 \operatorname{Var}(\bar{x}_j)
+ 2\!\!\sum_{j<k} w_j w_k \operatorname{Cov}\!\left(\bar{x}_j, \bar{x}_k\right)
$$

The affineness argument in the current §3.4 establishes only that $E[p]$ depends on the
$\bar{x}_j$ alone. The second sum is what an independent item bootstrap sets to zero — and the
last displayed line above, which is already in the wiki for the two-variable case, is the cleanest
way to show a reviewer that "the pairing never mattered" was a statement about the first moment.
Note also his own warning, quoted above, that zero covariance does not imply independence — the
converse of what the item bootstrap assumes, and worth one clause.

---

## 7. §3.4 — Sampling distributions, standard error, CLT

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/probability-and-statistics/sampling-and-estimation/statistics-and-sampling-distributions.mdx`

The bootstrap is a Monte Carlo approximation to a sampling distribution, and this note defines
that object cleanly. Worth a sentence in §3.4 so that "replicate cloud" has a referent.

> The probability distribution of a statistic (such as mean or std dev) is known as its
> **Sampling Distribution**. The standard deviation of a sampling distribution (e.g. sample means)
> is the **Standard Error (SE)**.
>
> The Central Limit Theorem states that the mean $\bar{W}$ of a random sample
> $D = (D_1, D_2, \ldots, D_n)$ is distributed as:
>
> $$
> \lim_{n \to \infty} P\left[a \leq \frac{\bar{W} - E[W]}{\sqrt{\operatorname{Var}[W]/n}} \leq b\right]
> = \frac{1}{\sqrt{2\pi}} \int_a^b e^{-z^2/2}\, dz
> $$
>
> which can equivalently be written as
> $\bar{W} \sim N\!\left(\mu = E[W],\; \sigma = \sqrt{\operatorname{Var}[W]/n}\right)$, with
>
> $$
> \text{Standard Error (SE)} = \frac{\sigma}{\sqrt{n}}
> $$
>
> As the sample size $n$ increases, the dispersion of the sampling distribution of the sample
> means becomes smaller.

**How it fits.** Two uses. (i) It licenses §3.4's normal-theory ellipse in one clause — the
replicate means are sample means of bounded discrete variates at $n \approx 50$ per item, so the
CLT applies and the review's verified Mahalanobis² check (5.55–6.33 vs $\chi^2_{0.95,2}=5.99$) is
the empirical confirmation. (ii) The final sentence is the argument the review flags in its
closing note on §2.4: SE $\to 0$ as $n \to \infty$ is precisely why sampling-only ellipses for
near-deterministic 2026 frontier models shrink toward zero and stop measuring anything a reader
cares about. His own note states the mechanism.

---

## 8. §3.4 — Confidence regions: the general definition, and the $\chi^2$ quantile convention

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/probability-and-statistics/sampling-and-estimation/interval-estimation.mdx`

The general definition is the useful part, because it is precisely the distinction the review asks
the paper to make in §2.4(c) — a region *for the parameter*, not for the response distribution.

> Let $D_1, \ldots, D_n$ be a random sample $D$ from a population with an unknown parameter
> $\theta$. Given a confidence level $(1-\alpha)$, and if $l(D), u(D)$ are computed from sample
> statistics with the property that
>
> $$
> P\big[\,l(D) < \theta < u(D)\,\big] = (1-\alpha)
> $$
>
> then we say that $[l(D), u(D)]$ is a $100(1-\alpha)\%$ confidence interval (CI) for $\theta$.
> Thus, the confidence interval contains the true value of the parameter $\theta$ with some known
> probability $(1-\alpha)$. […] this means that the true population parameter $\mu$ will be within
> this interval $95\%$ of the time. In other words, if we were to conduct 100 sampling trials for
> $D$, then the computed confidence interval would contain $\mu$ 95 times on average.

> It is often helpful to think of $z_{\alpha/2}$ as how many units of standard error
> $\left(\sigma/\sqrt{n}\right)$ we allow the confidence interval to be. Thus, the confidence
> interval is $\bar{X} \pm z_{\alpha/2} \times \text{Standard Error}$.

**Adaptation for §3.4.** The bivariate analogue, which the paper needs and the wiki does not have,
is one line away from the above:

$$
\mathcal{E}_{0.95} = \left\{\, \mathbf{p} \in \mathbb{R}^2 \;:\;
(\mathbf{p} - \bar{\mathbf{p}})^{\top} \hat{\Sigma}^{-1} (\mathbf{p} - \bar{\mathbf{p}})
\;\le\; \chi^2_{0.95,\,2} = 5.99 \,\right\}
$$

with $\hat{\Sigma}$ the covariance of the $B$ bootstrap replicate means. Phrase the caption as a
"confidence region for the model's mean position", per the review. The distribution-free variant
the review offers is the same set with $5.99$ replaced by the empirical 95th percentile of the
replicates' Mahalanobis².

**Notation trap — do not copy the wiki's $\chi^2$ subscripts unchanged.** The same file uses
$\chi^2$ for a genuinely different purpose (a CI for the variance of a normal population,
$(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$) and adopts the *upper-tail* convention
$P(Y \geq \chi^2_{\alpha/2,\,n-1}) = \alpha/2$:

> $$
> P\left[\frac{(n-1)S^2}{\chi^2_{\alpha/2,\,n-1}} \leq \sigma^2 \leq
> \frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\,n-1}}\right] = (1-\alpha)
> $$

Under that convention $\chi^2_{0.95,2}$ would denote the *5th* percentile, not the 95th, which is
the opposite of `scipy.stats.chi2.ppf(0.95, 2)` in the repo. Whichever convention the paper picks,
state it once — this is a cheap way to lose a reader who checks.

---

## 9. §3.4 — Cross-validation for the SVM

**Source:** `/Users/shav/Code/ShavWiki/pages/artificial-intelligence/real-world-data/k-fold-x-validation.mdx`

Directly relevant to the review's §2.5, and unusually apt: the worked example is *literally* a
5-fold CV kernel comparison on `sklearn.svm.SVC`, and its punchline is that a single split hid
overfitting that CV exposed.

> 1. **Divide** the data into $K$ buckets.
> 2. **Reserve** one of those buckets for testing purposes, for evaluating the results of the model.
> 3. **Train** your model on $K-1$ subsets.
> 4. **Test** the model on the remaining subset.
> 5. **Average** the test metrics from each fold for a final performance error metric.

> **Comparing kernels.** Is a more complex polynomial kernel better than a simple linear one? Will
> that be over-fitting or will it better fit the data that we have? […] The more complex
> polynomial kernel produced lower accuracy than a simple linear kernel. The polynomial kernel is
> over-fitting. But we couldn't have told that with a single train/test split — that's the same
> score we got with a single train/test split on the linear kernel.
>
> If we had relied solely on a single train/test split, we could've missed signs of overfitting.

**How it fits.** §3.4 currently says only "grid-searched, 5-fold CV". The review's finding is that
the reported 5-fold CV accuracy is 0.552 against a train accuracy of 0.679 on 109 countries across
8 classes, with a grid ($C \in \{500,\dots,2000\}$) that excludes the regularised regime entirely.
Shav's own note is the argument for *why the CV number must be reported at all*: it is the only
thing separating a fitted decision boundary from a memorised one. Cite his framing, report the
0.552, and let the SVM-free headline statistic (review §2.7) carry the Confucian claim.

**Caveat.** The wiki's CV note is about *model selection*, not about the sampling variability of a
prediction. It has nothing to say about the review's §5c point — that the reported "stability" is
the variability of the model's position under a *fixed* classifier and excludes the classifier's
own ~45% error rate. That decomposition has to be written fresh.

---

## 10. Nothing usable — checked, and worth recording so the search is not repeated

- **SVM margin / kernel mathematics.** `predictive-models/svm.mdx` is 159 lines and contains **no
  equations at all** — no margin, no dual, no RBF kernel, no $C$. It explicitly punts: "While the
  underlying math is complex, in summary the kernel trick helps the algorithm efficiently find
  hyperplanes in high dimensions." $\gamma$ is mentioned only as "a hyperparameter that is hard to
  visualise". Everything §3.4 needs about $K(\mathbf{x},\mathbf{x}') = \exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)$
  and the soft-margin objective must be written fresh.
- **Bootstrap theory.** The only occurrence in the entire wiki is one line in
  `ensemble-methods-advanced-models/ensemble-learning.mdx`: "Bagging (Bootstrap Aggregating):
  Generate $N$ new training sets by random sampling with replacement." No plug-in principle, no
  Efron, no percentile/BCa intervals, no cluster bootstrap. Not worth citing.
- **Intraclass correlation / variance components.** `predictive-models/multi-level.mdx` is 16 lines
  of pure prose about hierarchical *effects* (GCSE results, family/neighbourhood/institutional
  levels) with no model, no $\sigma^2_{\text{between}}/(\sigma^2_{\text{between}} +
  \sigma^2_{\text{within}})$, no random effects. Nothing to lift.
- **LDA.** `feature-extraction/lda.mdx` is code and output tables only; the PCA-vs-LDA comparison is
  qualitative. Not relevant to this paper regardless.
- **Model evaluation.** `model-evaluation.mdx` is ROUGE/BLEU/GLUE/MMLU — NLP benchmark metrics,
  unrelated to region-classification evaluation.
- **`cloud-mlops/`.** Swept. Infrastructure and AWS service notes; the one "feature engineering"
  page is about SageMaker Ground Truth labelling. Nothing statistical.

---

## Gaps — must be written fresh

Ordered by how exposed the paper is if they are missing.

1. **PPCA and the EM algorithm over missing entries.** No latent-variable model, no
   $\mathbf{x} = W\mathbf{z} + \mu + \epsilon$, no marginal likelihood, no E-step/M-step, no
   Jensen bound, nowhere in the wiki. This is §3.2's core citation (Tipping & Bishop 1999) and
   there is nothing to reuse. Also needed: the conditional-mean-imputation variance-shrinkage
   point behind the review's `_calc_var` note (total variance 9.64 vs nominal 10).
2. **Varimax.** No factor analysis anywhere in the wiki — no rotation criterion, no Kaiser (1958),
   no normalisation, no simple structure. The review needs the paper to state the criterion
   explicitly, name it as varimax-on-scores (Rohe & Zeng 2023) rather than varimax-on-loadings, and
   disclose Kaiser row normalisation. All fresh.
3. **Cluster bootstrap and ICC.** Both are new §3.4 primary machinery per the review (Field & Welsh
   2007; Davison & Hinkley §3.8; Cameron, Gelbach & Miller 2008 on $K=10$ under-coverage). Nothing
   in the wiki. The variance identity in §6 above is the only piece that transfers.
4. **Manski-style partial identification bounds.** No trace anywhere in the wiki — no bounds, no
   identified sets, no monotonicity assumptions. Entirely fresh.
5. **RBF-SVM formalism.** Kernel, soft margin, dual, and the meaning of $C$ and $\gamma$ —
   fresh (see §10).
6. **Confidence-ellipse geometry.** The Mahalanobis quadratic form, its $\chi^2_2$ distribution,
   and semi-axes $\sqrt{\chi^2_{0.95,2}\,\lambda_i}$ along the eigenvectors of $\hat{\Sigma}$. The
   symmetric-eigendecomposition half is in the wiki (§2 above); the probabilistic half is not.
7. **Orthogonal Procrustes and Tucker congruence.** Needed for the review's "Story B" external
   validation of the rotation. Nothing in the wiki (the SVD is not covered — only
   eigendecomposition and LU).
8. **Missing-data taxonomy.** MCAR/MAR/MNAR, Little & Rubin, missing-by-design. The wiki's
   treatment (§5 above) is applied and informal; the formal MAR statement that licenses EM must be
   written fresh.
