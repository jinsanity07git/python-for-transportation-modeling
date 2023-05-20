# %%
from transportation_tutorials.choice_theory_figures import *

# %% [markdown]
# # Discrete Choice Theory

# %% [markdown]
# 离散选择建模是许多交通规划模型的核心。
# 这是因为旅行的许多方面都代表着离散的选择：
# 我是走路还是开车？要不要买一辆车？走高速公路还是小路？
# 在上班路上去商店吗？去哪个商店？
# 
# 所有这些选择都具有一些共同特点：

# 它们是从一系列分类选项中进行选择的。
# 这些分类选项通常缺乏自然的排序（步行并不明确地比开车更多或更少，只是不同）。
# 选择一个选项意味着不选择其他选项。
# 除了这些选择的实际方面之外，我们还将对这些选择进行一些合理的假设，即：

# 决策者（通常是旅行的人们）从这些选项中做出理性选择，选择他们认为最好的选项。
# 对于哪个选项最好的判断是基于备选方案的属性，我们作为建模者可以观察到决策者正在考虑的至少一些相关属性。
# %% [markdown]
# ## 数学推导

# %% [markdown]
# 我们可以将这些合理的假设，再加上一些可能不太合理的假设，转化为数学模型来表示选择过程。
# 假设在一个选择问题中，每个备选方案都有与之相关联的某个值，并且决策者通过选择具有最高值的备选方案来做出选择。
# 我们将这个值称为**"效用"，从而使这个决策过程成为"效用最大化"**。
# 不幸的是，我们实际上无法观察到效用，我们只能观察到决策者实际或假设的选择。

# %% [markdown]
# 从数学上讲，决策者$t$在选择$C_t$中从备选方案$i$中选择，当且仅当满足以下条件：
# 
# $$
# U_i \ge U_j \quad \forall j \in C_t.
# $$
# 
# 这基于对效用的一些表示：
# 
# $$U_i = V_i + \varepsilon_i$$
# 
# 其中$V_i$是测量组成部分，也称为系统效用，$\varepsilon_i$是未观察到的组成部分，也称为随机效用。请注意，从决策者的角度来看，$\varepsilon_i$并不是随机的，它只在建模者的视角下是随机的。我们可以确定决策者$t$选择备选方案$i$的概率为：
# 
# $$
# \Pr_t(i | C_t) = \Pr(U_i \ge U_j, \forall j \in C_t) \\
# = \Pr(V_i + \varepsilon_i \ge V_j + \varepsilon_j, \forall j \in C_t) \\
# = \Pr(\varepsilon_j - \varepsilon_i \le V_i - V_j, \forall j \in C_t)
# $$
# 
# 在最后一种形式中，我们将所有随机效用分组在一起，这可能会有所帮助。如果我们能够将这些术语联合表示，我们可以将其写成一个累积分布函数（CDF）。在对多元密度函数$f(\varepsilon)$不做任何特定假设的情况下，我们可以将概率写成一个积分形式，如下所示

# %% [markdown]
# $$
# \Pr_t(i | C_t) = 
# \int_{\varepsilon_i = -\infty}^{\infty} 
# \int_{\varepsilon_j = -\infty}^{V_i - V_j + \varepsilon_i} 
# f(\varepsilon) d\varepsilon_J d\varepsilon_{i-1} d\varepsilon_{i} 
# $$

# %% [markdown]

# 特定的误差分布假设会导致特定的模型结构。

# * 如果$\varepsilon$服从多元正态分布，我们得到一个多项式Probit模型。
# * 如果$\varepsilon$服从独立同分布的Gumbel分布，我们得到一个多项式Logit模型。独立同分布表示每个维度具有相同的均值和方差，并且各个维度之间没有相关性。
#
# 从理论上讲，多项式Probit模型是合理的，但在实践中却不方便操作，因为上述方程中的积分很复杂，并且Probit模型没有提供闭式解析解来消除它。另一方面，多项式Logit模型具有许多很好的数学简化，使得它非常容易处理。这就是为什么在交通规划中你到处看到Logit模型，而不是Probit模型的原因。

# %% [markdown]
# ## Gumbel分布
# 正态分布是（除其他外）一组样本的均值的分布。你可能以前听说过它。它是许多事物的默认分布，有很好的原因。

# Gumbel分布则不太为人所知。它是从正态分布中的一组样本中的最大值（或最小值）的（极限）分布。我们可以通过从随机正态分布中抽取一组样本，并在每个样本中取最大值来展示它。

# %%
# 5 thousand repetitions of samples taken from a normal distribution
draws = [
    np.random.normal(size=[1,5000]),
    np.random.normal(size=[10,5000]),
    np.random.normal(size=[100,5000]),
    np.random.normal(size=[1000,5000]),
    np.random.normal(size=[10000,5000]),
    np.random.normal(size=[100000,5000]),
]  

# %%
gumbel_draws = [d.max(axis=0) for d in draws]

# %%
figure_gumbel_draws(draws)

# %% [markdown]
# As the size of the pool from which the maximum is taken does up, so does the mean of the maximum.

# %%
np.mean(gumbel_draws, axis=1)

# %% [markdown]
# If you want to work with gumbel distribution directly, you don't need to back
# into it by actually taking the maximum of a bunch of things.
# The gumbel distribution is available in the 
# [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) package,
# under the name [`gumbel_r`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html) 
# for the right-skewed version associated with the maximum, or the mirrored 
# [`gumbel_l`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_l.html) for the left-skewed version associated
# with the minimum.

# %%
from scipy.stats import gumbel_r, norm

# %%
figure_gumbel_draws(draws, gumbel_r)

# %% [markdown]
# Similar to the normal distribution, the Gumbel distribution is defined by two parameters, one for position and one for width.  These are similar to the mean and standard deviation that define the normal distribution, although the defining parameters are not themselves the mean and standard deviation of the Gumbel.
# 
# The probability density function (pdf) for `gumbel_r` is:
# 
# $$f(x; \mu, \sigma) = \frac{1}{\sigma} \exp\left(-\frac{x+\mu}{\sigma}\right) \exp\left(-\exp\left(-\frac{x+\mu}{\sigma}\right)\right)$$
# 
# The cumulative density function (cdf) for `gumbel_r` is:
# 
# $$F(x; \mu, \sigma) = \exp\left(-\exp\left(-\frac{x-\mu}{\sigma}\right)\right)$$
# 
# The Gumbel distribution is sometimes referred to as a *type I Fisher-Tippett
# distribution*. It is a bell-shaped distribution but obviously not "the" bell curve, which is generally the normal distribution. It is assymetric, but as we'll see below that is not important in the context of discrete choice analysis.
# 
# We can plot the PDF and CDF functions to get a better idea of the differences.

# %%
figure_distribution_vs_normal(gumbel_r)

# %% [markdown]
# The mode (peak) of the standard Gumbel distribution is at zero,
# but the mean is $0.577$, because the distribution is skewed.  The variance is $\frac{\pi^2}{6} = 1.645$.  

# %% [markdown]
# ### Scale and Translation
# 
# Some properties of the Gumbel distribution include the ability to scale and translate (shift) it. If $\varepsilon \sim G(\mu,\sigma)$, then
# 
# - You can define a scaled version by multiplying by some constant $\alpha$ against both characteristic parameters:
#   - $\alpha\varepsilon \sim Gumbel(\alpha\mu,\alpha\sigma)$
#   - mode (peak) = $\alpha\mu$
#   - mean = $\alpha\mu + \alpha\sigma 0.577$
#   - variance = $\frac{\pi^2 \alpha^2 \sigma^2}{6}$

# %%
figure_gumbel_scale(
    gumbel_r(0,0.5),
    gumbel_r(0,1),
    gumbel_r(0,1.5),
    gumbel_r(0,2),
)

# %% [markdown]
# - You can define a translated (shifted) version by adding only to the location parameter: 
#   - ($\varepsilon + V) \sim Gumbel(\mu + V,\sigma)$ where $V$ is some non-random value
#   - mode (peak) = $\mu + V$
#   - mean = $\mu + V + \sigma 0.577$
#   - variance = $\frac{\pi^2 \sigma^2}{6}$
#   - outcome: the distribution can be shifted left or right with no change in shape or variance
#   

# %%
figure_gumbel_translate(
    gumbel_r(0,1),
    gumbel_r(1,1),
    gumbel_r(2,1),
    gumbel_r(3,1),
)

# %% [markdown]
# We can simplify the derivation of the Multinomial Logit model by
# assuming $\mu = 0$ and $\sigma = 1$ (Standard Gumbel).
# We will examine effect of different values later.
# 
# Up above, we created draws from a Gumbel distribution by generating
# a lot of normally distributed random variables, and taking the maximum.
# Scipy also allows directly generating random draws from the Gumbel
# distribution (and 
# [many other distributions](https://docs.scipy.org/doc/scipy/reference/stats.html)
# also) using the `rvs` method.

# %%
gumbel_r(0,1).rvs()

# %% [markdown]
# If we want more than one random draw from the same distribution, or
# if we want to give a specific seed for reproducibility, we can give
# `size` or `random_state` arguments, respectively.

# %%
gumbel_r(0,1).rvs(size=4, random_state=123)

# %% [markdown]
# ## The Logistic Distribution

# %% [markdown]
# A handy property of the Gumbel distribution is that the difference between two independent 
# Gumbel distributions (with same variance) is a **logistic distribution**.  
# 
# The probability density function is:
# 
# $$
# \begin{align}
# f(x; \mu, \sigma)  & = \frac{e^{-(x-\mu)/\sigma}} {s\left(1+e^{-(x-\mu)/\sigma}\right)^2} \\[4pt]
# \end{align}
# $$
# 
# The cumulative density function is:
# 
# $$
# F(x; \mu, \sigma) = \frac{1}{1+\exp({-(x-\mu)/\sigma})}
# $$
# 
# It looks like this:

# %%
from scipy.stats import logistic
figure_distribution_vs_normal(logistic)

# %% [markdown]
# Unlike the Gumbel, the Logistic distribution is symmetric.  Since it is also
# unimodal (one hump) the mean and mode (peak) are at the same value.  When it
# is created by taking the difference between two IID Gumbels, this value is 
# equal to the difference between the location parameters (peaks) of those
# two Gumbels.

# %% [markdown]
# $\varepsilon_1 \sim Gumbel(V_1, 1)$
# 
# $\varepsilon_2 \sim Gumbel(V_2, 1)$
# 
# $\varepsilon_1 - \varepsilon_2 = \varepsilon_3 \sim Logistic(V_1-V_2, 1)$
# 

# %% [markdown]
# ## Maximium over Multiple Gumbels

# %% [markdown]
# Suppose we have two random variables, both with a Gumbel distribution, both with the same variance.
# 
# $$\varepsilon_1 \sim G(V_1, 1)$$
# 
# $$\varepsilon_2 \sim G(V_2, 1)$$
# 
# Then, the distribution of the maximum, i.e. $\max(\varepsilon_1, \varepsilon_2)$ is *also* a Gumbel distribution, and *also* with the same variance.  
# 
# $$\max(\varepsilon_1, \varepsilon_2) \sim G(V_\star, 1)$$
# 
# with $V_\star = \log\left(\exp(V_1)+\exp(V_2)\right)$.
# 
# By extension, the same holds for the maximum of any set of $N$ Gumbel distributions:
# 
# $$V_\star = \log\left(\sum_{j=1}^{N}\exp(V_j)\right)$$.
# 
# For obvious reasons, this term is called a "logsum". You will find that logsums are used in many places when working with discrete choice models.

# %% [markdown]
# ## Deriving the Multinomial Logit Choice Model

# %% [markdown]
# Recall our utility formulation.  The probability of a decision maker choosing a particular alternative $i$ is equal to the probability that the utility $U_i$ is greater than or equal to the utility $U_j$ for all other possible choices $j$.  This is equivalent to writing:
# 
# $$
# \Pr\left(U_i \ge \max(U_j, \forall j \ne i)\right)
# $$
# 
# or
# 
# $$
# \Pr\left(V_i + \varepsilon_i \ge \max(V_j + \varepsilon_j, \forall j \ne i)\right)
# $$
# 
# or
# 
# $$
# \Pr\left(V_i \ge \max(V_j + \varepsilon_j, \forall j \ne i) - \varepsilon_i\right)
# $$
# 

# %% [markdown]
# Some important features of this way of writing the probabilities:
# 
# - On the left side of the inequality we have just $V_i$ which is not a random variable.
# - On the right side we have $\max(V_j + \varepsilon_j, \forall j \ne i)$ which is 
#   a maximum of (shifted) Gumbel distributions, which then is itself a Gumbel distribution,
#   let's call it $\varepsilon_\star$ which has a (shifted) location of $V_\star$.
# - Then we take the difference of that Gumbel distribution and another (the one 
#   for $\varepsilon_i$), which makes one logistic distribution.
# - That one logistic distribution is the only term on the right side of the inequality.
# - We therefore reduced our multi-dimensional integral from before to a one-dimensional
#   integral on a single logistic distribution ... which conveniently has a closed form solution:
#   the CDF:
#   
# $$
# \Pr\left(V_i \ge \max(V_j + \varepsilon_j, \forall j \ne i) - \varepsilon_i\right) = 
# \frac{1}{1+\exp(V_\star - V_i)}
# $$

# %% [markdown]
# We can reformat this a bit to make it more obviously symmetric and generalizable: 

# %% [markdown]
# $$
# \frac{1}{1+\exp(V_k - V_i)}
# = \frac{1}{1+\exp(V_k)/\exp(V_i)} = \frac{\exp(V_i)}{\exp(V_i)+\exp(V_k)}
# $$

# %% [markdown]
# And recall that $V^\star = \log\left(\sum_{j=1}^{N}\exp(V_j)\right)$, which leads to
# 
# $$
# \Pr(i) = \frac{\exp(V_i)}{\sum_{j=1}^{N}\exp(V_j)}
# $$

# %% [markdown]
# ## Arbitrary Scale 

# %% [markdown]
# Consider what happens if set the scale of the Gumbel distributions in our utility
# functions to some value other than 1:
# 
# The utility function for alternative $j$ is now
# 
# $$
# U_j = V_j + \varepsilon_j,\quad \varepsilon_j \sim Gumbel(0,\mu)
# $$
# 
# 

# %% [markdown]
# Let's take this but define a different scaled measure of utility $U^\prime$.
# 
# $$
# \begin{align}
# U^\prime_j &= \mu U_j \\
# &= \mu (V_j + \varepsilon_j),\quad \varepsilon_j \sim Gumbel(0,1) \\
# &=  \mu V_j + \mu\varepsilon_j,\quad \mu\varepsilon_j \sim Gumbel(0,\mu)
# \end{align}
# $$

# %% [markdown]
# You can see that the two different models are practically the same, except for the scale of the variance.
# 
# However, because we will be choosing the functional form of $V$ to best fit the data, we can scale the
# values we get for $V$ and end up with an identical model with respect to the output probabilities.
# 
# If we were able to observe the *utility* values directly, we could identify a unique scale of the 
# $\varepsilon$ terms that would fit best.  But because we can only observe the choices, we cannot
# actually identify which scale is better, and we are free to choose any scale factor that is mathematically
# convenient (i.e., 1).

# %% [markdown]
# ## Arbitrary Location 
# 

# %% [markdown]
# Consider what happens if set the location (mode) of the Gumbel distributions in our utility
# functions to some value other than 0:
# 
# The utility function for alternative $j$ is now
# 
# $$
# U_j = V_j + \varepsilon_j,\quad \varepsilon_j \sim Gumbel(\delta, 1)
# $$

# %% [markdown]
# Let's take this but define a different shifted measure of utility $U^\prime$.
# 
# $$
# \begin{align}
# U^\prime_j &= V_j + \varepsilon_j,\quad \varepsilon_j \sim Gumbel(\delta, 1) \\
# &= V_j + \delta + \varepsilon_j,\quad \varepsilon_j \sim Gumbel(0,1)
# \end{align}
# $$

# %% [markdown]
# As for the scale, we can push an arbitrary location adjustment from $\varepsilon$ into $V$, and end up with the same model with respect to probabilities.

# %% [markdown]
# ## Too Much Arbitrary-ness in Location

# %% [markdown]
# Consider what happens if we introduce an arbitrary constant shift in utility for *every* alterantive.

# %% [markdown]
# What happens to the differences in utilities between alternatives?  
# 
# $$
# V_i^\prime - V_j^\prime = (V_i + \delta) - (V_j + \delta) = V_i  - V_j 
# $$

# %% [markdown]
# What happens to the probabilities of alternatives?  
# 
# 
# $$
# \Pr(i) = \frac{\exp(V_i + \delta)}{\sum_{j=1}^{N}\exp(V_j + \delta)}
# = \frac{\exp(V_i)\exp(\delta)}{\sum_{j=1}^{N}\exp(V_j)\exp(\delta)}
# = \frac{\exp(\delta)}{\exp(\delta)}\frac{\exp(V_i)}{\sum_{j=1}^{N}\exp(V_j)}
# = \frac{\exp(V_i)}{\sum_{j=1}^{N}\exp(V_j)}
# $$

# %% [markdown]
# - We can shift constants for all alternatives the same and get the same probability model.
# - We pick any alternative arbitrarily and set the constant to zero and get the same probability model.
# - **Only the differences in utility matter, not the absolute values.**

# %% [markdown]
# ## Typical Form of Systematic Utility

# %% [markdown]
# In most transportation planning applications, the systematic utility is given as a **linear in parameters** function.
# 
# This means that the functional form of $V_{ti}$ for decision maker $t$ and alternative $i$ looks like
# 
# $$
# V_{ti} = X_{ti} \beta_{i} = \sum_{k=1}^{K} X_{tik} \beta_{ik}
# $$

# %% [markdown]
# You may recall from the `statsmodel` package that in order to include a y-intercept or constant term in the model, it was necessary to explicitly add a column of all 1's to the dataframe.  The same is generally true when expressing the utility function for a discrete choice model like this.

# %% [markdown]
# One important feature to keep track of here that is different from an ordinary least squares linear regression model: **only the differences in utility matter**.
# 
# This means that *something* needs to induce some differences in the systematic utility between alternatives.  That *something* can be:
# 
# - differences in the observed $X$ data values (e.g. the travel time is different for different modes), or
# - differences in the $\beta$ parameter values across modes, or
# - both.
# 
# It is important to pay attention to instances where observed data values do *not* vary across modes, and ensure
# that the parameters *do* vary in those instances.  These are sometimes called alternative-specific variables.
# 
# (Yes, this is ironic, because the variables themselves are in fact not specific to the alternative.)

# %% [markdown]
# ## Independence of Irrelevant Alternatives (IIA)

# %% [markdown]
# One important property of the MNL model is "Independence of Irrelevant Alternatives".
# 
# This refers to the fact that the ratio of choice probabilities between pairs of alternatives is independent of the availability or attributes of other alternatives.
# 
# $$
# \frac{\Pr(i)}{\Pr(k)} = \frac{\exp(V_i)}{\sum_j \exp(V_j)}\frac{\sum_j \exp(V_j)}{\exp(V_k)}
# =\frac{\exp(V_i)}{\exp(V_k)} = \exp(V_i - V_k)
# $$
# 

# %% [markdown]
# How reasonable is this outcome?  In a lot of cases, not very reasonable at all.  In a later section, we'll look at the "nested multinomial logit" or "nested logit" model, which solves some of these issues.


