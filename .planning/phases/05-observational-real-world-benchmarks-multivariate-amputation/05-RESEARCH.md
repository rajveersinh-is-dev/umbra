# Phase 5: Observational Real-World Benchmarks & Multivariate Amputation - Research

**Researched:** 2026-09-09
**Domain:** Statistical Amputation, Schouten et al. (2018) Methodology, Observational Benchmark Datasets

## 1. Mathematical Formalism of Multivariate Amputation (Schouten et al. 2018)

### Step 1: Candidate Patterns Matrix
Let $\mathbf{P} \in \{0, 1\}^{K \times P}$ be the pattern matrix for $P$ variables across $K$ missingness patterns, where:
- $P_{kj} = 0$: variable $j$ is incomplete (subject to amputation) in pattern $k$.
- $P_{kj} = 1$: variable $j$ remains observed in pattern $k$.

Each observation $i \in \{1, \dots, N\}$ is assigned to pattern $k \in \{1, \dots, K\}$ according to pattern frequencies $\mathbf{f} = (f_1, \dots, f_K)$ with $\sum_{k=1}^K f_k = 1$.

### Step 2: Predictor Weights and Weighted Sum Scores
For each pattern $k$, define weight vector $\mathbf{w}_k \in \mathbb{R}^P$.
The weighted sum score for observation $i$ is:
$$S_{ik} = \sum_{j=1}^P w_{kj} \tilde{X}_{ij}$$
where $\tilde{X}_{ij} = \frac{X_{ij} - \bar{X}_j}{s_j}$ are standardized features.

**Mechanism Constraints on Weights**:
- **MCAR**: Scores $S_{ik}$ are ignored; missingness is uniform random: $P(R_{ij} = 0) = \pi$.
- **MAR**: Missingness in incomplete variables depends *only* on observed variables:
  $$w_{kj} = 0 \quad \forall j \text{ such that } P_{kj} = 0$$
- **MNAR**: Missingness in incomplete variables depends on the incomplete variables themselves:
  $$\exists j \text{ such that } P_{kj} = 0 \text{ and } w_{kj} \ne 0$$

### Step 3: Probability Mapping Functions
The standardized sum score $S_{ik}$ is mapped to missingness probability $P_{ik} \in (0, 1)$ via logit curves:
1. **`RIGHT`** (Tail Dropout High):
   $$P_{ik} = \frac{1}{1 + \exp(-(S_{ik} - b))}$$
2. **`LEFT`** (Tail Dropout Low):
   $$P_{ik} = \frac{1}{1 + \exp(S_{ik} - b)}$$
3. **`MID`** (Central Concentration):
   $$P_{ik} = \exp\left(-\frac{(S_{ik} - b)^2}{2}\right)$$
4. **`TAIL`** (Bimodal U-Shaped Extreme Dropout):
   $$P_{ik} = \frac{1}{1 + \exp(-(|S_{ik}| - b))}$$

The offset parameter $b$ is calibrated via 1D root-finding (Brent's method / binary search) so that the mean probability across group $k$ equals the target amputation proportion $\pi_k$:
$$\frac{1}{N_k} \sum_{i \in \text{group } k} P_{ik} = \pi_k$$

### Step 4: Missingness Realization
For observation $i$ in group $k$, draw $U_i \sim \text{Uniform}(0, 1)$.
If $U_i < P_{ik}$, set $X_{ij} = \text{NaN}$ for all features $j$ with $P_{kj} = 0$.

---

## 2. Observational Benchmark Datasets

### CPS Labor Wage Data (Mroz 1987 / Wooldridge)
- **Problem**: 753 married women in 1975 Current Population Survey.
- **Variables**: `hours`, `wage` (target), `educ`, `exper`, `age`, `kidslt6`, `nwifeinc`, `inlf`.
- **Selection**: 325 women did not work in the market, so `wage` is missing not at random.
- **Instruments**: `kidslt6`, `nwifeinc` serve as classical exclusion restrictions.

### NHANES Health & Clinical Biomarkers
- **Problem**: National Health and Nutrition Examination Survey.
- **Variables**: `age`, `bmi`, `systolic_bp`, `cholesterol`, `fasting_glucose` (target), `phlebotomy_difficulty` (auxiliary instrument).
- **Missingness**: Laboratory non-compliance and fasting non-response.

---

## 3. Comparative Benchmark Design

Evaluate 6 baseline & Umbra strategies:
1. **Complete Case Analysis (CCA)**
2. **Mean / Median Imputation**
3. **Standard MICE (MAR Chained Equations)**
4. **Umbra Auto-Router (MICE / Heckman / Pattern Mixture)**
5. **Umbra Heckman Selection Model**
6. **Umbra Pattern Mixture Sensitivity**

Metrics:
- Parameter recovery error: $\Delta \beta = \|\hat{\beta} - \beta^*\|_2$
- Imputation Cell RMSE and MAE against complete data
- Coverage probability of 95% confidence intervals
- Runtime efficiency
