# Umbra Empirical Benchmark & Reproducibility Suite

This directory contains the complete, deterministic experimental suite used to evaluate Umbra and generate the living leaderboard in [`results.md`](results.md).

---

## 1. Single-Command Reproduction

Umbra supports single-command reproduction with fixed random seeds:

### Fast Verification Mode (~30 seconds)
```bash
python -m benchmarks.reproduce_all --quick
```
Runs an abbreviated battery ($R=3$, smaller $N$) verifying that all pipeline components, baselines, misspecification regimes, and figure generation execute without error.

### Full Publication-Grade Battery (~3 minutes)
```bash
python -m benchmarks.reproduce_all --full
```
Runs the full Monte Carlo suite ($R=20, N=2,500$ across 6 regimes), the independent Auto Router matrix ($R=5$ across $N \in [500, 2500]$ and missingness $\in [20\%, 40\%]$), the 6 misspecification regimes ($R=20$), runtime scaling, and regenerates all 7 publication figures in [`figures/`](figures/).

---

## 2. Benchmark Components

| Script | Purpose | Output Artifact |
| :--- | :--- | :--- |
| `benchmarks/run_all.py` | Master Monte Carlo simulation suite & policy comparison | [`results.md`](results.md) |
| `benchmarks/router_benchmark.py` | Auto router classification accuracy, Wilson 95% CIs, and threshold sensitivity | Terminal output & tables |
| `benchmarks/misspecification_benchmark.py` | 6 boundary stress tests (weak instruments, exclusion violations, Student-$t$, nonlinear) | [`misspecification_results.md`](misspecification_results.md) |
| `benchmarks/performance_scaling.py` | Runtime scaling as a function of sample size $N$ and feature count $p$ | Markdown tables in `results.md` |
| `scripts/generate_figures.py` | Publication figure generator (300 DPI PNG + vector PDF) | [`figures/`](figures/) |
| `scripts/reproduce_case_studies.py` | 4 real-world domain case studies (CPS, NHANES, Housing, Clinical Trial) | Markdown reports |

---

## 3. Data Generating Processes (DGPs)

Mathematically defined in [`benchmarks/dgps.py`](dgps.py):
1. **MCAR**: Bernoulli dropout independent of all covariates.
2. **MAR**: Logistic probability dependent on observed covariates (`age`, `education`).
3. **MNAR_SELF_MASKING**: Logistic selection on unobserved target variable $Y$.
4. **MNAR_SELECTION**: Bivariate latent threshold Gaussian selection model with candidate auxiliary instrument $Z$.
5. **MNAR_PATTERN_MIXTURE**: Subpopulation mean shift $\delta \cdot \sigma$ for non-responders.
6. **MNAR_TAILS**: Symmetric quadratic U-shaped dropout in both tails.
7. **MNAR_WEAK_SIGNAL**: Subtle departure from MAR ($\gamma = 0.40$).
8. **MNAR_STRONG_SIGNAL**: Severe self-censoring ($\gamma = 2.80$).

---

## 4. Evaluated Policies & Baselines

1. **Policy A (Always MICE)**: Standard chained equations assuming MAR (PMM or Bayesian Ridge).
2. **Policy B (Always Heckman)**: Two-step Heckman selection assuming bivariate normality.
3. **Policy C (Always Sensitivity)**: Pattern-mixture sensitivity analysis across $\delta \in [-3\sigma, +3\sigma]$.
4. **Policy D (Umbra Auto)**: Evidence-conditioned decision policy switching between MICE and Heckman based on observable signals and instrument relevance ($F > 10$).
5. **Policy E (Oracle Route)**: Upper benchmark with ground truth knowledge of the true DGP.
