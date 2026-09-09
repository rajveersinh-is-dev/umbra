"""
Multivariate Amputation Engine (Schouten, Lugtig, & Vink, 2018).

Generates missing values in complete datasets under specified mechanisms:
- MCAR: Missing Completely At Random (uniform random dropout)
- MAR: Missing At Random (missingness probability depends only on observed variables)
- MNAR: Missing Not At Random (missingness probability depends on incomplete variables themselves)

Implements continuous logistic probability distribution functions:
- RIGHT: High scores have higher probability of missingness (positive shift)
- LEFT: Low scores have higher probability of missingness (negative shift)
- MID: Central/average scores have higher probability of missingness
- TAIL: Extreme scores in both tails have higher probability of missingness
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit


@dataclass
class AmputationResult:
    """Structured container for the output of multivariate amputation."""

    data_amputed: pd.DataFrame
    data_complete: pd.DataFrame
    mask: pd.DataFrame  # True = missing (NaN), False = observed
    patterns: np.ndarray  # Shape (K, P): 0 = amputed/incomplete, 1 = observed
    weights: np.ndarray  # Shape (K, P): predictor weights per pattern
    mechanisms: List[str]  # Mechanism per pattern ('MCAR', 'MAR', 'MNAR')
    odds_types: List[str]  # Odds function per pattern ('RIGHT', 'LEFT', 'MID', 'TAIL')
    probabilities: np.ndarray  # Shape (N,): individual amputation probability per sample
    pattern_assignments: np.ndarray  # Shape (N,): pattern index assigned to each sample
    empirical_prop: float  # Fraction of samples with at least one missing value
    n_incomplete: int  # Total count of incomplete samples
    cell_missing_prop: float  # Fraction of all cells that are missing

    def summary(self) -> Dict[str, Any]:
        """Return high-level amputation summary statistics."""
        return {
            "n_samples": len(self.data_amputed),
            "n_features": self.data_amputed.shape[1],
            "n_patterns": len(self.patterns),
            "mechanisms": self.mechanisms,
            "odds_types": self.odds_types,
            "target_prop": float(np.mean(self.probabilities)),
            "empirical_row_missing_rate": self.empirical_prop,
            "n_incomplete_rows": self.n_incomplete,
            "cell_missing_rate": self.cell_missing_prop,
            "features_incomplete": [
                col for col in self.data_amputed.columns if self.data_amputed[col].isna().any()
            ],
        }


def _validate_and_prepare_inputs(
    data: Union[pd.DataFrame, np.ndarray],
    prop: float,
    patterns: Optional[Union[np.ndarray, Sequence[Sequence[int]]]],
    freq: Optional[Union[np.ndarray, Sequence[float]]],
    mechanisms: Union[str, Sequence[str]],
    weights: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
    odds_type: Union[str, Sequence[str]],
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    List[str],
    np.ndarray,
    List[str],
]:
    """Validate and normalize all amputation inputs."""
    # 1. Format and validate data
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(f"Input data array must be 2-dimensional; got {data.ndim}D.")
        cols = [f"col_{j}" for j in range(data.shape[1])]
        df = pd.DataFrame(data.copy(), columns=cols)
    else:
        raise TypeError(f"data must be a pandas DataFrame or 2D numpy array, got {type(data)}.")

    n_samples, n_features = df.shape
    if n_features < 2:
        raise ValueError(f"Amputation requires at least 2 features, got {n_features}.")
    if n_samples < 2:
        raise ValueError(f"Amputation requires at least 2 samples, got {n_samples}.")
    if not (0.0 < prop < 1.0):
        raise ValueError(f"prop must be strictly between 0 and 1, got {prop}.")

    # 2. Patterns matrix (K, P)
    # 0 = incomplete, 1 = observed
    if patterns is None:
        # Default: P patterns, each missing one variable in turn
        pat_arr = np.ones((n_features, n_features), dtype=int)
        np.fill_diagonal(pat_arr, 0)
    else:
        pat_arr = np.asarray(patterns, dtype=int)
        if pat_arr.ndim != 2:
            raise ValueError(f"patterns must be a 2D array, got shape {pat_arr.shape}.")
        if pat_arr.shape[1] != n_features:
            raise ValueError(
                f"patterns column count ({pat_arr.shape[1]}) must match data features ({n_features})."
            )
        if not np.all(np.isin(pat_arr, [0, 1])):
            raise ValueError("patterns matrix elements must be binary (0 or 1).")

    n_patterns = pat_arr.shape[0]
    if n_patterns == 0:
        raise ValueError("patterns matrix must contain at least 1 pattern.")

    # 3. Frequencies vector (K,)
    if freq is None:
        freq_arr = np.ones(n_patterns, dtype=float) / n_patterns
    else:
        freq_arr = np.asarray(freq, dtype=float).flatten()
        if len(freq_arr) != n_patterns:
            raise ValueError(
                f"freq length ({len(freq_arr)}) must match patterns count ({n_patterns})."
            )
        if np.any(freq_arr < 0):
            raise ValueError("freq values must be non-negative.")
        freq_sum = np.sum(freq_arr)
        if freq_sum <= 0:
            raise ValueError("freq sum must be strictly positive.")
        freq_arr = freq_arr / freq_sum

    # 4. Mechanisms list (K,)
    valid_mechs = {"MCAR", "MAR", "MNAR"}
    if isinstance(mechanisms, str):
        mech_upper = mechanisms.upper()
        if mech_upper not in valid_mechs:
            raise ValueError(f"Invalid mechanism '{mechanisms}'. Must be one of {valid_mechs}.")
        mech_list = [mech_upper] * n_patterns
    else:
        mech_list = [str(m).upper() for m in mechanisms]
        if len(mech_list) != n_patterns:
            raise ValueError(
                f"mechanisms list length ({len(mech_list)}) must match patterns count ({n_patterns})."
            )
        for m in mech_list:
            if m not in valid_mechs:
                raise ValueError(f"Invalid mechanism '{m}'. Must be one of {valid_mechs}.")

    # 5. Odds types list (K,)
    valid_odds = {"RIGHT", "LEFT", "MID", "TAIL"}
    if isinstance(odds_type, str):
        odds_upper = odds_type.upper()
        if odds_upper not in valid_odds:
            raise ValueError(f"Invalid odds_type '{odds_type}'. Must be one of {valid_odds}.")
        odds_list = [odds_upper] * n_patterns
    else:
        odds_list = [str(o).upper() for o in odds_type]
        if len(odds_list) != n_patterns:
            raise ValueError(
                f"odds_type list length ({len(odds_list)}) must match patterns count ({n_patterns})."
            )
        for o in odds_list:
            if o not in valid_odds:
                raise ValueError(f"Invalid odds_type '{o}'. Must be one of {valid_odds}.")

    # 6. Weights matrix (K, P)
    if weights is None:
        weights_arr = np.zeros((n_patterns, n_features), dtype=float)
        for k in range(n_patterns):
            mech = mech_list[k]
            if mech == "MCAR":
                weights_arr[k, :] = 0.0
            elif mech == "MAR":
                # Only observed variables (pat == 1) receive positive weight
                # Incomplete variables (pat == 0) strictly receive 0 weight
                obs_mask = pat_arr[k] == 1
                if np.any(obs_mask):
                    weights_arr[k, obs_mask] = 1.0
                else:
                    weights_arr[k, :] = 0.0
            elif mech == "MNAR":
                # Incomplete variables (pat == 0) receive weight 1.0 (self-masking)
                incomp_mask = pat_arr[k] == 0
                if np.any(incomp_mask):
                    weights_arr[k, incomp_mask] = 1.0
                else:
                    weights_arr[k, :] = 1.0
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.ndim != 2:
            raise ValueError(f"weights must be a 2D array, got shape {weights_arr.shape}.")
        if weights_arr.shape != (n_patterns, n_features):
            raise ValueError(
                f"weights shape {weights_arr.shape} must match (n_patterns={n_patterns}, n_features={n_features})."
            )

        # Invariant Check (Schouten et al. 2018):
        # Under MAR, weights for incomplete variables must be zero
        for k in range(n_patterns):
            if mech_list[k] == "MAR":
                incomp_idx = np.where(pat_arr[k] == 0)[0]
                violating = [j for j in incomp_idx if abs(weights_arr[k, j]) > 1e-12]
                if violating:
                    raise ValueError(
                        f"Under MAR mechanism, weights for incomplete variables must be strictly zero "
                        f"(Schouten et al. 2018). Pattern {k} has non-zero weights on incomplete variable(s) "
                        f"{violating}. To enable self-masking on incomplete variables, use mechanism='MNAR'."
                    )

    return df, pat_arr, freq_arr, mech_list, weights_arr, odds_list


def _calibrate_logit_shift(
    scores: np.ndarray,
    target_prop: float,
    odds_type: str,
) -> Tuple[np.ndarray, float]:
    """
    Calibrate offset b via root-finding such that mean(probability) == target_prop.

    Continuous formulas from Schouten et al. (2018) / mice::ampute:
    - RIGHT: expit( (x - mean(x)) + b )
    - LEFT:  expit( -(x - mean(x)) + b )
    - MID:   expit( -abs(x - mean(x)) + 0.75 + b )
    - TAIL:  expit( abs(x - mean(x)) - 0.75 + b )
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=float), 0.0

    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    if std_s < 1e-12 or n == 1:
        # Uniform or single-observation group: constant probability
        return np.full(n, target_prop, dtype=float), 0.0

    centered = scores - mean_s

    if odds_type == "RIGHT":
        arg = centered
    elif odds_type == "LEFT":
        arg = -centered
    elif odds_type == "MID":
        arg = -np.abs(centered) + 0.75
    elif odds_type == "TAIL":
        arg = np.abs(centered) - 0.75
    else:
        arg = centered

    def objective(b: float) -> float:
        probs = expit(arg + b)
        return float(np.mean(probs) - target_prop)

    # Monotonically increasing in b, bounded between (0, 1)
    lower, upper = -30.0, 30.0
    val_low = objective(lower)
    val_high = objective(upper)

    # Expand bracket if target_prop is extreme
    if val_low * val_high > 0:
        if val_low > 0:
            lower = -100.0
        if val_high < 0:
            upper = 100.0

    try:
        b_star = float(brentq(objective, lower, upper, xtol=1e-5, maxiter=100))
    except (ValueError, RuntimeError):
        # Fallback binary search if brentq fails
        lo, hi = lower, upper
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if objective(mid) < 0:
                lo = mid
            else:
                hi = mid
        b_star = (lo + hi) / 2.0

    probabilities = expit(arg + b_star)
    return probabilities, b_star


def ampute_multivariate(
    data: Union[pd.DataFrame, np.ndarray],
    prop: float = 0.3,
    patterns: Optional[Union[np.ndarray, Sequence[Sequence[int]]]] = None,
    freq: Optional[Union[np.ndarray, Sequence[float]]] = None,
    mechanisms: Union[str, Sequence[str]] = "MAR",
    weights: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    odds_type: Union[str, Sequence[str]] = "RIGHT",
    std_scores: bool = True,
    random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
) -> AmputationResult:
    """
    Ampute multivariate missingness into a complete dataset (Schouten et al. 2018).

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Complete dataset to ampute. Must be 2-dimensional.
    prop : float, default=0.3
        Target proportion of incomplete samples (having at least one missing value).
    patterns : array-like of shape (n_patterns, n_features), optional
        Binary matrix of missingness patterns. 0 = variable amputed (missing),
        1 = variable observed. Default creates P patterns, each missing one variable.
    freq : array-like of shape (n_patterns,), optional
        Relative frequency of each pattern. Defaults to uniform frequency (1/K).
    mechanisms : str or sequence of str, default="MAR"
        Missingness mechanism per pattern or globally: 'MCAR', 'MAR', or 'MNAR'.
    weights : array-like of shape (n_patterns, n_features), optional
        Predictor weights for computing weighted sum scores.
        Under MAR: weights for incomplete variables must be strictly 0.
        Under MNAR: weights for incomplete variables can be non-zero (self-masking).
    odds_type : str or sequence of str, default="RIGHT"
        Continuous probability distribution type: 'RIGHT', 'LEFT', 'MID', or 'TAIL'.
    std_scores : bool, default=True
        Whether to standardize predictors (zero mean, unit variance) before computing sum scores.
    random_state : int, RandomState, or Generator, optional
        Random seed for reproducible pattern assignment and stochastic dropout.

    Returns
    -------
    AmputationResult
        Dataclass containing amputed data, original data, missingness mask,
        patterns, weights, probabilities, and summary metrics.
    """
    df, pat_arr, freq_arr, mech_list, weights_arr, odds_list = _validate_and_prepare_inputs(
        data=data,
        prop=prop,
        patterns=patterns,
        freq=freq,
        mechanisms=mechanisms,
        weights=weights,
        odds_type=odds_type,
    )

    n_samples, n_features = df.shape
    n_patterns = len(pat_arr)

    # Initialize random generator
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    elif isinstance(random_state, np.random.RandomState):
        rng = np.random.default_rng(random_state.randint(0, 2**31))
    elif random_state is not None:
        rng = np.random.default_rng(int(random_state))
    else:
        rng = np.random.default_rng()

    # Assign samples to patterns according to freq
    pattern_assignments = rng.choice(n_patterns, size=n_samples, p=freq_arr)

    # Prepare feature matrix for scoring
    x_mat = df.values.astype(float)
    if std_scores:
        means = np.nanmean(x_mat, axis=0)
        stds = np.nanstd(x_mat, axis=0)
        stds[stds < 1e-12] = 1.0
        x_scaled = (x_mat - means) / stds
    else:
        x_scaled = x_mat

    probabilities = np.zeros(n_samples, dtype=float)

    # Compute amputation probabilities per pattern group
    for k in range(n_patterns):
        idx_k = np.where(pattern_assignments == k)[0]
        if len(idx_k) == 0:
            continue

        mech = mech_list[k]
        if mech == "MCAR":
            # MCAR: missing completely at random with target probability
            probabilities[idx_k] = prop
        else:
            # MAR or MNAR: compute weighted sum scores
            w_k = weights_arr[k]
            scores_k = x_scaled[idx_k] @ w_k
            probs_k, _ = _calibrate_logit_shift(
                scores=scores_k,
                target_prop=prop,
                odds_type=odds_list[k],
            )
            probabilities[idx_k] = probs_k

    # Stochastic missingness realization
    uniform_draws = rng.uniform(0.0, 1.0, size=n_samples)
    ampute_row_flag = uniform_draws < probabilities

    # Construct amputed DataFrame and boolean mask
    df_amputed = df.copy()
    mask_arr = np.zeros((n_samples, n_features), dtype=bool)

    for i in range(n_samples):
        if ampute_row_flag[i]:
            k = pattern_assignments[i]
            incomp_cols = np.where(pat_arr[k] == 0)[0]
            for c in incomp_cols:
                df_amputed.iat[i, c] = np.nan
                mask_arr[i, c] = True

    mask_df = pd.DataFrame(mask_arr, index=df.index, columns=df.columns)
    incomplete_rows = mask_arr.any(axis=1)
    empirical_prop = float(np.mean(incomplete_rows))
    n_incomplete = int(np.sum(incomplete_rows))
    cell_missing_prop = float(np.mean(mask_arr))

    return AmputationResult(
        data_amputed=df_amputed,
        data_complete=df,
        mask=mask_df,
        patterns=pat_arr,
        weights=weights_arr,
        mechanisms=mech_list,
        odds_types=odds_list,
        probabilities=probabilities,
        pattern_assignments=pattern_assignments,
        empirical_prop=empirical_prop,
        n_incomplete=n_incomplete,
        cell_missing_prop=cell_missing_prop,
    )
