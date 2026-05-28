"""
QuantStream Dashboard — Market Regime Classifier
=================================================
backend/analytics/regimes.py

Implements a 3-state Hidden Markov Model (HMM) that classifies each trading
day into one of three latent market regimes:

    State 0 — Low-Volatility Bull  (quiet uptrend)
    State 1 — High-Volatility Bear (stressed / trending down)
    State 2 — Transitional / Ranging  (sideways, moderate volatility)

The *semantic* labels are inferred post-hoc from the trained model's per-state
mean and variance of log returns; the HMM itself is agnostic to their meaning.

Architecture
------------
* ``RegimeClassifier``  — high-level, API-facing facade.
* ``_HMMWrapper``       — thin wrapper around ``hmmlearn.hmm.GaussianHMM`` that
                          encapsulates fit/predict/score.
* ``RegimeResult``      — typed dataclass that bundles the output for easy
                          serialisation by FastAPI.

Lazy imports
------------
``numpy``, ``pandas``, ``hmmlearn``, and ``scikit-learn`` are all imported
*inside* method bodies, not at module scope, to keep the idle memory footprint
near zero on the Render free tier.

Usage
-----
>>> from backend.analytics.regimes import RegimeClassifier
>>> clf = RegimeClassifier(n_states=3, n_iter=200, random_state=42)
>>> result = clf.fit_predict(log_returns_series)
>>> print(result.regime_labels)        # pd.Series of int {0, 1, 2}
>>> print(result.current_regime_name)  # e.g. "Low-Volatility Bull"
>>> print(result.transition_matrix)    # 3×3 np.ndarray
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np      # noqa: F401
    import pandas as pd     # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_N_STATES: int = 3
_DEFAULT_N_ITER: int = 200
_DEFAULT_RANDOM_STATE: int = 42
_MIN_TRAINING_ROWS: int = 100  # HMM needs enough observations to converge

# Semantic regime names — assigned post-hoc based on state mean return
# and variance ordering.
_REGIME_NAMES: dict[str, str] = {
    "bull": "Low-Volatility Bull",
    "bear": "High-Volatility Bear",
    "transition": "Transitional / Ranging",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """
    Bundles all outputs from ``RegimeClassifier.fit_predict()`` into a single,
    JSON-serialisable structure (all numpy types are python-native here).

    Attributes
    ----------
    regime_labels : pd.Series
        Integer regime label (0, 1, or 2) for every row in the input,
        indexed by the original DatetimeIndex.
    regime_names : list[str]
        Human-readable name for each integer label, e.g.
        ``['Low-Volatility Bull', 'High-Volatility Bear', 'Transitional / Ranging']``.
    current_regime : int
        Integer label for the *most recent* observation.
    current_regime_name : str
        Human-readable name of ``current_regime``.
    state_means : list[float]
        Per-state mean of the feature vector used during training.
        For a univariate model this is a list of 3 floats (one per state).
    state_stds : list[float]
        Per-state standard deviation of the feature vector.
    transition_matrix : list[list[float]]
        Row-stochastic 3×3 transition probability matrix.
        ``transition_matrix[i][j]`` = P(state j | state i).
    log_likelihood : float
        Final log-likelihood of the training data under the fitted model.
        Higher (less negative) is better; useful for convergence checks.
    n_training_samples : int
        Number of observations used to fit the model.
    """

    regime_labels: "pd.Series"
    regime_names: list[str]
    current_regime: int
    current_regime_name: str
    state_means: list[float]
    state_stds: list[float]
    transition_matrix: list[list[float]]
    log_likelihood: float
    n_training_samples: int
    feature_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Serialise to a plain Python dict suitable for JSON encoding by FastAPI.

        ``regime_labels`` is converted to ``{date_str: int}`` mapping.
        """
        return {
            "regime_labels": {
                str(k.date()): int(v)
                for k, v in self.regime_labels.items()
            },
            "regime_names": self.regime_names,
            "current_regime": self.current_regime,
            "current_regime_name": self.current_regime_name,
            "state_means": self.state_means,
            "state_stds": self.state_stds,
            "transition_matrix": self.transition_matrix,
            "log_likelihood": self.log_likelihood,
            "n_training_samples": self.n_training_samples,
            "feature_columns": self.feature_columns,
        }


# ---------------------------------------------------------------------------
# Private HMM wrapper
# ---------------------------------------------------------------------------

class _HMMWrapper:
    """
    Internal thin wrapper around ``hmmlearn.hmm.GaussianHMM``.

    Encapsulates the fit/predict cycle so that ``RegimeClassifier`` can swap
    the underlying estimator in tests without touching business logic.

    Parameters
    ----------
    n_states : int
        Number of latent HMM states.
    n_iter : int
        Maximum EM iterations.
    covariance_type : str
        One of ``'diag'``, ``'full'``, ``'tied'``, ``'spherical'``.
        ``'diag'`` is recommended for stability with small observation vectors.
    random_state : int
        Seed for reproducible initialisation.
    tol : float
        Convergence threshold for the EM algorithm's log-likelihood delta.
    """

    def __init__(
        self,
        n_states: int = _DEFAULT_N_STATES,
        n_iter: int = _DEFAULT_N_ITER,
        covariance_type: str = "diag",
        random_state: int = _DEFAULT_RANDOM_STATE,
        tol: float = 1e-4,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.tol = tol
        self._model: object = None  # set after fit()

    def fit(self, X: "np.ndarray") -> "_HMMWrapper":
        """
        Fit the Gaussian HMM via the Baum-Welch (EM) algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Observation sequence.  NaN rows must be removed before calling.

        Returns
        -------
        self

        Raises
        ------
        ImportError
            If ``hmmlearn`` is not installed.
        RuntimeError
            If the EM algorithm fails to converge (non-fatal warning only;
            the best intermediate model is still returned).
        """
        try:
            from hmmlearn.hmm import GaussianHMM  # lazy
        except ImportError as exc:
            raise ImportError(
                "hmmlearn is required for regime detection.  "
                "Install it with: pip install hmmlearn"
            ) from exc

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=False,
        )
        model.fit(X)

        if not model.monitor_.converged:
            logger.warning(
                "_HMMWrapper: EM did not converge after %d iterations. "
                "Consider increasing n_iter or checking input quality.",
                self.n_iter,
            )
        else:
            logger.debug(
                "_HMMWrapper: EM converged in %d iterations.",
                len(model.monitor_.history),
            )

        self._model = model
        return self

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        """
        Decode the most likely state sequence via the Viterbi algorithm.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of int, shape (n_samples,)
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X)

    def score(self, X: "np.ndarray") -> float:
        """Return the per-sample log-likelihood of X under the fitted model."""
        if self._model is None:
            raise RuntimeError("Call fit() before score().")
        return float(self._model.score(X))

    @property
    def means(self) -> "np.ndarray":
        """Per-state mean vectors, shape (n_states, n_features)."""
        return self._model.means_

    @property
    def covars(self) -> "np.ndarray":
        """Per-state covariance matrices (shape depends on covariance_type)."""
        return self._model.covars_

    @property
    def transmat(self) -> "np.ndarray":
        """Row-stochastic transition matrix, shape (n_states, n_states)."""
        return self._model.transmat_


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class RegimeClassifier:
    """
    High-level market regime classifier backed by a 3-state Gaussian HMM.

    The classifier uses a **multi-feature observation vector** per day to
    improve state separation beyond single-feature models:

    1. **Log return**          — captures direction and magnitude of daily move.
    2. **5-day rolling vol**   — fast-reacting volatility signal.
    3. **21-day rolling vol**  — slower, regime-level volatility baseline.

    These three features are *standardised* (zero-mean, unit-variance) before
    fitting so that the HMM's Gaussian emissions are well-conditioned regardless
    of the stock's price level.

    The three latent states are labelled post-hoc:

    * **Low-Volatility Bull**:  state with the highest mean return *and* lowest
      variance (calm up-trend).
    * **High-Volatility Bear**: state with the lowest (most negative) mean return
      *or* highest variance (stressed market).
    * **Transitional / Ranging**: the remaining state.

    Parameters
    ----------
    n_states : int, default 3
        Number of HMM hidden states.  Changing this will alter the label-
        assignment logic; only override for research purposes.
    n_iter : int, default 200
        Maximum EM iterations for Baum-Welch.
    random_state : int, default 42
        Seed for reproducible model initialisation.
    covariance_type : str, default ``'diag'``
        HMM covariance structure.  ``'diag'`` is the most numerically stable
        choice for small feature vectors.
    """

    def __init__(
        self,
        n_states: int = _DEFAULT_N_STATES,
        n_iter: int = _DEFAULT_N_ITER,
        random_state: int = _DEFAULT_RANDOM_STATE,
        covariance_type: str = "diag",
    ) -> None:
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}.")
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.covariance_type = covariance_type
        self._hmm = _HMMWrapper(
            n_states=n_states,
            n_iter=n_iter,
            covariance_type=covariance_type,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_feature_matrix(
        self,
        log_returns: "pd.Series",
    ) -> tuple["pd.DataFrame", list[str]]:
        """
        Construct and standardise the multi-feature observation matrix.

        Parameters
        ----------
        log_returns : pd.Series
            Daily log returns (as produced by ``QuantEngine.log_returns()``).

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            * Standardised feature DataFrame (NaN rows dropped).
            * List of feature column names (for provenance tracking).
        """
        import pandas as pd   # lazy
        import numpy as np    # lazy

        features = pd.DataFrame(index=log_returns.index)
        features["log_return"] = log_returns

        # Fast and slow rolling vol — captures vol regime changes
        features["vol_5d"]  = log_returns.rolling(window=5,  min_periods=5).std()
        features["vol_21d"] = log_returns.rolling(window=21, min_periods=21).std()

        feature_cols = ["log_return", "vol_5d", "vol_21d"]
        features = features[feature_cols].dropna()

        if len(features) < _MIN_TRAINING_ROWS:
            raise ValueError(
                f"After building features and dropping NaN rows, only "
                f"{len(features)} observations remain (need ≥ {_MIN_TRAINING_ROWS}). "
                "Supply more historical data."
            )

        # Standardise each feature to zero-mean, unit-variance
        means = features.mean()
        stds  = features.std(ddof=1).replace(0, 1.0)  # avoid div-by-zero on flat series
        features_std = (features - means) / stds

        logger.debug(
            "_build_feature_matrix: %d observations, features=%s.",
            len(features_std),
            feature_cols,
        )
        return features_std, feature_cols

    # ------------------------------------------------------------------
    # Regime label assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_regime_names(
        hmm: _HMMWrapper,
        n_states: int,
    ) -> dict[int, str]:
        """
        Map raw integer HMM state indices to semantic regime names.

        Assignment logic (applied in priority order):

        1. The state with the **highest mean log-return** among those with
           **below-median variance** is labelled *"Low-Volatility Bull"*.
        2. The state with the **lowest mean log-return** OR the highest
           variance is labelled *"High-Volatility Bear"*.
        3. Any remaining state(s) become *"Transitional / Ranging"*.

        This heuristic is robust to the HMM's arbitrary state-index assignment
        across different random seeds or training windows.

        Parameters
        ----------
        hmm : _HMMWrapper
            A *fitted* HMM wrapper.
        n_states : int
            Expected number of states (used for validation).

        Returns
        -------
        dict[int, str]
            Maps ``{state_index: regime_name}``.
        """
        import numpy as np  # lazy

        # means_ shape: (n_states, n_features); take first feature (log_return)
        raw_means = hmm.means[:, 0]          # mean log return per state

        # For 'diag' covariance: covars_ shape (n_states, n_features)
        # Take first feature's variance
        covars = hmm.covars
        if covars.ndim == 2:
            raw_vars = covars[:, 0]          # diag: (n_states, n_features)
        elif covars.ndim == 3:
            raw_vars = covars[:, 0, 0]       # full: (n_states, n_features, n_features)
        else:
            raw_vars = np.ones(n_states)     # spherical fallback

        median_var = float(np.median(raw_vars))
        assignments: dict[int, str] = {}

        # Step 1: Bull — highest mean return among low-variance states
        low_var_states = [i for i in range(n_states) if raw_vars[i] <= median_var]
        if low_var_states:
            bull_state = int(max(low_var_states, key=lambda i: raw_means[i]))
        else:
            bull_state = int(np.argmax(raw_means))
        assignments[bull_state] = _REGIME_NAMES["bull"]

        # Step 2: Bear — lowest mean *or* highest variance, excluding bull
        remaining = [i for i in range(n_states) if i != bull_state]
        if remaining:
            # Score: heavily penalise low mean and high variance
            bear_score = {i: -raw_means[i] + raw_vars[i] for i in remaining}
            bear_state = max(bear_score, key=bear_score.get)
            assignments[bear_state] = _REGIME_NAMES["bear"]
            remaining = [i for i in remaining if i != bear_state]

        # Step 3: Everything else → Transitional
        for i in remaining:
            assignments[i] = _REGIME_NAMES["transition"]

        logger.debug(
            "_assign_regime_names: assignments=%s, means=%s, vars=%s.",
            assignments,
            raw_means.tolist(),
            raw_vars.tolist(),
        )
        return assignments

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        log_returns: "pd.Series",
    ) -> RegimeResult:
        """
        Fit the HMM on the full log-return series and decode regime labels.

        This is a single-pass operation: the model is trained on the entire
        input and then used to decode the same sequence via Viterbi.  For
        production use, re-fit daily with the latest data by calling this
        method on each request (stateless design).

        Parameters
        ----------
        log_returns : pd.Series
            Daily log returns indexed by a ``DatetimeIndex``.  Typically
            produced by ``QuantEngine.log_returns()``.

        Returns
        -------
        RegimeResult
            Fully populated result object.  Call ``.to_dict()`` for
            FastAPI serialisation.

        Raises
        ------
        ValueError
            If input is too short or malformed.
        ImportError
            If ``hmmlearn`` is not installed in the environment.

        Examples
        --------
        >>> from backend.analytics.engine import QuantEngine
        >>> from backend.analytics.regimes import RegimeClassifier
        >>> engine = QuantEngine(ohlcv_df)
        >>> clf    = RegimeClassifier()
        >>> result = clf.fit_predict(engine.log_returns())
        >>> result.current_regime_name
        'Low-Volatility Bull'
        """
        import numpy as np   # lazy
        import pandas as pd  # lazy

        if not isinstance(log_returns, pd.Series):
            raise ValueError(
                f"log_returns must be a pd.Series, got {type(log_returns).__name__}."
            )

        # --- Feature engineering ---
        features_df, feature_cols = self._build_feature_matrix(log_returns)
        X: "np.ndarray" = features_df.values.astype(np.float64)

        logger.info(
            "RegimeClassifier.fit_predict: fitting HMM on %d observations, "
            "%d features.",
            len(X),
            X.shape[1],
        )

        # --- Fit HMM ---
        self._hmm.fit(X)

        # --- Decode Viterbi path ---
        raw_labels: "np.ndarray" = self._hmm.predict(X)
        log_likelihood: float = self._hmm.score(X)

        # --- Assign semantic names ---
        name_map: dict[int, str] = self._assign_regime_names(self._hmm, self.n_states)
        regime_names: list[str] = [
            name_map.get(i, _REGIME_NAMES["transition"])
            for i in range(self.n_states)
        ]

        # --- Build output Series aligned to original DatetimeIndex ---
        labels_series = pd.Series(
            raw_labels,
            index=features_df.index,
            name="regime",
            dtype=int,
        )

        # --- Current regime (last available observation) ---
        current_regime: int = int(labels_series.iloc[-1])
        current_regime_name: str = name_map.get(current_regime, _REGIME_NAMES["transition"])

        # --- Extract per-state stats (back in original scale, first feature) ---
        raw_means_list: list[float] = [
            float(self._hmm.means[i, 0]) for i in range(self.n_states)
        ]
        covars = self._hmm.covars
        if covars.ndim == 2:
            raw_stds_list: list[float] = [
                float(np.sqrt(covars[i, 0])) for i in range(self.n_states)
            ]
        elif covars.ndim == 3:
            raw_stds_list = [
                float(np.sqrt(covars[i, 0, 0])) for i in range(self.n_states)
            ]
        else:
            raw_stds_list = [float(np.sqrt(covars[i])) for i in range(self.n_states)]

        transition_matrix: list[list[float]] = self._hmm.transmat.tolist()

        logger.info(
            "RegimeClassifier.fit_predict: done. "
            "Current regime: '%s' (state %d). Log-likelihood: %.4f.",
            current_regime_name,
            current_regime,
            log_likelihood,
        )

        return RegimeResult(
            regime_labels=labels_series,
            regime_names=regime_names,
            current_regime=current_regime,
            current_regime_name=current_regime_name,
            state_means=raw_means_list,
            state_stds=raw_stds_list,
            transition_matrix=transition_matrix,
            log_likelihood=log_likelihood,
            n_training_samples=len(X),
            feature_columns=feature_cols,
        )

    def predict_latest(
        self,
        log_returns: "pd.Series",
        lookback: Optional[int] = None,
    ) -> dict[str, object]:
        """
        Convenience method that runs ``fit_predict`` and returns only the
        fields needed by the Streamlit dashboard's status bar.

        Parameters
        ----------
        log_returns : pd.Series
            Full daily log-return series.
        lookback : int, optional
            If provided, only the last ``lookback`` rows are used for fitting.
            Useful for rolling-window regime detection.  Must be >= 100.

        Returns
        -------
        dict with keys:
            ``'current_regime'`` (int),
            ``'current_regime_name'`` (str),
            ``'transition_matrix'`` (list[list[float]]),
            ``'state_means'`` (list[float]),
            ``'state_stds'`` (list[float]),
            ``'log_likelihood'`` (float).
        """
        import pandas as pd  # lazy

        if lookback is not None:
            if lookback < _MIN_TRAINING_ROWS:
                raise ValueError(
                    f"lookback ({lookback}) must be >= {_MIN_TRAINING_ROWS}."
                )
            log_returns = log_returns.iloc[-lookback:]

        result = self.fit_predict(log_returns)
        return {
            "current_regime": result.current_regime,
            "current_regime_name": result.current_regime_name,
            "transition_matrix": result.transition_matrix,
            "state_means": result.state_means,
            "state_stds": result.state_stds,
            "log_likelihood": result.log_likelihood,
        }