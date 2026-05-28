"""
QuantStream Dashboard — Quantitative Analytics Engine
======================================================
backend/analytics/engine.py

Provides an object-oriented, pure-Python analytics engine that computes
base technical indicators and advanced risk metrics on OHLCV price data.

Design principles
-----------------
* **Lazy imports**: `pandas`, `numpy`, and `scipy` are imported *inside*
  methods, never at module scope, to respect the 512 MB RAM budget on the
  Render free tier.  Module-level imports are restricted to stdlib types used
  for annotations only.
* **Stateless calculations**: every public method is a pure function of its
  inputs; the ``QuantEngine`` instance itself holds *no* mutable state beyond
  the raw DataFrame passed at construction.
* **Strict typing**: all public signatures are annotated with ``pandas`` stub
  types kept behind ``TYPE_CHECKING`` so the runtime never loads the library
  just to satisfy annotations.
* **Explicit error handling**: every method raises ``ValueError`` with a
  descriptive message on bad inputs rather than silently returning NaN-filled
  frames.

Usage
-----
>>> from backend.analytics.engine import QuantEngine
>>> engine = QuantEngine(ohlcv_df)          # ohlcv_df from yfinance
>>> returns   = engine.log_returns()
>>> vol       = engine.rolling_volatility()
>>> macd_data = engine.macd()
>>> rsi_data  = engine.rsi()
>>> zscore    = engine.sma_zscore()
>>> gk_vol    = engine.garman_klass_volatility()
>>> cvar      = engine.cvar(confidence=0.95)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401 — annotation-only, never imported at runtime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_ROWS_BASE: int = 30     # minimum rows needed for most base indicators
_MIN_ROWS_SMA200: int = 200  # minimum rows needed for 200-day SMA z-score
_MIN_ROWS_GK: int = 5        # minimum rows for a meaningful GK estimate
_TRADING_DAYS: int = 252     # annualisation constant


class QuantEngine:
    """
    Stateless quantitative analytics engine for US equity OHLCV data.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        A *DatetimeIndex*-ed DataFrame with at minimum the columns
        ``['Open', 'High', 'Low', 'Close', 'Volume']`` as returned by
        ``yfinance.download()``.  Column names are case-sensitive.

    Raises
    ------
    ValueError
        If ``ohlcv`` is missing required columns or has fewer than
        ``_MIN_ROWS_BASE`` rows after dropping NaNs.
    """

    _REQUIRED_OHLCV_COLS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
    _REQUIRED_PRICE_COLS: tuple[str, ...] = ("Close",)

    # ------------------------------------------------------------------
    # Construction & validation
    # ------------------------------------------------------------------

    def __init__(self, ohlcv: "pd.DataFrame") -> None:
        import pandas as pd  # lazy

        self._df: pd.DataFrame = self._validate(ohlcv)
        logger.debug(
            "QuantEngine initialised with %d rows (%s → %s).",
            len(self._df),
            self._df.index[0].date(),
            self._df.index[-1].date(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: "pd.DataFrame") -> "pd.DataFrame":
        """
        Validate and lightly normalise the raw OHLCV DataFrame.

        * Sorts by index ascending.
        * Drops rows where *all* OHLCV values are NaN.
        * Enforces minimum row count.

        Returns
        -------
        pd.DataFrame
            A cleaned copy of the input.

        Raises
        ------
        ValueError
            On structural issues with the input.
        """
        import pandas as pd  # lazy

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )

        # yfinance sometimes returns multi-level columns; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(level=1, axis=1)

        missing = [c for c in QuantEngine._REQUIRED_OHLCV_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"OHLCV DataFrame is missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        df = df.sort_index().dropna(subset=list(QuantEngine._REQUIRED_OHLCV_COLS), how="all").copy()

        if len(df) < _MIN_ROWS_BASE:
            raise ValueError(
                f"Insufficient data: need at least {_MIN_ROWS_BASE} rows, "
                f"got {len(df)} after dropping NaN rows."
            )

        return df

    @property
    def close(self) -> "pd.Series":
        """Return the ``Close`` price series (lazy accessor, no copy)."""
        return self._df["Close"]

    # ------------------------------------------------------------------
    # Base metrics
    # ------------------------------------------------------------------

    def log_returns(self) -> "pd.Series":
        """
        Compute daily log returns: ``r_t = ln(P_t / P_{t-1})``.

        Log returns are preferred over simple returns for quantitative work
        because they are time-additive, approximately normally distributed,
        and well-defined under compounding.

        Returns
        -------
        pd.Series
            Named ``'log_return'``.  First element is ``NaN`` by construction.

        Examples
        --------
        >>> engine.log_returns().dropna().describe()
        """
        import numpy as np  # lazy

        returns: "pd.Series" = np.log(self.close / self.close.shift(1))
        returns.name = "log_return"
        logger.debug("log_returns(): computed %d values.", len(returns))
        return returns

    def rolling_volatility(
        self,
        window: int = 21,
        annualise: bool = True,
    ) -> "pd.Series":
        """
        Compute rolling close-to-close historical volatility (standard
        deviation of log returns).

        Parameters
        ----------
        window : int, default 21
            Look-back window in trading days.  Common choices:
            ``10`` (2-week), ``21`` (1-month), ``63`` (1-quarter).
        annualise : bool, default True
            If ``True``, multiplies by ``sqrt(252)`` to express as an
            annualised figure (consistent with implied-vol quoting).

        Returns
        -------
        pd.Series
            Named ``'rolling_vol_<window>d'`` (or ``..._ann`` when annualised).

        Raises
        ------
        ValueError
            If ``window`` is larger than the number of available rows.
        """
        import numpy as np  # lazy

        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}.")
        if window > len(self._df):
            raise ValueError(
                f"window ({window}) exceeds available rows ({len(self._df)})."
            )

        returns = self.log_returns()
        vol = returns.rolling(window=window, min_periods=window).std()
        if annualise:
            vol = vol * np.sqrt(_TRADING_DAYS)
            vol.name = f"rolling_vol_{window}d_ann"
        else:
            vol.name = f"rolling_vol_{window}d"

        logger.debug("rolling_volatility(window=%d, annualise=%s): done.", window, annualise)
        return vol

    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> "pd.DataFrame":
        """
        Compute the MACD (Moving Average Convergence/Divergence) indicator.

        Uses *exponential* moving averages as per the original Appel (1979)
        definition.  The ``adjust=False`` flag ensures the EMA is computed
        using the standard recursive formula rather than the expanding-window
        approximation.

        Parameters
        ----------
        fast : int, default 12
            Span for the fast EMA.
        slow : int, default 26
            Span for the slow EMA.
        signal : int, default 9
            Span for the signal-line EMA applied to the MACD line.

        Returns
        -------
        pd.DataFrame
            Columns: ``['macd_line', 'signal_line', 'histogram']``.

        Raises
        ------
        ValueError
            If fast >= slow or any parameter is < 1.
        """
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow}).")
        if any(p < 1 for p in (fast, slow, signal)):
            raise ValueError("fast, slow, and signal must all be >= 1.")

        ema_fast = self.close.ewm(span=fast, adjust=False).mean()
        ema_slow = self.close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        import pandas as pd  # lazy — needed for DataFrame construction

        result = pd.DataFrame(
            {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
            }
        )
        logger.debug("macd(fast=%d, slow=%d, signal=%d): done.", fast, slow, signal)
        return result

    def rsi(self, period: int = 14) -> "pd.Series":
        """
        Compute the Relative Strength Index (RSI) using Wilder's smoothed
        moving average (equivalent to ``EWM`` with ``alpha = 1 / period``).

        RSI ranges from 0 to 100.  Convention:
        * RSI > 70 → overbought territory.
        * RSI < 30 → oversold territory.

        Parameters
        ----------
        period : int, default 14
            Look-back period (Wilder originally used 14 days).

        Returns
        -------
        pd.Series
            Named ``'rsi_<period>'``, values in ``[0, 100]``.

        Raises
        ------
        ValueError
            If ``period < 2``.
        """
        if period < 2:
            raise ValueError(f"RSI period must be >= 2, got {period}.")

        delta = self.close.diff(1)
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing = EWM with com = period - 1  (alpha = 1/period)
        avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_series.name = f"rsi_{period}"

        logger.debug("rsi(period=%d): done.", period)
        return rsi_series

    def sma_zscore(
        self,
        window: int = 200,
        zscore_window: Optional[int] = None,
    ) -> "pd.DataFrame":
        """
        Compute the Z-Score of price relative to its rolling SMA.

        The z-score answers "how many standard deviations is today's price
        above/below its *n*-day moving average?" and is used to identify
        mean-reversion signals or trend strength.

        Formula
        -------
        ``z = (Close - SMA_n) / StdDev_n``

        where both ``SMA_n`` and ``StdDev_n`` use a rolling window of size
        ``window`` (or ``zscore_window`` if supplied).

        Parameters
        ----------
        window : int, default 200
            The SMA period (default 200-day SMA).
        zscore_window : int, optional
            The rolling window for the z-score denominator.  If ``None``
            (default), uses the same value as ``window``.

        Returns
        -------
        pd.DataFrame
            Columns: ``['sma', 'zscore']``.

        Raises
        ------
        ValueError
            If insufficient data for the requested window.
        """
        zw = zscore_window or window
        if window > len(self._df):
            raise ValueError(
                f"SMA window ({window}) exceeds available rows ({len(self._df)}). "
                f"Fetch more historical data or use a shorter window."
            )

        sma = self.close.rolling(window=window, min_periods=window).mean()
        std = self.close.rolling(window=zw, min_periods=zw).std()
        zscore = (self.close - sma) / std.replace(0, float("nan"))

        import pandas as pd  # lazy

        result = pd.DataFrame({"sma": sma, "zscore": zscore})
        logger.debug(
            "sma_zscore(window=%d, zscore_window=%d): done.", window, zw
        )
        return result

    # ------------------------------------------------------------------
    # Advanced risk metrics
    # ------------------------------------------------------------------

    def garman_klass_volatility(
        self,
        window: int = 21,
        annualise: bool = True,
    ) -> "pd.Series":
        """
        Compute the Garman-Klass (1980) volatility estimator.

        GK volatility uses the full intra-day range (Open, High, Low, Close)
        to extract more information per observation than the close-to-close
        estimator, resulting in a theoretically more efficient estimate
        (roughly 7–8× as efficient under geometric Brownian motion).

        Formula
        -------
        ``GK_t = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2``

        The rolling GK volatility is then:
        ``sigma_GK = sqrt( mean(GK_t over window) )``

        Parameters
        ----------
        window : int, default 21
            Look-back window in trading days.
        annualise : bool, default True
            If ``True``, multiplies by ``sqrt(252)``.

        Returns
        -------
        pd.Series
            Named ``'gk_vol_<window>d'`` (or ``..._ann`` when annualised).

        Raises
        ------
        ValueError
            If OHLC columns are missing or the window is too large.

        References
        ----------
        Garman, M. B. & Klass, M. J. (1980). *On the estimation of security
        price volatilities from historical data.* Journal of Business, 53(1),
        67–78.
        """
        import numpy as np  # lazy

        for col in ("Open", "High", "Low", "Close"):
            if col not in self._df.columns:
                raise ValueError(f"Column '{col}' required for Garman-Klass but not found.")

        if window > len(self._df):
            raise ValueError(
                f"GK window ({window}) exceeds available rows ({len(self._df)})."
            )

        log_hl = np.log(self._df["High"] / self._df["Low"])
        log_co = np.log(self._df["Close"] / self._df["Open"])

        # Per-day GK component (variance contribution)
        gk_daily = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2

        # Rolling mean of daily GK, then sqrt to get volatility
        gk_var = gk_daily.rolling(window=window, min_periods=window).mean()
        gk_vol = np.sqrt(gk_var.clip(lower=0))  # clip prevents sqrt(negative) from floating point

        if annualise:
            gk_vol = gk_vol * np.sqrt(_TRADING_DAYS)
            gk_vol.name = f"gk_vol_{window}d_ann"
        else:
            gk_vol.name = f"gk_vol_{window}d"

        logger.debug(
            "garman_klass_volatility(window=%d, annualise=%s): done.", window, annualise
        )
        return gk_vol

    def cvar(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        method: str = "historical",
    ) -> dict[str, float]:
        """
        Compute the Conditional Value-at-Risk (CVaR), also known as Expected
        Shortfall (ES).

        CVaR is the *expected loss* given that the loss exceeds the VaR
        threshold.  It is a coherent risk measure (Artzner et al., 1999) and
        preferred over VaR for capturing tail risk.

        Parameters
        ----------
        confidence : float, default 0.95
            Confidence level, e.g. ``0.95`` for 95% CVaR.
            Must be in ``(0, 1)``.
        horizon : int, default 1
            Holding period in trading days.  For horizons > 1, the
            square-root-of-time scaling rule is applied (assumes i.i.d. returns).
            Note: SRT is an approximation; use with caution for long horizons.
        method : str, default ``'historical'``
            Estimation method.  Currently supports:

            * ``'historical'``: non-parametric, uses the empirical return
              distribution directly.
            * ``'parametric'``: assumes normally-distributed returns and uses
              the closed-form normal CVaR formula via ``scipy.stats``.

        Returns
        -------
        dict[str, float]
            A mapping with the following keys:

            * ``'var'``: Value-at-Risk at the requested confidence level
              (expressed as a *negative* return, i.e. a loss).
            * ``'cvar'``: Conditional Value-at-Risk / Expected Shortfall.
            * ``'confidence'``: The ``confidence`` value used.
            * ``'horizon_days'``: The ``horizon`` value used.
            * ``'n_observations'``: Number of return observations used.

        Raises
        ------
        ValueError
            If ``confidence`` is not in ``(0, 1)`` or ``method`` is unknown.

        References
        ----------
        Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). *Coherent
        measures of risk.* Mathematical Finance, 9(3), 203–228.
        """
        import numpy as np  # lazy

        if not (0 < confidence < 1):
            raise ValueError(
                f"confidence must be in (0, 1), got {confidence}."
            )
        if method not in ("historical", "parametric"):
            raise ValueError(
                f"method must be 'historical' or 'parametric', got '{method}'."
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")

        returns = self.log_returns().dropna().values  # numpy array

        if len(returns) < 30:
            raise ValueError(
                f"CVaR requires at least 30 return observations, got {len(returns)}."
            )

        # Scale to horizon using square-root-of-time rule
        if horizon > 1:
            returns = returns * np.sqrt(horizon)

        if method == "historical":
            var_threshold = np.quantile(returns, 1.0 - confidence)
            tail_losses = returns[returns <= var_threshold]
            cvar_value = float(tail_losses.mean()) if len(tail_losses) > 0 else var_threshold
            var_value = float(var_threshold)

        else:  # parametric (normal)
            from scipy import stats  # lazy — only imported for parametric path

            mu: float = float(np.mean(returns))
            sigma: float = float(np.std(returns, ddof=1))
            z_alpha: float = stats.norm.ppf(1.0 - confidence)

            # Normal CVaR closed form: mu - sigma * phi(z) / (1 - alpha)
            pdf_z = stats.norm.pdf(z_alpha)
            var_value = float(mu + sigma * z_alpha)                          # negative loss
            cvar_value = float(mu - sigma * pdf_z / (1.0 - confidence))     # more negative

        logger.debug(
            "cvar(confidence=%.2f, horizon=%d, method=%s): VaR=%.4f, CVaR=%.4f.",
            confidence,
            horizon,
            method,
            var_value,
            cvar_value,
        )

        return {
            "var": var_value,
            "cvar": cvar_value,
            "confidence": confidence,
            "horizon_days": horizon,
            "n_observations": len(returns),
        }

    # ------------------------------------------------------------------
    # Convenience: compute all metrics at once
    # ------------------------------------------------------------------

    def compute_all(
        self,
        vol_window: int = 21,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        sma_window: int = 200,
        gk_window: int = 21,
        cvar_confidence: float = 0.95,
    ) -> dict[str, object]:
        """
        Compute all metrics in a single call and return as a named dict.

        This is the primary interface used by FastAPI endpoint handlers so they
        can call one method and serialise the result.  Gracefully skips metrics
        that cannot be computed (e.g. 200-day SMA when < 200 rows available)
        and logs a warning instead of raising.

        Returns
        -------
        dict[str, object]
            Keys: ``'log_returns'``, ``'rolling_vol'``, ``'macd'``, ``'rsi'``,
            ``'sma_zscore'``, ``'gk_vol'``, ``'cvar'``.
            Values are pandas Series/DataFrames or dicts as documented on
            each individual method.
        """
        results: dict[str, object] = {}

        # --- log returns ---
        results["log_returns"] = self.log_returns()

        # --- rolling volatility ---
        results["rolling_vol"] = self.rolling_volatility(window=vol_window)

        # --- MACD ---
        try:
            results["macd"] = self.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal)
        except ValueError as exc:
            logger.warning("MACD skipped: %s", exc)
            results["macd"] = None

        # --- RSI ---
        results["rsi"] = self.rsi(period=rsi_period)

        # --- 200-day SMA z-score (graceful degradation) ---
        try:
            results["sma_zscore"] = self.sma_zscore(window=sma_window)
        except ValueError as exc:
            logger.warning("SMA z-score skipped: %s", exc)
            results["sma_zscore"] = None

        # --- Garman-Klass volatility ---
        results["gk_vol"] = self.garman_klass_volatility(window=gk_window)

        # --- CVaR ---
        try:
            results["cvar"] = self.cvar(confidence=cvar_confidence)
        except ValueError as exc:
            logger.warning("CVaR skipped: %s", exc)
            results["cvar"] = None

        return results