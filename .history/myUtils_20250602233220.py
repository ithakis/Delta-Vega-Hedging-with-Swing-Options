# --- Standard library ---
import random
import warnings
from functools import partial
from math import floor
from typing import Optional, Tuple

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

# import pandas as pd
import matplotlib.pyplot as plt

# --- Third-party packages ---
import numpy as np
import py_vollib_vectorized
import QuantLib as ql
import scienceplots
import seaborn as sns
from joblib import Parallel, delayed
from numba import jit, njit, prange
from numpy import cos, exp, log, pi, sin, sqrt
from scipy.special import erf
from scipy.stats import (
    gaussian_kde,  # kernel density estimate
    kstest,  # Kolmogorov–Smirnov
    norm,  # Gaussian PDF/CDF
    normaltest,  # D’Agostino K²
    probplot,  # Q-Q plot
    qmc,  # Sobol sampler
)
from tqdm import trange

# --- Module-level settings ---
random.seed(1)
plt.style.use(["science", "no-latex"])
warnings.filterwarnings("ignore", message="Mean of empty slice")


def get_implied_volatility(price, S, K, t, r, q, flag) -> np.ndarray:
    """
    Compute the Black-Scholes-Merton implied volatility for given option prices.

    Uses py_vollib_vectorized to invert the Black-Scholes-Merton formula.

    Parameters
    ----------
    price : array_like
        Market option prices.
    S : array_like
        Underlying spot prices.
    K : array_like
        Option strike prices.
    t : array_like or float
        Time to maturity (in years).
    r : float
        Risk-free interest rate (annualized).
    q : float
        Continuous dividend yield (annualized).
    flag : {'c','p'} or {'call','put'}
        Option type: 'c' or 'call' for Call, 'p' or 'put' for Put.

    Returns
    -------
    implied_vol : ndarray
        Implied volatilities corresponding to each input price.
    """
    result = py_vollib_vectorized.vectorized_implied_volatility(
        price=price,
        S=S,
        K=K,
        t=t,
        r=r,
        flag=flag,
        q=q,
        on_error="ignore",
        model="black_scholes_merton",
        return_as="numpy",
    )
    return np.asarray(result)


def QESim(
    S0: float,
    V0: float,
    rho: float,
    theta: float,
    sigma: float,
    kappa: float,
    r: float,
    q: float,
    dt: float,
    T_steps: int,
    N_paths: int,
    psiC: float = 1.5,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
    Martingale_Correction: bool = True,
    shock_step: Optional[int] = None,
    sigma_eps:float = 0.08,
    _show_progress: bool = True,
    _plot: bool = False,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heston QE simulation using NumPy RNG instead of Sobol sequences.
    Returns
    -------
    S : ndarray, shape (N_paths, T_steps+1)
    V : ndarray, shape (N_paths, T_steps+1)
    """
    # Precompute constants
    E = np.exp(-kappa * dt)
    K0 = -(kappa * rho * theta) / sigma * dt
    K1 = (kappa * rho / sigma - 0.5) * gamma1 * dt - rho / sigma
    K2 = (kappa * rho / sigma - 0.5) * gamma2 * dt + rho / sigma
    K3 = (1 - rho**2) * gamma1 * dt
    K4 = (1 - rho**2) * gamma2 * dt
    A = K2 + 0.5 * K4
    if Martingale_Correction:
        K0_star = np.empty(N_paths, dtype=np.float64)

    # Create a numpy random Generator with a seed
    rng = np.random.default_rng(seed=seed)

    # Use the Generator as the seed parameter
    dim = 3 * (T_steps + 1)  # Z1,Z2,U per time step
    sampler = qmc.Sobol(d=dim, scramble=True, rng=rng)
    U_mat = sampler.random(n=N_paths)  # (N_paths, dim)

    split = T_steps + 1
    Z1 = norm.ppf(U_mat[:, :split]).T
    Z2 = norm.ppf(U_mat[:, split : 2 * split]).T
    U = U_mat[:, 2 * split :].T
    del dim, sampler, U_mat, split

    # rng = np.random.default_rng(seed)
    # Z1 = rng.standard_normal(size=(T_steps + 1, N_paths))
    # Z2 = rng.standard_normal(size=(T_steps + 1, N_paths))
    # U  = rng.uniform(size=(T_steps + 1, N_paths))

    # Storage
    S = np.zeros((T_steps + 1, N_paths))
    V = np.zeros((T_steps + 1, N_paths))
    S[0] = log(S0)
    V[0] = V0

    for t in trange(1, T_steps + 1, desc="Simulating Paths", disable=not _show_progress, leave=False):
        m = theta + (V[t - 1] - theta) * E
        s2 = (V[t - 1] * sigma**2 * E) / kappa * (1 - E) + (theta * sigma**2) / (2 * kappa) * (1 - E) ** 2
        psi = s2 / m**2

        # Indices for the two QE regimes
        idx = psi <= psiC
        idx_c = ~idx

        # Regime 1: non-central chi‐squared approx
        b = np.sqrt(2 / psi[idx] - 1 + np.sqrt(2 / psi[idx] * (2 / psi[idx] - 1)))
        a = m[idx] / (1 + b**2)
        V[t, idx] = a * (b + Z1[t, idx]) ** 2

        # Regime 2: two-point mixture
        p = (psi[idx_c] - 1) / (psi[idx_c] + 1)
        beta = (1 - p) / m[idx_c]
        draw_uniform = U[t, idx_c]
        zero_mask = draw_uniform <= p
        V[t, idx_c] = np.where(zero_mask, 0.0, np.log((1 - p) / (1 - draw_uniform)) / beta)

        # Stock‐price increment
        if Martingale_Correction:
            # Correction term per Andersen (2008)
            K0_star[idx] = ( # type: ignore
                -(A * b**2 * a) / (1 - 2 * A * a)
                + 0.5 * np.log(1 - 2 * A * a)
                - (K1 + 0.5 * K3) * V[t - 1, idx]
            )
            K0_star[idx_c] = -np.log(p + (beta * (1 - p)) / (beta - A)) - (K1 + 0.5 * K3) * V[t - 1, idx_c] # type: ignore
            drift = K0_star # type: ignore
        else:
            drift = K0

        S[t] = (
            S[t - 1]
            + (r - q) * dt
            + drift
            + K1 * V[t - 1]
            + K2 * V[t]
            + np.sqrt(K3 * V[t - 1] + K4 * V[t]) * Z2[t]
        )

        # Earnings Gap shock
        if shock_step is not None and t == shock_step:
            # — Assumptions for earnings‐announcement jump (structural, not data‐fit) —
            # σ_ε (sigma_eps) is the standard deviation of the consensus‐surprise ε ~ N(0, σ_ε²).
            # β       (beta)     is the “earnings‐elasticity”: % price move per 1% surprise.
            # Therefore jump J = β·ε has σ_J = β·σ_ε.
            # Under Q (risk‐neutral), we center the log‐jump so E[e^J]=1:
            #    μ_J = –½·σ_J²  and  σ_J = β·σ_ε.
            # We draw J_path ~ N(μ_J, σ_J²) and apply in log‐space:
            #    if t == t0:  S[t:] += J_path
            # Calm firm     : sigma_eps = 0.05 - beta = 0.5
            # Base case     : sigma_eps = 0.08 - beta = 1.0
            # Volatile firm : sigma_eps = 0.10 - beta = 1.5
            # sigma_eps = 0.08
            beta = 1.0
            mu_J = -0.5 * (beta * sigma_eps) ** 2
            sigma_J = beta * sigma_eps
            J_path = np.random.normal(mu_J, sigma_J, size=N_paths)
            
            #  plot the histogram of the jump distribution
            if _plot:
                mu_J_val = (1 - exp(J_path)).mean()
                sigma_J_val = (1 - exp(J_path)).std()
                # sns.histplot((1 - exp(J_path)), bins=50, kde=True, stat="density", color="skyblue")
                # plt.gcf().set_size_inches(7, 4)
                # plt.title(
                #     r"Jump distribution ($\sigma_\epsilon$={:.4f}, $\beta$={:.2f})".format(sigma_eps, beta)
                #     + r" - $X \sim \mathrm{Lognormal}$"
                # )
                # plt.xlabel("Jump size")
                # plt.ylabel("Density")
                # plt.legend([f"Mean={mu_J_val:.4f}, Std={sigma_J_val:.4f}"])
                # plt.grid()
                # plt.show()

            S[t] += J_path
            del beta, mu_J, sigma_J, J_path

    # Exponentiate and transpose to shape (N_paths, T_steps+1)
    S = exp(S).T
    V = V.T

    # Handle any NaNs by replacing entire paths with the analytical mean
    good = ~np.isnan(S).any(axis=1)
    if not np.all(good):
        n_bad = np.sum(~good)
        print(f"Warning: {n_bad} paths contained NaNs; substituting with E[S_t], E[V_t]")
        t_grid = np.arange(T_steps + 1)
        # Analytical means
        S_mean = S0 * np.exp((r - q) * dt * t_grid)
        V_mean = V0 * np.exp(-kappa * dt * t_grid) + theta * (1 - np.exp(-kappa * dt * t_grid))
        S[~good, :] = S_mean
        V[~good, :] = V_mean

    return S, V


@jit(fastmath=True, cache=True, nopython=True)
def __get_european_option_price_Heston_COS(
    S0,
    K,
    T,
    r,
    # Heston Model Paramters
    kappa,  # Speed of the mean reversion
    theta,  # Long term mean
    rho,  # correlation between 2 random variables
    zeta,  # Volatility of volatility
    v0,  # Initial volatility
    opt_type,
    N=64,  # 1024
    z=24,
):
    def heston_char(u):
        t0 = 0.0
        q = 0.0
        m = log(S0) + (r - q) * (T - t0)
        D = sqrt((rho * zeta * 1j * u - kappa) ** 2 + zeta**2 * (1j * u + u**2))
        C = (kappa - rho * zeta * 1j * u - D) / (kappa - rho * zeta * 1j * u + D)
        beta = ((kappa - rho * zeta * 1j * u - D) * (1 - exp(-D * (T - t0)))) / (
            zeta**2 * (1 - C * exp(-D * (T - t0)))
        )
        alpha = ((kappa * theta) / (zeta**2)) * (
            (kappa - rho * zeta * 1j * u - D) * (T - t0) - 2 * log((1 - C * exp(-D * (T - t0)) / (1 - C)))
        )
        return exp(1j * u * m + alpha + beta * v0)

    # # Parameters for the Function to make sure the approximations are correct.
    c1 = log(S0) + r * T - 0.5 * theta * T
    c2 = (
        theta
        / (8 * kappa**3)
        * (
            -(zeta**2) * exp(-2 * kappa * T)
            + 4 * zeta * exp(-kappa * T) * (zeta - 2 * kappa * rho)
            + 2 * kappa * T * (4 * kappa**2 + zeta**2 - 4 * kappa * zeta * rho)
            + zeta * (8 * kappa * rho - 3 * zeta)
        )
    )
    a = c1 - z * sqrt(abs(c2))
    b = c1 + z * sqrt(abs(c2))

    def h(n):
        return (n * pi) / (b - a)

    def g_n(n):
        return (exp(a) - (K / h(n)) * sin(h(n) * (a - log(K))) - K * cos(h(n) * (a - log(K)))) / (1 + h(n) ** 2)

    g0 = K * (log(K) - a - 1) + exp(a)

    F = g0
    for n in range(1, N + 1):
        h_n = h(n)
        F += 2 * heston_char(h_n) * exp(-1j * a * h_n) * g_n(n)

    F = exp(-r * T) / (b - a) * np.real(F)
    F = F if opt_type == "p" else F + S0 - K * exp(-r * T)
    return F if F > 0 else 0


def get_European_Option_BS_Greeks(S, K, T, sigma, r, q=0.0, type="call"):
    """
    Vectorized BS Greeks for European options.

    Parameters
    ----------
    S     : array_like    Spot prices
    K     : array_like    Strike prices
    T     : array_like    Time to maturities (years)
    sigma : array_like    Implied volatilities
    r     : float         Risk-free rate (annual)
    q     : float, opt    Dividend yield (annual), default 0
    flag  : {'call','put'}

    Returns
    -------
    Delta, Gamma, Vega : ndarray
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    sigma = np.asarray(sigma)
    sqrtT = np.sqrt(T)

    # 1) compute d1
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)

    # 2) PDF and CDF of standard normal at d1
    pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    cdf_d1 = 0.5 * (1 + erf(d1 / np.sqrt(2)))

    # 3) Greeks
    # Delta
    if type.lower().startswith("c"):
        Delta = np.exp(-q * T) * cdf_d1
    else:
        Delta = np.exp(-q * T) * (cdf_d1 - 1)

    # Gamma
    Gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * sqrtT)

    # Vega (per unit vol)
    Vega = S * np.exp(-q * T) * pdf_d1 * sqrtT

    return Delta, Gamma, Vega


@njit(fastmath=True, cache=False)
def __bs_option_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Calculate Black-Scholes price for a European option with continuous dividend yield.

    Parameters:
    S: float - Spot price of underlying asset
    K: float - Strike price
    T: float - Time to maturity (in years)
    r: float - Risk-free interest rate (annualized)
    sigma: float - Volatility (annualized)
    option_type: str - 'call' or 'put'
    q: float - Continuous dividend yield (annualized)

    Returns:
    float: Option price
    """

    def norm_cdf_approx(x):
        """
        Fast approximation of the normal CDF (Φ(x)) using the Abramowitz and Stegun method.
        Compatible with numba njit.

        Parameters:
        x: float or numpy.ndarray - Input value(s)

        Returns:
        float or numpy.ndarray - Approximated normal CDF value(s)
        """
        # Constants for the approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # Save the sign of x
        sign = np.sign(x)
        x = np.abs(x)

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

        # Adjust based on sign
        return 0.5 * (1.0 + sign * y)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        return S * np.exp(-q * T) * norm_cdf_approx(d1) - K * np.exp(-r * T) * norm_cdf_approx(d2)
    else:
        return K * np.exp(-r * T) * norm_cdf_approx(-d2) - S * np.exp(-q * T) * norm_cdf_approx(-d1)


def __get_european_option_value_Heston_FFT(S0, K, TTM, v0, r, q, kappa, theta, volvol, rho, flag="Call"):
    """
    Price a single European vanilla option under the Heston stochastic volatility model.

    This helper uses QuantLib to build a Heston process and an AnalyticHestonEngine
    to compute the net present value (NPV) of a European call or put.

    Parameters
    ----------
    S0 : float
        Spot price of the underlying asset.
    K : float
        Option strike price.
    TTM : float
        Time to maturity in days.
    v0 : float
        Initial variance (volatility squared).
    r : float
        Annualized risk-free interest rate.
    q : float
        Annualized continuous dividend yield.
    kappa : float
        Mean-reversion speed of variance.
    theta : float
        Long-run variance level.
    volvol : float
        Volatility of volatility (vol-of-vol).
    rho : float
        Correlation between asset and variance Brownian motions.
    flag : {'Call','Put'}, optional
        Option type; default is 'Call'.

    Returns
    -------
    float or None
        The option price (NPV). Returns None if the Heston model cannot be constructed.
    """
    today = ql.Settings.instance().evaluationDate = ql.Date().todaysDate()
    day_counter = ql.Actual365Fixed()
    riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_counter))
    dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_counter))

    # 2) split TTM into integer days + fractional day for time-of-day
    full_days = int(floor(TTM))
    frac_days = TTM - full_days

    # 2a) advance calendar by integer days
    base_date = today + ql.Period(full_days, ql.Days)

    # 2b) convert fractional day to h/m/s
    total_seconds = int(round(frac_days * 24 * 3600))
    hour = total_seconds // 3600
    minute = (total_seconds % 3600) // 60
    second = total_seconds % 60

    # 2c) build the high-resolution maturity date
    maturityDate = ql.Date(
        base_date.dayOfMonth(),
        base_date.month(),
        base_date.year(),
        hour,
        minute,
        second,
    )

    eps = np.finfo(float).eps
    relative_h = eps**0.2  # 1/5 power
    dS = max(S0 * relative_h, 1e-8)
    # dS = S0*0.001
    # dv = max( (sqrt(v0) * 2e-2), 2e-5)  # Ensure dv is at least 1e-6 to avoid division by zero

    hestonProcess = ql.HestonProcess(
        riskFreeTS,
        dividendTS,
        ql.QuoteHandle(ql.SimpleQuote(S0)),
        v0,
        kappa,
        theta,
        volvol,
        rho,
    )
    hestonProcess_S_up = ql.HestonProcess(
        riskFreeTS,
        dividendTS,
        ql.QuoteHandle(ql.SimpleQuote(S0 + dS)),
        v0,
        kappa,
        theta,
        volvol,
        rho,
    )
    hestonProcess_S_up2 = ql.HestonProcess(
        riskFreeTS,
        dividendTS,
        ql.QuoteHandle(ql.SimpleQuote(S0 + 2 * dS)),
        v0,
        kappa,
        theta,
        volvol,
        rho,
    )
    hestonProcess_S_do = ql.HestonProcess(
        riskFreeTS,
        dividendTS,
        ql.QuoteHandle(ql.SimpleQuote(S0 - dS)),
        v0,
        kappa,
        theta,
        volvol,
        rho,
    )
    hestonProcess_S_do2 = ql.HestonProcess(
        riskFreeTS,
        dividendTS,
        ql.QuoteHandle(ql.SimpleQuote(S0 - 2 * dS)),
        v0,
        kappa,
        theta,
        volvol,
        rho,
    )

    # v0_up , v0_do = (sqrt(v0) + dv)**2, (sqrt(v0) - dv)**2
    # theta_up, theta_do = (sqrt(theta) + dv)**2, (sqrt(theta) - dv)**2
    # hestonProcess_v_up  = ql.HestonProcess(riskFreeTS, dividendTS, ql.QuoteHandle(ql.SimpleQuote(S0)), v0_up, kappa, theta_up, volvol, rho)
    # hestonProcess_v_do  = ql.HestonProcess(riskFreeTS, dividendTS, ql.QuoteHandle(ql.SimpleQuote(S0)), v0_do, kappa, theta_do, volvol, rho)

    def get_NPV(hestonProcess) -> float:
        try:
            hestonModel = ql.HestonModel(hestonProcess)
        except Exception:
            print(f"Inputed params: v0={v0}, kappa={kappa}, theta={theta}, volvol={volvol}, rho={rho}")
            # return None

        # --- Step 1: Define Option Parameters ---
        optionType = ql.Option.Call if flag == "Call" else ql.Option.Put

        # --- Step 2: Construct the Option Instrument ---
        payoff = ql.PlainVanillaPayoff(optionType, float(K))
        exercise = ql.EuropeanExercise(maturityDate)
        europeanOption = ql.VanillaOption(payoff, exercise)

        # --- Step 3: Set Up the Pricing Engine ---
        # Increase accuracy by using a more precise integration method
        # Try multiple tolerances, reducing by an order of magnitude if it fails
        tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
        for tol in tolerances:
            try:
                engine = ql.AnalyticHestonEngine(hestonModel, tol, 10_000)  # type: ignore
                europeanOption.setPricingEngine(engine)
                # --- Step 4: Compute the Option Price ---
                return europeanOption.NPV()
            except Exception as e:
                if tol == tolerances[-1]:
                    # print(f"Failed to set AnalyticHestonEngine with all tolerances ({TTM=}). Last error:\n {e}")
                    # revert to a BS approximation where v0 is sigma
                    bs_value = __bs_option_price(
                        S=S0,
                        K=K,
                        T=TTM / 365,
                        r=r,
                        sigma=np.sqrt(v0),
                        option_type=flag,
                        q=q,
                    )
                    return bs_value
                continue

        return 0.0

    V = get_NPV(hestonProcess)
    V_up = get_NPV(hestonProcess_S_up)
    V_up2 = get_NPV(hestonProcess_S_up2)
    V_do = get_NPV(hestonProcess_S_do)
    V_do2 = get_NPV(hestonProcess_S_do2)

    # V_sigma_up  = get_NPV(hestonProcess_v_up)
    # V_sigma_do  = get_NPV(hestonProcess_v_do)

    # Delta: five‐point stencil
    Delta = (-V_up2 + 8 * V_up - 8 * V_do + V_do2) / (12 * dS)

    # Gamma: five‐point stencil
    Gamma = (-V_up2 + 16 * V_up - 30 * V + 16 * V_do - V_do2) / (12 * dS**2)

    # Vega: three‐point central difference in volatility
    Vega = 0  # (1/100) * (V_sigma_up - V_sigma_do) / (2 * dv)

    # when TTM is <=0 then V is the intrinsic value and the greeks are zero
    # use np.where to return V0 with the intrinsic value if TTM <= 0
    V = np.where(TTM <= 0, np.maximum(S0 - K, 0), V)
    Delta = np.where(TTM <= 0, np.where(S0 > K, 1.0, 0.0), Delta)
    Gamma = np.where(TTM <= 0, 0.0, Gamma)
    Vega = np.where(TTM <= 0, 0.0, Vega)

    return V, Delta, Gamma, Vega


def european_Option_Valuation_Heston_Heston_Greeks(
    S_array,
    K_array,
    TTM_array,
    v0_array,
    r,
    q,
    kappa,
    theta,
    volvol,
    rho,
    type="Call",
):
    """
    Compute the European option price and its sensitivities (Greeks) under the Heston model.

    This function evaluates a European option (typically a call or put) using the Heston stochastic
    volatility model. It employs QuantLib to construct a Heston process and an analytic pricing engine
    for the option instrument. The Greeks (Delta, Gamma, Vega) are calculated using finite difference
    approximations via a five-point stencil for Delta and Gamma and a three-point stencil for Vega.
    The calculation is performed in parallel across all provided input cases.

    Parameters
    ----------
    S_array : array_like
        1D array of underlying asset spot prices for each option instance.
    K_array : array_like
        1D array of option strike prices corresponding to each spot price.
    TTM_array : array_like
        1D array of times to maturity (in days) for each option instance.
    v0_array : array_like
        1D array of initial variances (volatility squared) for each option instance.
    r : float
        Annualized risk-free interest rate.
    q : float
        Annualized continuous dividend yield.
    kappa : float
        Mean-reversion speed of the variance process in the Heston model.
    theta : float
        Long-run variance (mean reversion level) in the Heston model.
    volvol : float
        Volatility of volatility (vol-of-vol) parameter in the Heston model.
    rho : float
        Correlation coefficient between the Brownian motions driving the asset price and its variance.
    type : str, optional
        Option type; typically 'Call' or 'Put'. Default is 'Call'.

    Returns
    -------
    tuple of ndarray
        A tuple containing four numpy arrays:
            - Option Price Array: The computed option prices for each input case.
            - Delta Array: The first-order sensitivity of the option price with respect to the underlying asset price.
            - Gamma Array: The second-order sensitivity with respect to the underlying asset price.
            - Vega Array: The sensitivity of the option price with respect to volatility.

    Notes
    -----
    - The function internally defines a helper function that constructs the Heston process
      and uses QuantLib's AnalyticHestonEngine to price the option.
    - Finite difference approximations (using a five-point stencil for Delta and Gamma, and a
      three-point stencil for Vega) are used to compute the Greeks.
    - The computation for each option instance is performed in parallel using joblib's Parallel processing.
    - The time to maturity is converted from days to the appropriate time measure within QuantLib.
    - In cases where QuantLib is unable to construct the Heston model for a given set of parameters,
      the function prints the parameters and returns None for that instance.
    """
    # Choose the pricing function based on METHOD
    # METHOD='FFT'  # 'COS' or 'FFT'
    # pricing_func = __get_european_option_value_Heston_FFT if METHOD == 'FFT' else __get_european_option_value_Heston_COS
    # __get_european_option_value_Heston_stochvol
    pricing_func = __get_european_option_value_Heston_FFT
    
    # To avoid numerical errors since dt is calculated with division, last step might be ver close to maturity but not exactly
    TTM_array[TTM_array < 0.02 ] = 0.0 # equal to half an hour. As 1/48 (ttm is measured in days here)
    
    # run in parallel over all input rows
    results = Parallel(
        n_jobs=-1,
        backend="loky",  # still uses loky but starts threads
        prefer="threads",
        batch_size=128,
    )(
        delayed(pricing_func)(S0, K, TTM, v0, r, q, kappa, theta, volvol, rho)
        for S0, K, TTM, v0 in zip(S_array, K_array, TTM_array, v0_array)
    )
    # unzip & cast to arrays
    _V0_array, _Delta_array, _Gamma_array, _Vega_array = map(np.array, zip(*results))

    return _V0_array, _Delta_array, _Gamma_array, _Vega_array


def norm_test(pnl_distribution):
    """
    Visual and statistical normality diagnostics for a P&L (profit and loss) distribution.

    This function provides a comprehensive assessment of whether a given P&L distribution is consistent with normality.
    It generates diagnostic plots and performs standard statistical tests:

    1. **Histogram, KDE, and Normal Fit**: Plots a histogram of the data, overlays a kernel density estimate (KDE), and fits a normal distribution to the data, displaying its PDF and parameters (mean and standard deviation).
    2. **Q-Q Plot**: Displays a quantile-quantile plot comparing the empirical quantiles of the data to those of a standard normal distribution, visually highlighting deviations from normality.
    3. **Statistical Tests**:
        - D'Agostino's K² test: Tests for normality based on skewness and kurtosis.
        - Kolmogorov-Smirnov test: Compares the standardized data to a standard normal distribution.

    Parameters
    ----------
    pnl_distribution : array-like
        Array or sequence of P&L values to be tested for normality.

    Returns
    -------
    None
        Displays plots and prints test statistics to the console.

    Notes
    -----
    - Uses seaborn and matplotlib for plotting, and scipy.stats for statistical tests.
    - Prints the test statistics and p-values for both D'Agostino's K² and Kolmogorov–Smirnov tests.
    - Intended for exploratory data analysis of simulation or empirical P&L results.
    """
    plt.figure(figsize=(12, 5))

    # 1) Histogram + KDE + fitted normal PDF
    plt.subplot(1, 2, 1)
    sns.histplot(
        pnl_distribution,
        stat="density",
        bins=30,
        color="skyblue",
        edgecolor="white",
        label="Histogram",
    )
    sns.kdeplot(pnl_distribution, color="darkblue", lw=2, label="KDE")
    mu, std = norm.fit(pnl_distribution)
    x = np.linspace(pnl_distribution.min(), pnl_distribution.max(), 200)
    plt.plot(
        x,
        norm.pdf(x, mu, std),
        "r--",
        label=f"Normal fit (mean={mu:.2f}, std={std:.2f})",
    )
    plt.title("P&L Distribution with Normal Fit")
    plt.xlabel("P&L")
    plt.ylabel("Density")
    plt.legend()

    # 2) Q–Q plot against a normal
    plt.subplot(1, 2, 2)
    probplot(pnl_distribution, dist="norm", plot=plt)
    plt.title("Normal Q-Q Plot of P&L")
    plt.tight_layout()
    plt.show()

    # 3) Statistical normality tests
    stat_k2, p_k2 = normaltest(pnl_distribution)
    stat_ks, p_ks = kstest((pnl_distribution - mu) / std, "norm")
    print(f"D'Agostino K² test:     stat={stat_k2:.3f}, p={p_k2:.3f}")
    print(f"Kolmogorov-Smirnov test: stat={stat_ks:.3f}, p={p_ks:.3f}")


def __stat_var(arr: np.ndarray, q=0.05):
    """
    Calculate Value at Risk while preserving the original sign.
    If the tail contains negative values (gains), VaR will be negative.

    Bootstrapped's internal engine calls this twice:
      • once with the original 1-D sample  (shape = (N,))
      • many times with 2-D bootstrap batches (shape = (B, N))
    It must therefore handle both ranks.
    """
    p = 100 * q  # left-tail percentile (e.g. 5 for 95% VaR)
    if arr.ndim == 1:
        return np.array([np.percentile(arr, p)])  # shape (1,)
    else:
        return np.percentile(arr, p, axis=1)  # shape (B,)
    
@njit(parallel=True, fastmath=True)
def __stat_var_fast(arr, q=0.05):
    """5 % one-sided VaR using O(N) partition."""
    k = int(q * arr.shape[-1])
    if arr.ndim == 1:
        return np.array([np.partition(arr, k)[k]], dtype=arr.dtype)
    out = np.empty(arr.shape[0], dtype=arr.dtype)
    for i in prange(arr.shape[0]):
        out[i] = np.partition(arr[i], k)[k]
    return out


def __stat_cvar(arr: np.ndarray, q=0.05):
    """
    Calculate Conditional Value at Risk while preserving the original sign.
    If the tail contains negative values (gains), CVaR will be negative.

    Handles both 1-D input (original sample) and 2-D batches coming
    from bootstrapped's internal engine.
    """
    p = 100 * q  # left-tail percentile
    if arr.ndim == 1:  # --- single vector
        threshold = np.percentile(arr, p)
        tail_values = arr[arr <= threshold]  # preserve original sign
        return np.array([tail_values.mean()])  # shape (1,)
    else:  # --- bootstrap matrix  (B × N)
        result = np.zeros(arr.shape[0])
        for i in range(arr.shape[0]):
            threshold = np.percentile(arr[i], p)
            tail_values = arr[i, arr[i] <= threshold]
            result[i] = tail_values.mean()
        return result  # shape (B,)


####################################################################################################
def compute_variance_swap_strike_analytical(v0: float, kappa: float, theta: float, T: float) -> float:
    """
    Continuous-sampling fair strike K_var under Heston.

    K = theta + (v0 - theta)/(kappa*T)*(1 - exp(-kappa*T))
    """
    return theta + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / (kappa * T)


def compute_variance_swap_strike_analytical_discrete(
    v0: float, kappa: float, theta: float, T: float, n_steps: int
) -> float:
    """
    Compute the fair strike of a variance swap under the Heston model with discrete sampling.

    ### Mathematical Explanation

    The fair strike of a variance swap, denoted as K_var, is the level at which the expected payoff of the swap is zero.
    For a variance swap with **discrete sampling**, the strike is computed as:

        K_var = term_endpoints + term_path

    where:

    - **term_endpoints** accounts for the contributions from the initial and final variance values:

        term_endpoints = theta * (1 - 1 / (2 * N)) + v0 / (2 * N)

    - **term_path** accounts for the contributions from the variance path over time:

        term_path = (v0 - theta) * [
            (exp(-kappa * dt) - exp(-kappa * T)) / (N * (1 - exp(-kappa * dt)))
            + exp(-kappa * T) / (2 * N)
        ]

    Here:
    - v0: Initial variance.
    - theta: Long-term mean variance (mean reversion level).
    - kappa: Speed of mean reversion of the variance process.
    - T: Total time to maturity of the variance swap.
    - N: Number of discrete sampling intervals.
    - dt = T / N: Time step between sampling points.

    ### Parameters
    - `v0` (float): Initial variance.
    - `kappa` (float): Speed of mean reversion of the variance process.
    - `theta` (float): Long-term mean variance.
    - `T` (float): Time to maturity of the variance swap.
    - `n_steps` (int): Number of discrete sampling intervals.

    ### Returns
    - `float`: The fair strike K_var of the variance swap.

    ### Notes
    - This formula assumes a Heston stochastic volatility model with discrete sampling.
    - The strike is derived by integrating the variance process over time and normalizing by the total time horizon.

    """
    dt = T / n_steps
    N = n_steps
    # geometric sum S not needed explicitly; we build terms directly:
    term_endpoints = theta * (1 - 1 / (2 * N)) + v0 / (2 * N)
    term_path = (v0 - theta) * (
        (np.exp(-kappa * dt) - np.exp(-kappa * T)) / (N * (1 - np.exp(-kappa * dt)))
        + np.exp(-kappa * T) / (2 * N)
    )
    return term_endpoints + term_path


def get_variance_swap_notional(
    S0: float,
    K: float,
    TTM: float,  # days to expiry of the option
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    volvol: float,
    rho: float,
    flag: str = "Call",
) -> float:
    """
    Compute the variance-swap notional N that hedges the Heston-FFt European option
    exposure to instantaneous variance v0.

    Returns
    -------
    N : float
        Variance-swap notional.
    """
    # 1) Step for finite-difference in v0
    eps = np.finfo(float).eps
    relative_h = eps**0.2
    dv = max(v0 * relative_h, 1e-8)

    # 2) Price the option at v0 + dv and v0 - dv
    C_up, _, _, _ = __get_european_option_value_Heston_FFT(
        S0, K, TTM, v0 + dv, r, q, kappa, theta, volvol, rho, flag
    )
    C_do, _, _, _ = __get_european_option_value_Heston_FFT(
        S0, K, TTM, v0 - dv, r, q, kappa, theta, volvol, rho, flag
    )

    # 3) Approximate ∂C/∂v0
    dC_dv = (C_up - C_do) / (2 * dv)

    # 4) Build denominator ∂V_vs/∂v0 for a variance swap of matching tenor
    tau = TTM / 365.0
    # ∂V_vs/∂v0  =  e^{-r τ} / T  * (1 - e^{-κ τ}) / κ,  here T=τ
    denom = np.exp(-r * tau) / tau * (1.0 - np.exp(-kappa * tau)) / kappa

    # 5) Hedge ratio
    N = dC_dv / denom
    return N


def price_variance_swap(
    S_paths: np.ndarray,  # underlying prices, shape (N, M)
    v_paths: np.ndarray,  # instantaneous variances, shape (N, M)
    tidx: int,  # current observation index
    ttm: float,  # remaining time (days)
    K_vswap: float,  # variance-swap strike (σ²)
    swap_duration_yrs: float,  # contract tenor T (years)
    notional: float,
    r: float,
    kappa: float,
    theta: float,
) -> np.ndarray:
    """
    Mark-to-market value of a variance swap at observation index `tidx`,
    consistent with `price_variance_swap_terminal` at maturity.
    """
    # ------------------------------------------------------------------
    # 1. Realised variance up to t  ─────────────────────────────────────
    # ------------------------------------------------------------------
    if tidx == 0:
        realised_integral = np.zeros_like(S_paths[:, 0])  # shape (N,)
    else:
        log_rets = np.diff(np.log(S_paths[:, : tidx + 1]), axis=1)  # shape (N, tidx)
        realised_integral = np.sum(log_rets**2, axis=1)  # ∑ (Δln S)²

    # ------------------------------------------------------------------
    # 2. Time remaining and expected future variance (Heston) ───────────
    # ------------------------------------------------------------------
    tau = ttm / 365.0  # years left
    if tau > 0.0:
        F_t = theta * tau + (v_paths[:, tidx] - theta) / kappa * (1.0 - np.exp(-kappa * tau))
        discount = np.exp(-r * tau)
    else:  # maturity: no future variance, no discount
        F_t = 0.0
        discount = 1.0

    # ------------------------------------------------------------------
    # 3. Present value per path  ────────────────────────────────────────
    # ------------------------------------------------------------------
    V_t: np.ndarray = discount * ((realised_integral + F_t) / swap_duration_yrs - K_vswap)
    return notional * V_t


def compute_hedge_notional_bs(
    S0: float,
    K: float,
    ttm: int,  # time to maturity in trading days
    t_steps: int,  # N
    r: float,  # risk-free rate
    q: float,  # dividend yield
    v0: float,  # initial variance
    theta: float,  # long-term variance (unused here)
    kappa: float,  # mean reversion speed
    volvol: float,  # volatility of variance (unused here)
    rho: float,  # correlation (unused here)
) -> float:
    """
    BS‐based approximation of the variance‐swap notional N* needed to hedge a European option’s variance‐exposure.

    Returns
    -------
    N_star : float
        The notional of the long variance‐swap position.
    """
    # 1. Convert days to years (assuming 365 trading days)
    T = ttm / 365.0
    N = t_steps + 1

    # 2. Black‐Scholes implied vol = sqrt(v0)
    sigma_bs = np.sqrt(v0)
    sqrtT = np.sqrt(T)

    # 3. Compute d1 and BS‐vega
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma_bs**2) * T) / (sigma_bs * sqrtT)
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    vega_bs = S0 * np.exp(-q * T) * sqrtT * phi

    # 4. Chain‐rule: dC/dv0 ≈ vega_bs * dσ/dv0 = vega_bs / (2√v0)
    dC_dv0 = vega_bs / (2 * sigma_bs)

    # 5. Variance‐swap sensitivity dK_var/dv0
    dt = T / N
    term1 = 1 / (2 * N)
    term2 = (np.exp(-kappa * dt) - np.exp(-kappa * T)) / (N * (1 - np.exp(-kappa * dt)))
    term3 = np.exp(-kappa * T) / (2 * N)
    dKvar_dv0 = term1 + term2 + term3

    # 6. Hedge ratio
    N_star = dC_dv0 / dKvar_dv0
    return N_star


def delta_vswap_hedging_simulation(
    S0: float,
    K: float,
    ttm: int,  # liability option TTM   (days)
    hedging_steps: int,
    r: float,
    q: float,
    v0: float,
    theta: float,
    kappa: float,
    volvol: float,
    rho: float,
    DELTA_HEDGE_PERCT: float = 1.0,  # % of Δ exposure to offset
    Underlying_SPREAD: float = 0.0001,  # proportional cost
    VEGA_HEDGE_RATIO: float = 1.0,  # % of Vega exposure to offset
    Vswap_TC: float = 0.0010,  # numerical % of Vswap strike
    shock_step: int | None = None,
    sigma_eps:float = 0.08,
    tdpy: int = 365,  # trading-days / year
    Option_SPREAD: float = 0.0050,
    MC_paths: int = 1_024,
    ci_level: float = 0.99,
    _plot: bool = False,
    _print: bool = False,
    seed: Optional[int] = None,
) -> Tuple[
    np.ndarray,  # final PnL vector
    float,
    Tuple[float, float],  # mean & CI
    float,  # std
    float,
    Tuple[float, float],  # VaR95 & CI
    float,
    Tuple[float, float],  # CVaR95 & CI
    np.ndarray,  # cumulative Δ+Γ costs per path
    np.ndarray,  # variance swap final PnL
]:
    """
    Monte Carlo simulation of a Delta-Hedging strategy enhanced with a Variance Swap
    for Vega risk management under the Heston stochastic volatility model.

    This function simulates the P&L of hedging a short position in a European call option
    using a combination of delta hedging with the underlying asset and vega hedging with
    a variance swap. The simulation accounts for discrete hedging intervals and transaction
    costs for both the underlying and the variance swap.

    Parameters
    ----------
    S0 : float
        Initial spot price of the underlying asset.
    K : float
        Strike price of the option being hedged.
    ttm : int
        Time to maturity of the option in days.
    hedging_steps : int
        Number of discrete hedging adjustments over the option's life.
    r : float
        Annual risk-free interest rate (continuously compounded).
    q : float
        Annual dividend yield of the underlying asset.
    v0 : float
        Initial variance in the Heston model.
    theta : float
        Long-run variance (mean reversion level) in the Heston model.
    kappa : float
        Mean-reversion speed parameter in the Heston model.
    volvol : float
        Volatility of variance (vol-of-vol) in the Heston model.
    rho : float
        Correlation between asset returns and variance changes.
    DELTA_HEDGE_PERCT : float, optional
        Fraction of delta exposure to hedge at each step (1.0 = full hedge).
    Underlying_SPREAD : float, optional
        Proportional transaction cost for trading the underlying (e.g., 0.0001 = 1bp).
    VEGA_HEDGE_RATIO : float, optional
        Scaling factor for the variance swap position relative to optimal vega hedge.
    Vswap_TC : float, optional
        Transaction cost for the variance swap as markup on strike.
    sigma_eps : float, optional
        Standard deviation of the jump size of the earnings gap surprise
    tdpy : int, optional
        Trading days per year (calendar convention).
    Option_SPREAD : float, optional
        Proportional transaction cost for options (used in P&L reporting).
    MC_paths : int, optional
        Number of Monte Carlo simulation paths.
    ci_level : float, optional
        Confidence level for statistical intervals (e.g., 0.99 for 99% confidence).
    _plot : bool, optional
        If True, generates diagnostic plots of the simulation results.
    _print : bool, optional
        If True, prints detailed debugging information during simulation.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    final_pnl : ndarray
        Array of final P&L values for each Monte Carlo path.
    pnl_mean : float
        Mean P&L across all simulated paths.
    μ_ci : tuple
        Confidence interval (lower, upper) for mean P&L.
    pnl_std : float
        Standard deviation of the P&L distribution.
    VaR95 : float
        Value at Risk at the 95% level.
    VaR_ci : tuple
        Confidence interval for VaR95.
    CVaR95 : float
        Conditional Value at Risk (Expected Shortfall) at 95% level.
    CVaR_ci : tuple
        Confidence interval for CVaR95.
    cum_total : ndarray
        Cumulative transaction costs for each path.
    Vswap_value : ndarray
        Array of final P&L values for the Variance Swap for each Monte Carlo path.

    Notes
    -----
    - The variance swap is sized using a Black-Scholes approximation that targets
      offsetting the option's vega exposure.
    - The delta hedge is dynamically adjusted at each hedging step.
    - All confidence intervals are computed via bootstrapping.
    - Transaction costs are modeled as proportional to trade value.
    - The function uses the Heston Quadratic-Exponential (QE) scheme for path simulation.

    Examples
    --------
    >>> results = delta_vswap_hedging_simulation(
    ...     S0=100, K=100, ttm=30, hedging_steps=10,
    ...     r=0.02, q=0.01, v0=0.04, theta=0.04, kappa=1.0,
    ...     volvol=0.2, rho=-0.7, _plot=True)
    >>> print(f"Mean P&L: {results[1]:.4f}, Std Dev: {results[3]:.4f}")
    """
    # Cast S0, K ttms to floats
    S0, K, ttm = float(S0), float(K), int(ttm)
    K_array = np.full(fill_value=K, shape=MC_paths, dtype=np.float32)
    ttm_array = np.full(fill_value=ttm, shape=MC_paths, dtype=np.float32)
    D_net_exposure = np.zeros((MC_paths, hedging_steps + 2), dtype=np.float16)

    # ------------------------------------------------------------------ setup
    t_steps = hedging_steps + 1
    dt = ttm / (t_steps) / tdpy
    if _print:
        print(f"hedging_steps: {hedging_steps}, t_steps: {t_steps}, dt: {dt:.4f}, tdpy: {tdpy}")

    # Simulate paths
    S_paths, v_paths = QESim(
        S0=S0,
        V0=v0,
        rho=rho,
        theta=theta,
        sigma=volvol,
        kappa=kappa,
        r=r,
        q=q,
        dt=dt,
        T_steps=t_steps,
        N_paths=MC_paths,
        Martingale_Correction=True,
        shock_step=shock_step,
        sigma_eps=sigma_eps,
        _show_progress=False,
        _plot=True,
        seed=seed or 1234,
    )
    # 2.5) (If plot) Plot the paths
    if _plot:
        # Create a 1x2 plot
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # Define time steps for x-axis
        time_steps = np.arange(S_paths.shape[1])

        # --- Plot 1: Stock Price Paths ---
        axs[0].plot(time_steps, S_paths.mean(axis=0), label="Mean Stock Price", color="tab:blue")
        axs[0].plot(
            time_steps,
            S_paths.min(axis=0),
            label="Min Stock Price",
            color="tab:orange",
            linewidth=0.9,
        )
        axs[0].plot(
            time_steps,
            S_paths.max(axis=0),
            label="Max Stock Price",
            color="tab:orange",
            linewidth=0.9,
        )
        quantiles_price = np.quantile(S_paths, [0.025, 0.975], axis=0)
        axs[0].fill_between(
            time_steps,
            quantiles_price[0],
            quantiles_price[1],
            alpha=0.3,
            label="95% IQR",
            color="tab:blue",
        )
        axs[0].set_title("Heston Model: Stock Price Paths")
        axs[0].set_xlabel("Time Steps")
        axs[0].set_ylabel("Price")
        axs[0].legend(loc="upper left")
        axs[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # --- Plot 2: Volatility Paths ---
        axs[1].plot(time_steps, v_paths.mean(axis=0), label="Mean Volatility", color="tab:blue")
        axs[1].plot(
            time_steps,
            v_paths.min(axis=0),
            label="Min Volatility",
            color="tab:orange",
            linewidth=0.9,
        )
        axs[1].plot(
            time_steps,
            v_paths.max(axis=0),
            label="Max Volatility",
            color="tab:orange",
            linewidth=0.9,
        )
        quantiles_vol = np.quantile(v_paths, [0.025, 0.975], axis=0)
        axs[1].fill_between(
            time_steps,
            quantiles_vol[0],
            quantiles_vol[1],
            alpha=0.3,
            label="95% IQR",
            color="tab:blue",
        )
        axs[1].set_title("Heston Model: Volatility Paths")
        axs[1].set_xlabel("Time Steps")
        axs[1].set_ylabel("Volatility")
        axs[1].legend(loc="upper left")
        axs[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.show()

    # Convenience handles -----------------------------------------------------
    option_val_greeks = partial(
        european_Option_Valuation_Heston_Heston_Greeks,
        r=r,
        q=q,
        kappa=kappa,
        theta=theta,
        volvol=volvol,
        rho=rho,
        type="Call",
    )
    variance_swap_strike = compute_variance_swap_strike_analytical_discrete(
        v0=v0, kappa=kappa, theta=theta, T=ttm / tdpy, n_steps=t_steps
    )
    variance_swap_strike = variance_swap_strike * (1 + Vswap_TC)  # add transaction cost to strike
    if _print:
        print(f"Variance swap strike: {variance_swap_strike:.4f}")
    vswap_notional = (
        compute_hedge_notional_bs(
            S0=S0,
            K=K,
            ttm=ttm,
            t_steps=t_steps,
            r=r,
            q=q,
            v0=v0,
            theta=theta,
            kappa=kappa,
            volvol=volvol,
            rho=rho,
        )
        * VEGA_HEDGE_RATIO
    )
    vswap_notional = (
        get_variance_swap_notional(
            S0=S0,
            K=K,
            TTM=ttm,
            v0=v0,
            r=r,
            q=q,
            kappa=kappa,
            theta=theta,
            volvol=volvol,
            rho=rho,
            flag="Call",
        )
        * VEGA_HEDGE_RATIO
    )
    print()
    # vswap_hedge_ratio_bs_proxy(S0=S0, K=K, r=r, T=ttm/tdpy, v0=v0, kappa=kappa) * VEGA_HEDGE_RATIO

    # Variance swap price function ----------------------------------------
    get_vswap_price = partial(
        price_variance_swap,
        notional=vswap_notional,
        K_vswap=variance_swap_strike,
        swap_duration_yrs=ttm / tdpy,
        S_paths=S_paths,
        v_paths=v_paths,
        r=r,
        kappa=kappa,
        theta=theta,
    )
    Vswap_value = get_vswap_price(tidx=0, ttm=ttm)
    print(f'Vswap expected cost: {Vswap_value[0]}')
    if _print:
        print(f"Variance swap price: {Vswap_value.mean():.8f} w/ Vswap Notional*: {vswap_notional:.8f}")

    Vswap_value_array = np.zeros(shape=(MC_paths, t_steps + 1), dtype=np.float32)
    Vswap_value_array[:, 0] = Vswap_value

    # Liability option Greeks at t=0
    if _print:
        print(f"** init ttm: {ttm} days")
    V0_L, Δ0_L, Γ0_L, _ = (
        val[0]
        for val in option_val_greeks(
            S_array=np.array([S0]),
            K_array=np.array([K]),
            v0_array=np.array([v0]),
            TTM_array=np.array([ttm]),
        )
    )
    if _print:
        print(f"Liability option value: {V0_L:.4f}, Δ: {Δ0_L:.4f}")

    # Delta_Porfolio to store at each t, the number of delta shares -------------
    Delta_Portfolio = np.zeros(MC_paths, dtype=np.float32)

    Δ_exposure_0 = Δ0_L * DELTA_HEDGE_PERCT
    if _print:
        print(f"Δ_exposure_0: {Δ_exposure_0:.4f} = - (-Δ0_L: {Δ0_L:.4f}")
    Delta_Portfolio.fill(Δ_exposure_0)

    # PnL & cost trackers
    Portfolio_val = np.zeros((MC_paths, t_steps + 1), dtype=np.float64)
    Portfolio_roll_val = np.zeros_like(Portfolio_val, dtype=np.float32)
    Δ_costs = np.zeros_like(Portfolio_val, dtype=np.float32)

    # Initial cashflows -------------------------------------------------------
    if _print:
        print(f"Delta costs: {np.abs(Δ_exposure_0 * S0 * Underlying_SPREAD):.4f}")
    if _print:
        print(f"    Δ_exposure_0: {Δ_exposure_0:.4f}  S0: {S0:.4f}  Underlying_SPREAD: {Underlying_SPREAD:.4f}")
    Δ_costs[:, 0] = np.abs(Δ_exposure_0 * S0 * Underlying_SPREAD)

    Portfolio_val[:, 0] = (
        V0_L  # receive liability option premium
        # + V0_L * Option_SPREAD        # receive the liability option spread (since they are crossing the spread)
        - Delta_Portfolio * S0  # buy underlying
        - Δ_costs[:, 0]  # subtract the Delta costs
        - Vswap_value  # register PnL from the variance swap
    )
    Portfolio_roll_val[:, 0] = (
        Portfolio_val[:, 0]
        - V0_L  # buy back the liability option
        + Delta_Portfolio * S0  # sell underlying
        + Vswap_value  # sell the variance swap
    )
    if _print:
        print(f"Portfolio value at t=0: {Portfolio_val[0, 0]:.4f}")
    if _print:
        print(f"Portfolio rolling value at t=0: {Portfolio_roll_val[0, 0]:.4f}")

    # =================================================================== loop
    tidx = 0
    for tidx in trange(1, t_steps, desc="Δ hedge", leave=False):
        if _print:
            print()
        # ageing of options
        ttm_array -= dt * tdpy  # dt already in years
        # print(f'ttm_array: {ttm_array.mean()}')
        # Portfolio accrual
        Portfolio_val[:, tidx] = Portfolio_val[:, tidx - 1] * np.exp(r * dt)

        # Get the Value and Delta of the Option ----------------------------------------
        V_arr, Δ_arr, _, _ = option_val_greeks(
            S_array=S_paths[:, tidx],
            K_array=K_array,
            TTM_array=ttm_array,
            v0_array=v_paths[:, tidx],
        )
        if _print:
            print(f"V_arr.mean(): {V_arr.mean():.4f}, V_arr.std(): {V_arr.std():.4f}")
        if _print:
            print(f"Δ_arr.mean(): {Δ_arr.mean():.4f}, Δ_arr.std(): {Δ_arr.std():.4f}")

        # Δ re-hedge ----------------------------------------------------------
        if _print:
            print(f"Delta_Portfolio: {Delta_Portfolio.mean():.4f} = {Delta_Portfolio.mean(axis=0)}")
        D_net_exposure[:, tidx] = Δ_arr - Delta_Portfolio
        new_Δ_shares_to_buy = Δ_arr * DELTA_HEDGE_PERCT - Delta_Portfolio
        if _print:
            print(
                f"new_Δ_shares_to_buy: {new_Δ_shares_to_buy.mean():.4f} = Δ_arr: {Δ_arr.mean():.4f} + Δ_Portfolio: {Delta_Portfolio.mean():.4f}"
            )
        # -- Buy new Delta shares to hedge Δ_exposure
        Portfolio_val[:, tidx] += -new_Δ_shares_to_buy * S_paths[:, tidx]
        Δ_costs[:, tidx] = np.abs(new_Δ_shares_to_buy) * S_paths[:, tidx] * Underlying_SPREAD
        Portfolio_val[:, tidx] -= Δ_costs[:, tidx]
        # print(f'**Portfolio_val[:, tidx]: {Portfolio_val[:, tidx].mean():.4f}')

        # -- Update the Delta shares currently held
        if _print:
            print(
                f"Delta_Portfolio: {(Delta_Portfolio + new_Δ_shares_to_buy).mean():.4f} = Delta_Portfolio: {Delta_Portfolio.mean():.4f} + new_Δ_shares_to_buy: {new_Δ_shares_to_buy.mean():.4f}"
            )
        Delta_Portfolio = Delta_Portfolio + new_Δ_shares_to_buy

        # Variance swap value ----------------------------------------
        Vswap_value = get_vswap_price(tidx=tidx, ttm=ttm_array[0])
        Vswap_value_array[:, tidx] = Vswap_value
        if _print:
            print(f"Vswap_value.mean(): {Vswap_value.mean():.4f}, Vswap_value.std(): {Vswap_value.std():.4f}")

        Portfolio_roll_val[:, tidx] = (
            Portfolio_val[:, tidx]
            - V_arr  # sell the Option Portfolio - includes buying back the liability option
            + Delta_Portfolio * S_paths[:, tidx]  # sell the underlying shares
            + Vswap_value  # money received from the swap
        )
        if _print:
            print(f"Portfolio rolling value: {Portfolio_roll_val[:, tidx].mean():.4f}")
        if _print:
            print(f"    Portfolio_val[:, tidx]: {Portfolio_val[:, tidx].mean():.4f}")
        if _print:
            print(f"    - V_arr: {V_arr.mean():.4f}")
        if _print:
            print(f"    + Delta_Portfolio*S_paths[:, tidx]: {(Delta_Portfolio * S_paths[:, tidx]).mean():.4f}")
        if _print:
            print(f"    - Vswap_value: {Vswap_value.mean():.4f}")

    # ================================================================= finish
    tidx += 1
    ttm_array -= dt * tdpy  # dt already in years
    Portfolio_val[:, tidx] = Portfolio_val[:, tidx - 1] * np.exp(r * dt)

    # terminal option values
    VT_arr, ΔT_arr, _, _ = option_val_greeks(
        S_array=S_paths[:, tidx],
        K_array=K_array,
        TTM_array=ttm_array,
        v0_array=v_paths[:, tidx],
    )

    Portfolio_val[:, tidx] += -VT_arr + Delta_Portfolio * S_paths[:, tidx]
    Portfolio_roll_val[:, tidx] = Portfolio_val[:, tidx]

    D_net_exposure[:, tidx] = ΔT_arr - Delta_Portfolio

    # Variance swap value ----------------------------------------
    Vswap_value = get_vswap_price(tidx=tidx, ttm=0)
    Vswap_value_array[:, tidx] = Vswap_value
    if _print:
        print(f"Vswap_value.mean(): {Vswap_value.mean():.4f}, Vswap_value.std(): {Vswap_value.std():.4f}")
    Portfolio_roll_val[:, tidx] += Vswap_value

    # ----------------------------------------------------------------- stats
    cum_Δ_costs = Δ_costs.sum(axis=1)
    cum_total = cum_Δ_costs

    final_pnl = Portfolio_roll_val[:, -1]
    pnl_std = final_pnl.std()

    # Bootstrap CIs -----------------------------------------------------------
    μ_boot_result = bs.bootstrap(final_pnl, stat_func=bs_stats.mean, alpha=1 - ci_level, num_threads=-1, num_iterations=2048, iteration_batch_size=128)
    pnl_mean, μ_ci = (
        μ_boot_result.value,
        (μ_boot_result.lower_bound, μ_boot_result.upper_bound),
    )
    VaR_boot_result = bs.bootstrap(final_pnl, stat_func=__stat_var, alpha=1 - ci_level, num_threads=-1, num_iterations=2048, iteration_batch_size=128)
    VaR95, VaR_ci = (
        VaR_boot_result.value,
        (VaR_boot_result.lower_bound, VaR_boot_result.upper_bound),
    )
    CVaR_boot_result = bs.bootstrap(final_pnl, stat_func=__stat_cvar, alpha=1 - ci_level, num_threads=-1, num_iterations=2048, iteration_batch_size=128)
    CVaR95, CVaR_ci = (
        CVaR_boot_result.value,
        (CVaR_boot_result.lower_bound, CVaR_boot_result.upper_bound),
    )
    Vswap_results = bs.bootstrap(Vswap_value_array[:, tidx], stat_func=bs_stats.mean, alpha=1 - ci_level, num_threads=-1, num_iterations=2048, iteration_batch_size=128)
    Vswap_pnl_mean, Vswap_pnl_ci = (
        Vswap_results.value,
        (Vswap_results.lower_bound, Vswap_results.upper_bound),
    )
    # Optional diagnostics ----------------------------------------------------
    if _plot:
        _time = np.linspace(ttm, 0, t_steps + 1)

        fig, axs = plt.subplots(3, 2, figsize=(14, 12), dpi=300, tight_layout=True)  # Increased to 3 rows

        # ── (1) Rolling PnL ─────────────────────────────────────────────────────────
        _mean = Portfolio_roll_val.mean(axis=0)
        _var95 = np.quantile(Portfolio_roll_val, 0.05, axis=0)
        _cvar95 = np.nanmean(np.where(Portfolio_roll_val < _var95, Portfolio_roll_val, np.nan), axis=0)
        lower_q, upper_q = np.quantile(Portfolio_roll_val, [0.025, 0.975], axis=0)

        axs[0, 0].fill_between(_time, lower_q, upper_q, color="tab:blue", alpha=0.2)
        axs[0, 0].plot(_time, _mean, c="tab:blue", marker=".", lw=1, label="Mean")
        axs[0, 0].plot(_time, _var95, c="tab:green", marker=".", lw=1, label="VaR 95%")
        axs[0, 0].plot(_time, _cvar95, c="tab:orange", marker=".", lw=1, label="CVaR 95%")
        axs[0, 0].set(
            xlabel=r"Days to maturity (d)",
            ylabel=r"PnL Portfolio",
            title=r"PnL Portfolio Simulation",
        )
        axs[0, 0].grid(which="major", alpha=0.5)
        axs[0, 0].invert_xaxis()

        # ── (2) Portfolio value paths ───────────────────────────────────────────────
        axs[1, 0].plot(_time, Portfolio_val.T[:, :1000], lw=0.8, alpha=0.35, color="tab:blue")
        axs[1, 0].invert_xaxis()
        axs[1, 0].set(title=r"Portfolio Value Paths", xlabel=r"Days to Maturity", ylabel=r"Value")
        axs[1, 0].grid(alpha=0.5)

        # ── (3) Final-PnL KDE ───────────────────────────────────────────────────────
        kde = gaussian_kde(final_pnl, bw_method="scott")
        x_min, x_max = np.percentile(final_pnl, [0.1, 99.9])
        xg = np.linspace(x_min, x_max, 1_000)
        axs[0, 1].fill_between(xg, kde(xg), color="tab:blue", alpha=0.3)
        axs[0, 1].plot(xg, kde(xg), lw=2, color="tab:blue")
        axs[0, 1].axvline(pnl_mean, color="tab:blue", label=f"Mean: {pnl_mean:.4f}")
        axs[0, 1].axvline(VaR95, color="tab:green", label=f"VaR95: {VaR95:.4f}")
        mean, std = final_pnl.mean(), final_pnl.std()
        axs[0, 1].axvline(CVaR95, color="tab:orange", label=f"CVaR95: {CVaR95:.4f}")
        axs[0, 1].axvline(
            mean - 1.96 * std,
            color="tab:purple",
            linestyle="-",
            label=f"Mean-1.96*Std: {(mean - 1.96 * std):.4f}",
        )
        axs[0, 1].grid(alpha=0.5)
        axs[0, 1].legend()
        axs[0, 1].set(title=r"Final-PnL KDE", xlabel=r"PnL", ylabel=r"Density")

        # ── (4) Cumulative Δ + Γ costs ──────────────────────────────────────────────
        quantile_level = 0.95
        ax = axs[1, 1]
        Γ_Δ_costs = np.cumsum(Δ_costs, axis=1)

        mean_cum = Γ_Δ_costs.mean(axis=0)
        ax.plot(_time, mean_cum, color="tab:red", lw=1.5, label=r"Mean Cum $\Delta$ Costs")

        lower_q = (1 - quantile_level) / 2
        upper_q = 1 - lower_q
        q_lo, q_hi = np.quantile(Γ_Δ_costs, [lower_q, upper_q], axis=0)
        ax.fill_between(
            _time,
            q_lo,
            q_hi,
            color="tab:red",
            alpha=0.3,
            label=rf"{int(quantile_level * 100)}% IQR",
        )

        ax.set(
            xlabel=r"Days to maturity (d)",
            ylabel=r"Cum $\Delta$ Costs",
            title=r"Cumulative Delta Costs",
        )
        ax.legend(loc="upper right")
        ax.grid(which="major", alpha=0.5)
        ax.invert_xaxis()

        # ── (5) Variance Swap Price Evolution ─────────────────────────────────────────
        ax = axs[2, 0]

        # Prepare data for plotting - use 95% IQR for the fill between
        q_lo, q_hi = np.quantile(Vswap_value_array, [0.025, 0.975], axis=0)

        # Plot mean variance swap price
        ax.plot(
            _time,
            Vswap_value_array.mean(axis=0),
            color="tab:purple",
            lw=2,
            label="Mean Vswap Price",
        )

        # Add 95% IQR band
        ax.fill_between(_time, q_lo, q_hi, color="tab:purple", alpha=0.2, label="95% IQR")

        # Highlight strike price
        ax.axhline(
            y=0,
            color="tab:red",
            linestyle="--",
            alpha=0.5,
        )

        ax.set(
            xlabel=r"Days to maturity (d)",
            ylabel=r"Variance Swap Price",
            title=r"Variance Swap Price Evolution",
        )
        ax.legend(loc="upper left")
        ax.grid(which="major", alpha=0.5)
        ax.invert_xaxis()

        # ── (6) Delta Exposure Over Time ───────────────────────────────────────────
        ax = axs[2, 1]
        ax.plot(_time, D_net_exposure.T[:, :1000], lw=0.8, alpha=0.35, color="tab:blue")
        ax.invert_xaxis()
        ax.set(
            title=r"Delta Net Exposure Over Time",
            xlabel=r"Days to Maturity",
            ylabel=r"Net Delta Exposure",
        )
        ax.grid(alpha=0.5)

        # ── Overall title ───────────────────────────────────────────────────────────
        fig.suptitle(r"Delta-Vswap Hedging Simulation", y=1.01, fontsize=16)
        plt.tight_layout()
        plt.show()

        # ── Console summary ─────────────────────────────────────────────────────────
        print(f"Option Price : {V0_L: .5f}, Spread Earned: {V0_L * Option_SPREAD:.5f}")
        print(f"PnL mean     : {pnl_mean:.6f}  CI [{μ_ci[0]:.5f},{μ_ci[1]:.5f}]")
        print(f"PnL std      : {pnl_std: .6f}")
        print(f"VaR95        : {VaR95:.4f}  CI [{VaR_ci[0]:.4f},{VaR_ci[1]:.4f}]")
        print(f"CVaR95       : {CVaR95:.4f}  CI [{CVaR_ci[0]:.4f},{CVaR_ci[1]:.4f}]")
        print(f"Mean Δ cost  : {cum_Δ_costs.mean(): .6f}")
        print(f"Vswap Notiona: {vswap_notional:.4f}  Strike: {variance_swap_strike:.4f}")
        print(f"Mean Swap PnL: {Vswap_pnl_mean:.4f} CI [{Vswap_pnl_ci[0]:.4f},{Vswap_pnl_ci[1]:.4f}]")

    return (
        final_pnl,
        pnl_mean,
        μ_ci,
        pnl_std,
        VaR95,
        VaR_ci,
        CVaR95,
        CVaR_ci,
        cum_total,
        Vswap_value,
    )
