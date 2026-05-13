"""
metrics.py

Performance, Sequential, and Discrepancy Metrics for the Stop-Signal Task (SST).

This module contains functions to calculate comprehensive behavioral statistics 
from both observed (real) and simulated SST data. It includes basic accuracy 
metrics, temporal dependencies (e.g., Post-Error Slowing), regression slopes, 
and distance metrics used for Posterior Predictive Checks (PPC).
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, linregress
import warnings

# Suppress runtime warnings often caused by division by zero in empty categories 
# (e.g., when a subject has no Stop Errors).
warnings.simplefilter("ignore", RuntimeWarning)


# BASIC PERFORMANCE METRICS

def get_percentage(df):
    """
    Computes the proportions of the four primary SST response types.
    
    Response Types:
      - GS: Go Success (Correct Go direction)
      - GE: Go Error (Incorrect Go direction)
      - GM: Go Missing (Failed to respond to Go)
      - SS: Stop Success (Successfully inhibited)
      - SE: Stop Error (Failed to inhibit)
      
    Returns:
        tuple: (perc_gs, perc_ge, perc_gm, perc_ss)
    """
    stats = df['result'].value_counts()
    n_gs = stats.get("GS", 0)
    n_ge = stats.get("GE", 0)
    n_gm = stats.get("GM", 0)
    n_ss = stats.get("SS", 0)
    n_se = stats.get("SE", 0)
    
    n_go = n_gs + n_ge + n_gm
    n_stop = n_ss + n_se
    
    perc_gs = n_gs / n_go if n_go > 0 else 0
    perc_ge = n_ge / n_go if n_go > 0 else 0
    perc_gm = n_gm / n_go if n_go > 0 else 0
    perc_ss = n_ss / n_stop if n_stop > 0 else 0
    
    return perc_gs, perc_ge, perc_gm, perc_ss


# SEQUENTIAL & TEMPORAL METRICS

def get_sequential_stats(df, rt_col='rt_real'):
    """
    Computes sequential dependencies and global temporal trends.
    Assumes the dataframe is strictly sorted by trial order.
    
    Metrics Calculated:
      - pes: Post-Error Slowing (Go RT after Stop Error - Go RT after Go Success)
      - pss: Post-Stop Slowing (Go RT after Stop Success - Go RT after Go Success)
      - pges: Post-Go Error Slowing (Go RT after Go Error - Go RT after Go Success)
      - rt_acf_1: Lag-1 Autocorrelation of Go RTs (Short-term dependency)
      - rt_acf_2: Lag-2 Autocorrelation of Go RTs
      - slope: Global RT trend indicating fatigue or practice effects
      
    Args:
        df (pd.DataFrame): Trial data.
        rt_col (str): Column name containing the reaction times.
        
    Returns:
        tuple: (pes, pss, pges, rt_acf_1, rt_acf_2, slope)
    """
    try:
        # Create shifted columns for previous trial info
        df['prev_result'] = df['result'].shift(1)
        
        # Filter for current Go Success trials (we only analyze RT on correct goes)
        df_gs = df[df['result'] == 'GS'].copy()
        
        if df_gs.empty:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # --- 1. Post-Trial Adjustments (PES / PSS) ---
        # Baseline: RT when previous trial was also Go Success
        rt_after_gs = df_gs[df_gs['prev_result'] == 'GS'][rt_col].mean()
        
        # Condition A: RT after Stop Error (PES)
        rt_after_se = df_gs[df_gs['prev_result'] == 'SE'][rt_col].mean()
        pes = rt_after_se - rt_after_gs if pd.notna(rt_after_se) and pd.notna(rt_after_gs) else np.nan
        
        # Condition B: RT after Stop Success (PSS)
        rt_after_ss = df_gs[df_gs['prev_result'] == 'SS'][rt_col].mean()
        pss = rt_after_ss - rt_after_gs if pd.notna(rt_after_ss) and pd.notna(rt_after_gs) else np.nan
        
        # Condition C: RT after Go Error (PGE)
        rt_after_ge = df_gs[df_gs['prev_result'] == 'GE'][rt_col].mean()
        pges = rt_after_ge - rt_after_gs if (pd.notna(rt_after_ge) and pd.notna(rt_after_gs)) else np.nan
        
        # --- 2. Autocorrelation (Temporal Dependency) ---
        # We compute ACF only on the sequence of Go Success RTs to avoid outlier noise
        rt_series = df_gs[rt_col]
        rt_acf_1 = rt_series.autocorr(lag=1)
        rt_acf_2 = rt_series.autocorr(lag=2)
        
        # --- 3. Global Trend (Slope) ---
        # Regression of RT vs Trial Index (Are they getting slower/tired?)
        # Re-index to ensure x is 0, 1, 2...
        y = rt_series.values
        x = np.arange(len(y))
        
        if len(x) > 2:
            slope, _, _, _, _ = linregress(x, y)
        else:
            slope = np.nan
            
        return pes, pss, pges, rt_acf_1, rt_acf_2, slope

    except Exception as e:
        # Fail gracefully if columns missing or data too sparse
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

# DISTANCE METRICS (OBSERVED VS. SIMULATED)

def get_distance(df_obs, df_sim):
    """
    Main objective function. Calculates a tuple of discrepancy metrics 
    between observed and simulated data.
    """
    # 1. Percentage Distances
    dis_perc = get_dis_perc(df_obs, df_sim)
    
    # 2. RT Wasserstein Distances (Normalized)
    dis_ws_gs = get_ws_distance(df_obs, df_sim, "GS")
    dis_ws_ge = get_ws_distance(df_obs, df_sim, "GE")
    dis_ws_se = get_ws_distance(df_obs, df_sim, "SE")

    # 3. KS Distances (Distribution Shape)
    dis_ks_gs = get_ks_distance(df_obs, df_sim, 'rt', 'GS')
    dis_ks_se = get_ks_distance(df_obs, df_sim, 'rt', 'SE')

    # 4. SSD Mean Difference
    dis_ssd_mean = get_dis_ssd_mean(df_obs, df_sim)

    # Return flattened tuple of 10 metrics
    return (
        *dis_perc,      # (perc_gs, perc_ge, perc_gm, perc_ss)
        dis_ws_gs, dis_ws_ge, dis_ws_se,
        dis_ks_gs, dis_ks_se,
        dis_ssd_mean
    )


def get_dis_perc(df_obs, df_sim):
    """Absolute difference in response type proportions."""
    p_obs = get_percentage(df_obs)
    p_sim = get_percentage(df_sim)
    return tuple(abs(o - s) for o, s in zip(p_obs, p_sim))


def get_ws_distance(df_obs, df_sim, result_type):
    try:
        obs = df_obs[df_obs['result'] == result_type]['rt']
        sim = df_sim[df_sim['result'] == result_type]['rt']
        if len(obs) == 0 or len(sim) == 0: return 1.0
        return wasserstein_distance(obs, sim) / 40.0
    except ValueError:
        return 1.0


def get_ks_distance(df_obs, df_sim, column, result_type):
    df_obs_f = df_obs[df_obs['result'] == result_type]
    df_sim_f = df_sim[df_sim['result'] == result_type]
    x = df_obs_f[column].dropna().values.astype(float)
    y = df_sim_f[column].dropna().values.astype(float)
    if x.size == 0 or y.size == 0: return np.nan
    x.sort()
    y.sort()
    grid = np.unique(np.concatenate([x, y]))
    if grid.size == 0: return np.nan
    Fx = np.searchsorted(x, grid, side='right') / x.size
    Fy = np.searchsorted(y, grid, side='right') / y.size
    return float(np.max(np.abs(Fx - Fy)))


def get_dis_ssd_mean(df_obs, df_sim):
    obs = df_obs[df_obs['ssd'] >= 0]['ssd']
    sim = df_sim[df_sim['ssd'] >= 0]['ssd']
    if obs.empty or sim.empty: return np.nan
    return abs(obs.mean() - sim.mean())


# DESCRIPTIVE STATISTICS (ANALYSIS)

def get_stats_mean(df):
    """
    Compute summary stats for OBSERVED data (uses 'rt_real', 'ssd_real').
    NOW INCLUDES: Sequential metrics.
    """
    # Standard metrics
    percs = get_percentage(df)
    mrt_gs = df[df.result == "GS"].rt_real.mean()
    mrt_ge = df[df.result == "GE"].rt_real.mean()
    mrt_se = df[df.result == "SE"].rt_real.mean()
    mssd = df.ssd_real.mean() 
    ssrt = mrt_gs - mssd
    
    # Slopes
    rate_perc_ss = get_rate_perc_ss_ssd(df)
    rate_rt_se = get_rate_rt_se_ssd(df)
    
    # --- NEW: Sequential Metrics ---
    pes, pss, pges, acf1, acf2, slope = get_sequential_stats(df, rt_col='rt_real')
    
    return (*percs, mrt_gs, mrt_ge, mrt_se, mssd, ssrt, rate_perc_ss, rate_rt_se, 
            pes, pss, pges, acf1, acf2, slope)


def get_stats_mean_sim(df):
    """
    Compute summary stats for SIMULATED data (uses 'rt' * 25).
    """
    percs = get_percentage(df)
    mrt_gs = df[df.result == "GS"].rt.mean() * 25
    mrt_ge = df[df.result == "GE"].rt.mean() * 25
    mrt_se = df[df.result == "SE"].rt.mean() * 25
    mssd = df.ssd.mean() * 25
    ssrt = mrt_gs - mssd
    rate_perc_ss = get_rate_perc_ss_ssd(df)
    rate_rt_se = get_rate_rt_se_ssd(df)
    
    # Calculate sequential stats (scaled RTs)
    # Note: Simulated data usually needs 'rt' scaled to ms for comparison
    df['rt_ms'] = df['rt'] * 25
    pes, pss, pges, acf1, acf2, slope = get_sequential_stats(df, rt_col='rt_ms')
    
    return (*percs, mrt_gs, mrt_ge, mrt_se, mssd, ssrt, rate_perc_ss, rate_rt_se, 
            pes, pss, pges, acf1, acf2, slope)


# REGRESSION / SLOPE HELPERS

def get_rate_perc_ss_ssd(df):
    try:
        df_stop = df[df['result'].isin(['SS', 'SE'])]
        if df_stop.empty: return None
        counts = df_stop.groupby('ssd')['result'].value_counts().unstack(fill_value=0)
        if 'SS' not in counts: counts['SS'] = 0
        if 'SE' not in counts: counts['SE'] = 0
        perc_ss = counts['SS'] / (counts['SS'] + counts['SE'])
        x = perc_ss.index.values
        y = perc_ss.values
        if len(x) < 2: return None
        rate, _, _, _, _ = linregress(x, y)
        return rate
    except Exception:
        return None


def get_rate_rt_se_ssd(df):
    try:
        df_se = df[df['result'] == 'SE']
        means = df_se.groupby('ssd')['rt'].mean()
        x = means.index.values
        y = means.values
        if len(x) < 2: return None
        rate, _, _, _, _ = linregress(x, y)
        return rate
    except Exception:
        return None