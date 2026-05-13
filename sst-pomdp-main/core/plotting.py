# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Local application/library imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from core import simulation


def plot_belief(out_go, out_stop, show_cnt=5, saveto=None):
    """
    Generates a 2x3 composite figure illustrating the evolution of belief states 
    (Go belief and Stop trial belief) over time across five distinct behavioral outcomes.
    
    Args:
        out_go: Simulation output dictionary for Go trials.
        out_stop: Simulation output dictionary for Stop trials.
        show_cnt: Number of individual trial sample paths to overlay.
        saveto: Optional filepath to save the output figure.
    """
    
    # Configure PLOS-compliant typography and color palette
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    style = {
        "label_fontsize": 9,
        "title_fontsize": 10,
        "tick_fontsize": 8,
        "legend_fontsize": 9,
        "panel_label_size": 12,

        "mean_lw": 1.5,
        "mean_alpha": 1.0,
        "sem_alpha": 0.25,

        "sample_active_lw": 0.6,
        "sample_active_alpha": 0.4,
        
        "marker_size": 15, 
        "marker_edge_width": 0.5,

        "ssd_lw": 1.0,
        "ssd_alpha": 0.6,
    }

    colors = {
        "beta": "#4E7C59",    # Sage/Forest Green
        "sigma": "#800020",   # Burgundy
        "survival": "gray"
    }

    df_go = pd.DataFrame(out_go)
    df_stop = pd.DataFrame(out_stop)
    ssd = df_stop.iloc[0]['ssd'] if len(df_stop) > 0 else None

    # Categorize data by behavioral outcome
    df_gs = df_go[df_go['result'] == 'GS']
    df_ge = df_go[df_go['result'] == 'GE']
    df_gm = df_go[df_go['result'] == 'GM']
    df_ss = df_stop[df_stop['result'] == 'SS']
    df_se = df_stop[df_stop['result'] == 'SE']

    df_list = [df_gs, df_ge, df_gm, df_ss, df_se]
    condition_names = ['Go Success', 'Go Error', 'Go Missing', 'Stop Success', 'Stop Error']

    # Initialize standard journal-sized figure
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 4.8))
    
    MAX_TIME_STEPS = 40
    xticks = [1, 10, 20, 30, 40]
    panel_labels = ['A', 'B', 'C', 'D', 'E']

    for i, df in enumerate(df_list):
        row, col = divmod(i, 3)
        axis = ax[row, col]

        axis.text(-0.25, 1.1, panel_labels[i], transform=axis.transAxes,
                  fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

        if len(df) == 0:
            axis.axis('off')
            continue

        # Extract truncated belief sequences and reaction times up to the decision point
        beta_trunc_list = []
        sigma_trunc_list = []
        rt_indices = []

        for _, row_data in df.iterrows():
            beta_seq = row_data['beta_seq']
            sigma_seq = row_data['sigma_seq']
            policy_seq = row_data['policy_seq']

            decision_idx = len(policy_seq) - 1 
            for t_idx, act in enumerate(policy_seq):
                if act != 2: 
                    decision_idx = t_idx
                    break
            
            rt_indices.append(decision_idx)
            limit = decision_idx + 1
            beta_trunc_list.append(beta_seq[:limit])
            sigma_trunc_list.append(sigma_seq[:limit])

        # Configure secondary y-axis for survival probability shadow
        ax2 = axis.twinx()
        valid_rts = np.sort([r + 1 for r in rt_indices])
        
        if len(valid_rts) > 0:
            n_samples = len(valid_rts)
            y_survival = 1.0 - np.arange(1, n_samples + 1) / n_samples
            x_surv_plot = np.concatenate(([1], valid_rts))
            y_surv_plot = np.concatenate(([1.0], y_survival))

            ax2.fill_between(x_surv_plot, y_surv_plot, step='post', 
                             color=colors['survival'], alpha=0.15,
                             linewidth=0) 

        ax2.set_ylabel('Survival Prob.', color='gray', fontsize=style['label_fontsize'])
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=style['tick_fontsize'])
        ax2.set_ylim(0, 1.05) 
        ax2.set_yticks([0, 1.0])
        ax2.set_yticklabels(["0", "1"])
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('gray')
        ax2.spines['right'].set_linewidth(0.5)

        # Plot aggregate mean and standard error for belief states
        df_beta_trunc = pd.DataFrame(beta_trunc_list)
        df_sigma_trunc = pd.DataFrame(sigma_trunc_list)

        beta_mean = df_beta_trunc.mean(axis=0)
        beta_sem = df_beta_trunc.sem(axis=0)
        sigma_mean = df_sigma_trunc.mean(axis=0)
        sigma_sem = df_sigma_trunc.sem(axis=0)

        x_vals_mean = np.arange(1, len(beta_mean) + 1)

        axis.plot(x_vals_mean, beta_mean, 
                  color=colors["beta"], linewidth=style['mean_lw'], alpha=style['mean_alpha'])
        axis.fill_between(x_vals_mean, beta_mean - beta_sem, beta_mean + beta_sem,
                          color=colors["beta"], alpha=style['sem_alpha'], linewidth=0)

        axis.plot(x_vals_mean, sigma_mean, 
                  color=colors["sigma"], linewidth=style['mean_lw'], alpha=style['mean_alpha'])
        axis.fill_between(x_vals_mean, sigma_mean - sigma_sem, sigma_mean + sigma_sem,
                          color=colors["sigma"], alpha=style['sem_alpha'], linewidth=0)

        # Overlay individual sample paths
        subset_indices = list(range(min(show_cnt, len(df))))
        for j in subset_indices:
            d_idx = rt_indices[j]
            limit = d_idx + 1
            b_seq = df.iloc[j]['beta_seq']
            s_seq = df.iloc[j]['sigma_seq']
            x_sample = np.arange(1, limit + 1)
            is_last_step = (d_idx >= MAX_TIME_STEPS - 1)

            axis.plot(x_sample, b_seq[:limit],
                      color=colors["beta"], linewidth=style['sample_active_lw'], 
                      alpha=style['sample_active_alpha'])
            if not is_last_step:
                axis.scatter(x_sample[-1], b_seq[d_idx], 
                             s=style['marker_size'], color=colors["beta"], 
                             alpha=0.9, zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

            axis.plot(x_sample, s_seq[:limit],
                      color=colors["sigma"], linewidth=style['sample_active_lw'], 
                      alpha=style['sample_active_alpha'])
            if not is_last_step:
                axis.scatter(x_sample[-1], s_seq[d_idx], 
                             s=style['marker_size'], color=colors["sigma"], 
                             alpha=0.9, zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

        # Format subplot axes and boundaries
        if condition_names[i].startswith('Stop') and ssd is not None:
            # Assuming a standard 12-timestep Stop Signal Duration
            axis.axvline(x=ssd + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])
            axis.axvline(x=ssd + 12 + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])

        axis.set_xlim(1, MAX_TIME_STEPS)
        axis.set_ylim(0, 1.05)
        axis.set_xticks(xticks)
        axis.set_yticks([0, 0.5, 1.0])
        
        axis.set_title(f"{condition_names[i]}", pad=6, fontsize=style['title_fontsize'], fontweight='regular')
        axis.tick_params(axis='both', labelsize=style['tick_fontsize'])

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False) 
        
        axis.set_zorder(ax2.get_zorder() + 1) 
        axis.patch.set_visible(False)
        
        axis.set_ylabel(r"Belief State", fontsize=style['label_fontsize'])
        axis.set_xlabel("Time", fontsize=style['label_fontsize'])

    ax[1, 2].axis("off")
    
    # Increase horizontal space to accommodate internal Y-labels
    plt.subplots_adjust(hspace=0.5, wspace=0.55, top=0.88, bottom=0.12, left=0.08, right=0.92)

    # Construct global legend
    legend_handles = [
        Line2D([0], [0], color=colors["beta"], lw=style['mean_lw'], label=r'Go Belief $\beta^\nu_t$'),
        Line2D([0], [0], color=colors["sigma"], lw=style['mean_lw'], label=r'Stop Trial Belief $\sigma^\nu_t$'),
        Patch(facecolor=colors['survival'], alpha=0.15, label='Survival Probability'),
        Line2D([0], [0], color='k', linestyle='--', lw=style['ssd_lw'], alpha=style['ssd_alpha'], label='Stop Signal Onset / Offset')
    ]

    fig.legend(handles=legend_handles, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), 
               ncol=4, 
               frameon=False, 
               fontsize=style['legend_fontsize'],
               handlelength=1.5,
               columnspacing=1.5)

    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()


def plot_action_value(out_go, out_stop, show_cnt=5, saveto=None):
    """
    Generates a 2x3 composite figure illustrating the action values Q(b, a) 
    over time across five distinct behavioral outcomes.
    
    Args:
        out_go: Simulation output dictionary for Go trials.
        out_stop: Simulation output dictionary for Stop trials.
        show_cnt: Number of individual trial sample paths to overlay.
        saveto: Optional filepath to save the output figure.
    """
    
    # Configure PLOS-compliant typography and color palette
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    style = {
        "label_fontsize": 9,
        "title_fontsize": 10,
        "tick_fontsize": 8,
        "legend_fontsize": 9,
        "panel_label_size": 12,

        "mean_lw": 1.5,
        "mean_alpha": 1.0,
        "sem_alpha": 0.25,

        "sample_active_lw": 0.6,
        "sample_active_alpha": 0.4,
        
        "marker_size": 15, 
        "marker_edge_width": 0.5,

        "ssd_lw": 1.0,
        "ssd_alpha": 0.6,
    }

    colors = {
        "Q_L": "#E3B23C", # Gold
        "Q_R": "#3A6EA5", # Blue
        "Q_W": "#7A7A7A", # Dark Gray for Wait Action
        "survival": "gray"
    }

    df_go = pd.DataFrame(out_go)
    df_stop = pd.DataFrame(out_stop)
    ssd = df_stop.iloc[0]['ssd'] if len(df_stop) > 0 else None

    # Categorize data by behavioral outcome
    df_gs = df_go[df_go['result'] == 'GS']
    df_ge = df_go[df_go['result'] == 'GE']
    df_gm = df_go[df_go['result'] == 'GM']
    df_ss = df_stop[df_stop['result'] == 'SS']
    df_se = df_stop[df_stop['result'] == 'SE']

    df_list = [df_gs, df_ge, df_gm, df_ss, df_se]
    condition_names = ['Go Success', 'Go Error', 'Go Missing', 'Stop Success', 'Stop Error']

    # Initialize standard journal-sized figure
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 4.8))
    
    MAX_TIME_STEPS = 40
    xticks = [1, 10, 20, 30, 40]
    panel_labels = ['A', 'B', 'C', 'D', 'E']

    for i, df in enumerate(df_list):
        row, col = divmod(i, 3)
        axis = ax[row, col]

        axis.text(-0.25, 1.1, panel_labels[i], transform=axis.transAxes,
                  fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

        if len(df) == 0:
            axis.axis('off')
            continue

        # Extract truncated action value sequences and reaction times up to the decision point
        q_l_trunc_list = []
        q_r_trunc_list = []
        q_w_trunc_list = []
        rt_indices = []

        for _, row_data in df.iterrows():
            ql_seq = row_data['value_left_seq']
            qr_seq = row_data['value_right_seq']
            qw_seq = row_data['value_wait_seq']
            policy_seq = row_data['policy_seq']

            decision_idx = len(policy_seq) - 1 
            for t_idx, act in enumerate(policy_seq):
                if act != 2: 
                    decision_idx = t_idx
                    break
            
            rt_indices.append(decision_idx)
            limit = decision_idx + 1
            
            q_l_trunc_list.append(ql_seq[:limit])
            q_r_trunc_list.append(qr_seq[:limit])
            q_w_trunc_list.append(qw_seq[:limit])

        # Configure secondary y-axis for survival probability shadow
        ax2 = axis.twinx()
        valid_rts = np.sort([r + 1 for r in rt_indices])
        
        if len(valid_rts) > 0:
            n_samples = len(valid_rts)
            y_survival = 1.0 - np.arange(1, n_samples + 1) / n_samples
            x_surv_plot = np.concatenate(([1], valid_rts))
            y_surv_plot = np.concatenate(([1.0], y_survival))
            
            ax2.fill_between(x_surv_plot, y_surv_plot, step='post', 
                             color=colors['survival'], alpha=0.15,
                             linewidth=0) 

        ax2.set_ylabel('Survival Prob.', color='gray', fontsize=style['label_fontsize'])
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=style['tick_fontsize'])
        ax2.set_ylim(0, 1.05) 
        ax2.set_yticks([0, 1.0])
        ax2.set_yticklabels(["0", "1"])
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('gray')
        ax2.spines['right'].set_linewidth(0.5)

        # Plot aggregate mean and standard error for action values
        df_ql = pd.DataFrame(q_l_trunc_list)
        df_qr = pd.DataFrame(q_r_trunc_list)
        df_qw = pd.DataFrame(q_w_trunc_list)

        def plot_mean_with_sem(data_df, color):
            """Helper to compute and plot the mean sequence with SEM shading."""
            mean = data_df.mean(axis=0)
            sem = data_df.sem(axis=0)
            x_vals = np.arange(1, len(mean) + 1)
            axis.plot(x_vals, mean, 
                      color=color, linewidth=style['mean_lw'], alpha=style['mean_alpha'])
            axis.fill_between(x_vals, mean - sem, mean + sem,
                              color=color, alpha=style['sem_alpha'], linewidth=0)

        # Layer order: Wait action (background), then active choices
        plot_mean_with_sem(df_qw, colors["Q_W"])
        plot_mean_with_sem(df_ql, colors["Q_L"])
        plot_mean_with_sem(df_qr, colors["Q_R"])

        # Overlay individual sample paths
        subset_indices = list(range(min(show_cnt, len(df))))
        for j in subset_indices:
            d_idx = rt_indices[j]
            limit = d_idx + 1
            
            ql_seq = df.iloc[j]['value_left_seq']
            qr_seq = df.iloc[j]['value_right_seq']
            qw_seq = df.iloc[j]['value_wait_seq']
            
            x_sample = np.arange(1, limit + 1)
            is_last_step = (d_idx >= MAX_TIME_STEPS - 1)

            def plot_sample(seq, color):
                """Helper to plot individual trial traces and decision markers."""
                axis.plot(x_sample, seq[:limit],
                          color=color, linewidth=style['sample_active_lw'], 
                          alpha=style['sample_active_alpha'])
                if not is_last_step:
                    axis.scatter(x_sample[-1], seq[d_idx], 
                                 s=style['marker_size'], color=color, 
                                 alpha=0.9, zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

            plot_sample(qw_seq, colors["Q_W"])
            plot_sample(ql_seq, colors["Q_L"])
            plot_sample(qr_seq, colors["Q_R"])

        # Format subplot axes and boundaries
        if condition_names[i].startswith('Stop') and ssd is not None:
            # Assuming a standard 12-timestep Stop Signal Duration
            axis.axvline(x=ssd + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])
            axis.axvline(x=ssd + 12 + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])

        axis.set_xlim(1, MAX_TIME_STEPS)
        axis.set_xticks(xticks)
        
        axis.set_title(f"{condition_names[i]}", pad=6, fontsize=style['title_fontsize'], fontweight='regular')
        axis.tick_params(axis='both', labelsize=style['tick_fontsize'])

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False) 
        
        axis.set_zorder(ax2.get_zorder() + 1) 
        axis.patch.set_visible(False)

        axis.set_ylabel("Action Value", fontsize=style['label_fontsize'])
        axis.set_xlabel("Time Step", fontsize=style['label_fontsize'])

    ax[1, 2].axis("off")
    
    plt.subplots_adjust(hspace=0.5, wspace=0.55, top=0.88, bottom=0.12, left=0.08, right=0.92)

    # Construct global legend
    legend_handles = [
        Line2D([0], [0], color=colors["Q_W"], lw=style['mean_lw'], label=r'$Q_t(\mathbf{b}_t,W)$'),
        Line2D([0], [0], color=colors["Q_L"], lw=style['mean_lw'], label=r'$Q_t(\mathbf{b}_t,L)$'),
        Line2D([0], [0], color=colors["Q_R"], lw=style['mean_lw'], label=r'$Q_t(\mathbf{b}_t,R)$'),
        Patch(facecolor=colors['survival'], alpha=0.15, label='Survival Probability'),
        Line2D([0], [0], color='k', linestyle='--', lw=style['ssd_lw'], alpha=style['ssd_alpha'], label='Stop Signal Onset / Offset')
    ]

    fig.legend(handles=legend_handles, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), 
               ncol=5, 
               frameon=False, 
               fontsize=style['legend_fontsize'],
               handlelength=1.5,
               columnspacing=1.5)
    
    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()


def plot_stop_prior_context(
    model,
    params: dict,
    rates=(0.05, 1/6, 0.35, 0.55, 0.80),
    ssd=10,
    n_batch=500,     
    n_session=1000,  
    saveto=None,
    t_max=40
):
    """
    Generates a 2x3 composite figure illustrating the effect of stop priors on 
    behavioral accuracy, belief updates, and decision variables (gap) over time.
    
    Args:
        model: The computational model object.
        params: Dictionary of model parameters.
        rates: Tuple of stop trial prior probabilities to simulate.
        ssd: Stop signal delay for stop trials.
        n_batch: Number of batches for timecourse simulations.
        n_session: Number of sessions for behavioral statistic simulations.
        saveto: Optional filepath to save the output figure.
        t_max: Maximum timestep limit for visualization.
    """
    
    # Configure PLOS-compliant typography and earth-tone color palettes
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    style = {
        "label_fontsize": 9,     
        "tick_fontsize": 8,      
        "legend_fontsize": 8,    
        "panel_label_size": 12,  
        "line_width": 1.0,      
        "marker_size": 3.5,        
        "cap_size": 1.5,
        "shadow_color": "gray"
    }

    # Behavioral metrics palette (Neutral/Grays)
    color_acc = '#363636' 
    color_rt  = "#7f7f7f" 
    
    # Belief updates palette (Burgundy gradient)
    cmap_bel = LinearSegmentedColormap.from_list("earth_red", ["#E6B8B8", "#800020"]) 
    rate_colors_bel = [cmap_bel(i) for i in np.linspace(0.1, 1.0, len(rates))]

    # Action gap palette (Coffee/Bronze gradient)
    cmap_gap = LinearSegmentedColormap.from_list("earth_brown", ["#D7CCC8", "#5D4037"])
    rate_colors_gap = [cmap_gap(i) for i in np.linspace(0.1, 1.0, len(rates))]

    # Helper function to truncate sequence lists and compute descriptive statistics
    def _truncate_and_collect(seq_list):
        if not seq_list: 
            return np.full(t_max, np.nan), np.full(t_max, np.nan)
        df = pd.DataFrame(seq_list).reindex(columns=range(t_max))
        with np.errstate(invalid='ignore'):
            mu = df.mean(axis=0).to_numpy()
            sd = df.std(axis=0, ddof=1).to_numpy()
            count = df.count(axis=0).to_numpy()
            se = np.where(count > 1, sd / np.sqrt(count), np.nan)
        return mu, se

    beh_rows = []
    timecourse_go = []
    timecourse_stop = []

    print("Simulating across varying stop priors...")
    
    # Run model simulations across the provided prior rates
    for r_idx, r in enumerate(rates):
        p = params.copy()
        p['rate_stop_trial'] = float(r)
        m = model.__class__(**p)
        m.value_iteration_tensor()

        # Simulate behavioral sessions
        out_sess = []
        for _ in range(n_session):
            is_stop = np.random.rand() < float(r)
            gs = np.random.choice(["left", "right"])
            type_ = "stop" if is_stop else "nonstop"
            ssd_ = ssd if is_stop else None
            res_tuple = simulation.simu_trial(m, gs, type_, ssd_, verbose=False) 
            out_sess.append(res_tuple)
        
        df_beh = pd.DataFrame(out_sess, columns=["result", "rt", "ssd"])
        
        # Calculate behavioral statistics for Go and Stop contexts
        go_df = df_beh[df_beh["result"].isin(["GS", "GE", "GM"])]
        if len(go_df) > 0:
            go_acc = (go_df["result"]=="GS").mean()
            go_acc_sem = np.sqrt(go_acc*(1-go_acc)/len(go_df))
            gs_df = go_df[go_df["result"]=="GS"]
            go_rt = gs_df["rt"].mean() if len(gs_df)>0 else np.nan
            go_rt_sem = gs_df["rt"].std(ddof=1)/np.sqrt(len(gs_df)) if len(gs_df)>0 else np.nan
        else:
            go_acc, go_acc_sem, go_rt, go_rt_sem = [np.nan]*4

        stop_df = df_beh[df_beh["result"].isin(["SS", "SE"])]
        if len(stop_df) > 0:
            stop_acc = (stop_df["result"]=="SS").mean()
            stop_acc_sem = np.sqrt(stop_acc*(1-stop_acc)/len(stop_df))
            se_df = stop_df[stop_df["result"]=="SE"]
            stop_rt = se_df["rt"].mean() if len(se_df)>0 else np.nan 
            stop_rt_sem = se_df["rt"].std(ddof=1)/np.sqrt(len(se_df)) if len(se_df)>0 else np.nan
        else:
            stop_acc, stop_acc_sem, stop_rt, stop_rt_sem = [np.nan]*4

        beh_rows.append({
            "rate": r,
            "go_acc": go_acc, "go_acc_sem": go_acc_sem, 
            "go_rt": go_rt, "go_rt_sem": go_rt_sem,
            "stop_acc": stop_acc, "stop_acc_sem": stop_acc_sem, 
            "stop_rt": stop_rt, "stop_rt_sem": stop_rt_sem
        })

        # Simulate detailed timecourses for decision variables
        g_data = {"GS": {"sig": [], "gap": [], "rt": []}, 
                  "GE": {"sig": [], "gap": [], "rt": []}, 
                  "GM": {"sig": [], "gap": [], "rt": []}}
        s_data = {"SS": {"sig": [], "gap": [], "rt": []}, 
                  "SE": {"sig": [], "gap": [], "rt": []}}

        def process_trial_result(res_dict, store_dict):
            """Extracts belief and value gap metrics up to the decision point."""
            out_type = res_dict["result"] 
            if out_type not in store_dict: return
            
            policy_seq = res_dict["policy_seq"]
            decision_idx = len(policy_seq) - 1
            for t_idx, act in enumerate(policy_seq):
                if act != 2: 
                    decision_idx = t_idx
                    break
            limit = decision_idx + 1
            sigma_seq = np.array(res_dict["sigma_seq"])[:limit]
            vw = np.array(res_dict["value_wait_seq"])[:limit]
            vl = np.array(res_dict["value_left_seq"])[:limit]
            vr = np.array(res_dict["value_right_seq"])[:limit]
            
            if len(vl) > 0:
                min_go = np.minimum(vl, vr)
                gap_seq = vw - min_go
            else:
                gap_seq = []

            store_dict[out_type]["sig"].append(sigma_seq)
            store_dict[out_type]["gap"].append(gap_seq)
            store_dict[out_type]["rt"].append(decision_idx)

        for _ in range(n_batch):
            true_go = np.random.choice(["left", "right"])
            res_g = simulation.simu_trial(m, true_go, "nonstop", None, verbose=True)
            process_trial_result(res_g, g_data)
            res_s = simulation.simu_trial(m, true_go, "stop", ssd, verbose=True)
            process_trial_result(res_s, s_data)

        # Aggregate timecourse results
        pkt_g = {
            "rate": r, 
            "color_bel": rate_colors_bel[r_idx], 
            "color_gap": rate_colors_gap[r_idx],
            "rts": {k: v["rt"] for k,v in g_data.items()}
        }
        for cat in ["GS", "GE", "GM"]:
            pkt_g[f"sigma_{cat}_mu"], pkt_g[f"sigma_{cat}_sem"] = _truncate_and_collect(g_data[cat]["sig"])
            pkt_g[f"gap_{cat}_mu"],   pkt_g[f"gap_{cat}_sem"]   = _truncate_and_collect(g_data[cat]["gap"])
        timecourse_go.append(pkt_g)

        pkt_s = {
            "rate": r, 
            "color_bel": rate_colors_bel[r_idx], 
            "color_gap": rate_colors_gap[r_idx],
            "rts": {k: v["rt"] for k,v in s_data.items()}
        }
        for cat in ["SS", "SE"]:
            pkt_s[f"sigma_{cat}_mu"], pkt_s[f"sigma_{cat}_sem"] = _truncate_and_collect(s_data[cat]["sig"])
            pkt_s[f"gap_{cat}_mu"],   pkt_s[f"gap_{cat}_sem"]   = _truncate_and_collect(s_data[cat]["gap"])
        timecourse_stop.append(pkt_s)

    df_beh_summary = pd.DataFrame(beh_rows)

    # Initialize plotting layout
    fig = plt.figure(figsize=(7.5, 4.8)) 
    gs = fig.add_gridspec(2, 3, wspace=0.60, hspace=0.35)

    ax_go_beh   = fig.add_subplot(gs[0, 0])
    ax_go_bel   = fig.add_subplot(gs[0, 1])
    ax_go_gap   = fig.add_subplot(gs[0, 2])
    ax_stop_beh = fig.add_subplot(gs[1, 0])
    ax_stop_bel = fig.add_subplot(gs[1, 1])
    ax_stop_gap = fig.add_subplot(gs[1, 2])

    axes_all = [ax_go_beh, ax_go_bel, ax_go_gap, ax_stop_beh, ax_stop_bel, ax_stop_gap]
    axes_time = [ax_go_bel, ax_go_gap, ax_stop_bel, ax_stop_gap]
    x_time = np.arange(1, t_max+1)

    # Create shadow twin axes for survival probability overlays
    shadow_axes = {}
    for ax in axes_time:
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        ax_shad = ax.twinx()
        ax_shad.set_ylim(0, 1.05)
        ax_shad.set_yticks([0, 1.0]) 
        ax_shad.set_yticklabels(["0", "1"], fontsize=style['tick_fontsize'], color='gray')
        ax_shad.set_ylabel('Survival Prob.', fontsize=style['label_fontsize'], color='gray')
        ax_shad.spines['top'].set_visible(False)
        ax_shad.spines['left'].set_visible(False)
        ax_shad.spines['right'].set_color('gray')
        ax_shad.spines['right'].set_linewidth(0.5)
        ax_shad.set_zorder(0)
        shadow_axes[ax] = ax_shad
    
    def draw_shadow(target_ax, rt_list, alpha):
        """Draws the cumulative survival probability shadow based on RT distributions."""
        if not rt_list or len(rt_list) == 0: return
        try:
            valid_rts = np.sort([int(float(r)) + 1 for r in rt_list])
        except: return
        n = len(valid_rts)
        if n == 0: return
        y = 1.0 - np.arange(1, n + 1) / n
        x_plot = np.concatenate(([1], valid_rts))
        y_plot = np.concatenate(([1.0], y))
        
        target_ax.fill_between(x_plot, y_plot, step='post', 
                               color="gray", 
                               alpha=alpha, 
                               linewidth=0)

    mid_idx = len(rates) // 2
    rts_go = timecourse_go[mid_idx]["rts"]
    rts_stop = timecourse_stop[mid_idx]["rts"]

    # Overlay survival probabilities (shadows) for Go and Stop contexts based on specific outcomes
    draw_shadow(shadow_axes[ax_go_bel], rts_go.get("GM", []), alpha=0.05)
    draw_shadow(shadow_axes[ax_go_bel], rts_go.get("GS", []), alpha=0.10)
    draw_shadow(shadow_axes[ax_go_bel], rts_go.get("GE", []), alpha=0.15)
    
    draw_shadow(shadow_axes[ax_go_gap], rts_go.get("GM", []), alpha=0.05)
    draw_shadow(shadow_axes[ax_go_gap], rts_go.get("GS", []), alpha=0.10)
    draw_shadow(shadow_axes[ax_go_gap], rts_go.get("GE", []), alpha=0.15)

    draw_shadow(shadow_axes[ax_stop_bel], rts_stop.get("SS", []), alpha=0.05)
    draw_shadow(shadow_axes[ax_stop_bel], rts_stop.get("SE", []), alpha=0.10)
    
    draw_shadow(shadow_axes[ax_stop_gap], rts_stop.get("SS", []), alpha=0.05)
    draw_shadow(shadow_axes[ax_stop_gap], rts_stop.get("SE", []), alpha=0.10)
    
    # Plot primary time series variables
    def plot_tc(ax, dataset, metric_prefix, outcome_keys, styles_map, color_key):
        """Plots timecourse data with shaded standard error regions."""
        for d in dataset:
            col = d[color_key]
            for out_key in outcome_keys:
                mu = d[f"{metric_prefix}_{out_key}_mu"]
                se = d[f"{metric_prefix}_{out_key}_sem"]
                ls = styles_map[out_key]["ls"]
                mask = ~np.isnan(mu)
                if np.any(mask):
                    ax.plot(x_time[mask], mu[mask], color=col, ls=ls, lw=style['line_width'])
                    ax.fill_between(x_time[mask], mu[mask]-se[mask], mu[mask]+se[mask], 
                                    color=col, alpha=0.15, lw=0)

    st_map = {
        "GS": {"ls": "-"}, "GE": {"ls": "--"}, "GM": {"ls": ":"},
        "SS": {"ls": "-"}, "SE": {"ls": "--"}
    }
    
    plot_tc(ax_go_bel, timecourse_go, "sigma", ["GS", "GE", "GM"], st_map, "color_bel")
    plot_tc(ax_go_gap, timecourse_go, "gap",   ["GS", "GE", "GM"], st_map, "color_gap")
    
    plot_tc(ax_stop_bel, timecourse_stop, "sigma", ["SS", "SE"], st_map, "color_bel")
    plot_tc(ax_stop_gap, timecourse_stop, "gap",   ["SS", "SE"], st_map, "color_gap")

    # Construct behavior summary panels (Leftmost columns)
    def plot_behavior_panel(ax, df, acc_key, acc_sem_key, rt_key, rt_sem_key, context_label):
        """Generates dual-axis behavioral panels (Accuracy and RT) snapped to the bottom-right legend."""
        
        # Plot Accuracy (Left Axis)
        ax.plot(df["rate"], df[acc_key], color=color_acc, linestyle='-', lw=style['line_width'], zorder=1)
        ax.errorbar(df["rate"], df[acc_key], yerr=df[acc_sem_key],
                    fmt='none', ecolor=color_acc, capsize=style['cap_size'], zorder=1)
        ax.scatter(df["rate"], df[acc_key], s=20, 
                   c=rate_colors_bel, marker='o', 
                   edgecolors=color_acc, linewidth=0.5, zorder=2)
        
        ax.set_ylabel("Accuracy", color=color_acc, fontsize=style['label_fontsize'])
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis='y', colors=color_acc, labelsize=style['tick_fontsize'])
        ax.spines['left'].set_color(color_acc)
        ax.spines['top'].set_visible(False)

        # Plot RT (Right Axis)
        axR = ax.twinx()
        axR.plot(df["rate"], df[rt_key], color=color_rt, linestyle='-', lw=style['line_width'], zorder=1)
        axR.errorbar(df["rate"], df[rt_key], yerr=df[rt_sem_key],
                    fmt='none', ecolor=color_rt, capsize=style['cap_size'], zorder=1)
        axR.scatter(df["rate"], df[rt_key], s=20, 
                    c=rate_colors_bel, marker='s', 
                    edgecolors=color_rt, linewidth=0.5, zorder=2)
        
        axR.set_ylabel("Time Step", color=color_rt, fontsize=style['label_fontsize'])
        axR.set_ylim(1, 40)
        axR.set_yticks([1, 20, 40])
        axR.tick_params(axis='y', colors=color_rt, labelsize=style['tick_fontsize'])
        axR.spines['right'].set_color(color_rt)
        axR.spines['top'].set_visible(False)
        
        # Add localized panel legend for markers
        h_acc = Line2D([0], [0], color=color_acc, marker='o', linestyle='-', 
                       lw=style['line_width'], markersize=style['marker_size'], label=f"{context_label} Acc.")
        h_rt  = Line2D([0], [0], color=color_rt, marker='s', linestyle='-', 
                       lw=style['line_width'], markersize=style['marker_size'], label=f"{context_label} RT")

        ax.legend(handles=[h_acc, h_rt], loc='lower right', bbox_to_anchor=(1.0, 0.02), 
                  frameon=False, fontsize=style['tick_fontsize'], handlelength=1.0)
        
        return axR

    plot_behavior_panel(ax_go_beh, df_beh_summary, "go_acc", "go_acc_sem", "go_rt", "go_rt_sem", "Go")
    plot_behavior_panel(ax_stop_beh, df_beh_summary, "stop_acc", "stop_acc_sem", "stop_rt", "stop_rt_sem", "Stop")

    # Apply global axis formatting and constraints
    for ax in [ax_go_beh, ax_stop_beh]:
        ax.set_xlabel(r"Stop Prior $\tilde{r}^\nu$", fontsize=style['label_fontsize'])
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', labelsize=style['tick_fontsize'])

    for ax in axes_time:
        ax.set_xlim(1, 40)
        ax.set_xticks([1, 20, 40])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=style['tick_fontsize'])
        ax.set_xlabel("Time Step", fontsize=style['label_fontsize'])

    ax_go_bel.set_ylabel(r"Belief $\sigma^\nu_t$", fontsize=style['label_fontsize'])
    ax_stop_bel.set_ylabel(r"Belief $\sigma^\nu_t$", fontsize=style['label_fontsize'])
    label_gap = r"$Q(W) - \min(Q(L), Q(R))$"
    ax_go_gap.set_ylabel(label_gap, fontsize=style['label_fontsize'])
    ax_stop_gap.set_ylabel(label_gap, fontsize=style['label_fontsize'])
    
    stop_signal_duration = 12 
    if ssd:
        for ax in [ax_stop_bel, ax_stop_gap]:
            ax.axvline(ssd+1, color='k', ls='--', alpha=0.5, linewidth=0.8)
            ax.axvline(ssd+1 + stop_signal_duration, color='k', ls='--', alpha=0.5, linewidth=0.8)

    # Insert panel lettering
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for ax, label in zip(axes_all, labels):
        ax.text(-0.15, 1.15, label, transform=ax.transAxes,
                fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

    # Construct globally aligned bottom legend 
    y_row1 = 0.12  
    y_row2 = 0.08  
    
    fig.text(0.02, y_row1, r"Stop Prior $\tilde{r}^\nu$", ha='left', va='center', 
             fontsize=style['legend_fontsize']+1, fontweight='normal')
    
    h_bel = [Line2D([0],[0], color=c, lw=1, label=f"{r:.2f}") for r,c in zip(rates, rate_colors_bel)]
    fig.legend(handles=h_bel, loc='center left', bbox_to_anchor=(0.12, y_row1),
               ncol=len(rates), frameon=False, fontsize=style['legend_fontsize'],
               columnspacing=0.8, handletextpad=0.4)
    
    h_gap = [Line2D([0],[0], color=c, lw=1, label=f"{r:.2f}") for r,c in zip(rates, rate_colors_gap)]
    fig.legend(handles=h_gap, loc='center left', bbox_to_anchor=(0.12, y_row2),
               ncol=len(rates), frameon=False, fontsize=style['legend_fontsize'],
               columnspacing=0.8, handletextpad=0.4)

    fig.text(0.55, y_row1, "Outcome", ha='left', va='center', 
             fontsize=style['legend_fontsize']+1, fontweight='normal')

    h_go = [
        Line2D([0],[0], color='k', ls='-',  lw=1, label='Go Success'),
        Line2D([0],[0], color='k', ls='--', lw=1, label='Go Error'),
        Line2D([0],[0], color='k', ls=':',  lw=1, label='Go Missing'),
    ]
    fig.legend(handles=h_go, loc='center left', bbox_to_anchor=(0.64, y_row1),
               ncol=3, frameon=False, fontsize=style['legend_fontsize'],
               columnspacing=1.2, handletextpad=0.5)

    h_stop = [
        Line2D([0],[0], color='k', ls='-',  lw=1, label='Stop Success'),
        Line2D([0],[0], color='k', ls='--', lw=1, label='Stop Error'),
    ]
    fig.legend(handles=h_stop, loc='center left', bbox_to_anchor=(0.64, y_row2),
               ncol=2, frameon=False, fontsize=style['legend_fontsize'],
               columnspacing=1.2, handletextpad=0.5)

    plt.subplots_adjust(bottom=0.24, top=0.92, left=0.08, right=0.98)

    if saveto:
        fig.savefig(saveto, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()


def plot_policy_summary(model, out_dict, selected_timesteps, saveto=None, dpi=300, gamma=0.3):
    """
    Generates a publication-quality compact summary plot of the policy across selected timesteps.
    
    Args:
        model: The computational model containing value arrays.
        out_dict: Dictionary containing simulation outcomes.
        selected_timesteps: List of specific timesteps to visualize.
        saveto: Optional filepath to save the figure.
        dpi: Resolution for the saved figure.
        gamma: Gamma correction factor for visual clarity of probabilities.
    """
    
    # Configure PLOS-compliant typography and color palette
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    colors = {
        "Q_L": "#E3B23C",   # Gold (Left)
        "Q_R": "#3A6EA5",   # Blue (Right)
        "Q_W": "#F7F7F7",   # Light Gray (Wait)
    }
    
    rgb_L = np.array(to_rgb(colors["Q_L"]))
    rgb_R = np.array(to_rgb(colors["Q_R"]))
    rgb_W = np.array(to_rgb(colors["Q_W"]))

    style = {
        "title_fontsize": 11,
        "label_fontsize": 12, 
        "tick_fontsize": 9,
        "legend_fontsize": 10,
    }

    # Compute action probabilities using softmax and apply gamma correction
    limit = -1
    V_L = model.value_left[:limit, :, :]
    V_R = model.value_right[:limit, :, :]
    V_W = model.value_wait[:limit, :, :]
    
    Costs = np.stack([V_L, V_R, V_W], axis=-1)
    
    inv_temp = getattr(model, 'inv_temp', 1.0)
    logits = -Costs * inv_temp
    
    max_logits = np.nanmax(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.nansum(exp_logits, axis=-1, keepdims=True)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        probs = exp_logits / sum_exp
    probs = np.nan_to_num(probs, nan=0.0)

    probs_vis = probs ** gamma
    probs_vis = probs_vis / np.sum(probs_vis, axis=-1, keepdims=True)

    img_rgb_all = (probs_vis[..., 0:1] * rgb_L.reshape(1,1,1,3) + 
                   probs_vis[..., 1:2] * rgb_R.reshape(1,1,1,3) + 
                   probs_vis[..., 2:3] * rgb_W.reshape(1,1,1,3))
    img_rgb_all = np.clip(img_rgb_all, 0.0, 1.0)

    def get_belief_scatter(data, t):
        """Helper to extract belief states (zeta, beta) for a given timestep."""
        if data is None or len(data) == 0:
            return [], []
        betas, zetas = [], []
        for trial in data:
            b_seq = trial.get('beta_seq', [])
            z_seq = trial.get('sigma_seq', [])
            
            rt = trial.get('rt')
            if rt is None:
                rt = float('inf')
            
            if t < len(b_seq) and t < len(z_seq) and t <= rt:
                betas.append(b_seq[t])
                zetas.append(z_seq[t])
                
        return zetas, betas

    def draw_right_triangle_legend(fig, rect, gamma, colors_dict, angle_deg=135):
        """Helper to draw a rotated colormap legend with shifted axis ticks."""
        x_orig, y_orig, w_orig, h_orig = rect
        w_new, h_new = w_orig * 3.0, h_orig * 3.0
        x_new, y_new = x_orig - w_orig * 1.0, y_orig - h_orig * 1.0
        
        ax_leg = fig.add_axes([x_new, y_new, w_new, h_new])
        ax_leg.axis('off')
        
        n = 300
        x_vals = np.linspace(-0.5, 1.5, n)
        y_vals = np.linspace(-0.5, 1.5, n)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        cx, cy = 0.5, 0.5
        theta_inv = np.deg2rad(-angle_deg)
        c_inv, s_inv = np.cos(theta_inv), np.sin(theta_inv)
        
        X_orig = (X - cx) * c_inv - (Y - cy) * s_inv + cx
        Y_orig = (X - cx) * s_inv + (Y - cy) * c_inv + cy
        
        V_R = X_orig
        V_L = Y_orig
        V_W = 1.0 - X_orig - Y_orig
        
        valid = (V_R >= -0.001) & (V_L >= -0.001) & (V_W >= -0.001)
        V_R, V_L, V_W = np.clip(V_R, 0, 1), np.clip(V_L, 0, 1), np.clip(V_W, 0, 1)
        
        r_W = np.array(to_rgb(colors_dict["Q_W"]))
        r_L = np.array(to_rgb(colors_dict["Q_L"]))
        r_R = np.array(to_rgb(colors_dict["Q_R"]))
        
        img = np.zeros((n, n, 4))
        img[valid, :3] = V_W[valid, None]*r_W + V_L[valid, None]*r_L + V_R[valid, None]*r_R
        img[valid, 3] = 1.0 
        
        ax_leg.imshow(img, origin='lower', extent=[-0.5, 1.5, -0.5, 1.5], clip_on=False)
        
        def fwd_rot(px, py):
            t = np.deg2rad(angle_deg)
            return (px - cx) * np.cos(t) - (py - cy) * np.sin(t) + cx, \
                   (px - cx) * np.sin(t) + (py - cy) * np.cos(t) + cy

        def true_to_visual(p, g):
            if p == 0: return 0.0
            if p == 1: return 1.0
            return (p**g) / (p**g + (1-p)**g)

        tick_probs = [0, 0.01, 0.1, 0.5, 1.0]
        tick_pos = [true_to_visual(p, gamma) for p in tick_probs]
        tick_labels = ['0', '.01', '.1', '.5', '1']
        
        ax_leg.plot(*zip(fwd_rot(0,0), fwd_rot(1,0)), color='black', lw=1, clip_on=False)
        ax_leg.plot(*zip(fwd_rot(0,0), fwd_rot(0,1)), color='black', lw=1, clip_on=False)
        
        base_pad = -0.22
        
        for p, lbl in zip(tick_pos, tick_labels):
            # Go Right Axis (Top)
            tx, ty = fwd_rot(p, 0)
            dx, dy = fwd_rot(p, -0.04)
            ax_leg.plot([tx, dx], [ty, dy], color='black', lw=0.8, clip_on=False)
            
            p_shift = p
            if lbl == '.5':   p_shift += 0.07  
            if lbl == '.01':  p_shift -= 0.07  
            
            lx, ly = fwd_rot(p_shift, base_pad)
            ax_leg.text(lx, ly, lbl, rotation=0, ha='center', va='center', fontsize=8, fontstretch='ultra-condensed')
            
            # Go Left Axis (Bottom)
            tx2, ty2 = fwd_rot(0, p)
            dx2, dy2 = fwd_rot(-0.04, p)
            ax_leg.plot([tx2, dx2], [ty2, dy2], color='black', lw=0.8, clip_on=False)
            
            p_shift_2 = p
            if lbl == '.5':   p_shift_2 += 0.08  
            if lbl == '.01':  p_shift_2 -= 0.08  

            lx2, ly2 = fwd_rot(base_pad, p_shift_2)
            ax_leg.text(lx2, ly2, lbl, rotation=0, ha='center', va='center', fontsize=8, fontstretch='ultra-condensed')
                
        # Placement for outer peripheral text labels
        label_color = "#000000"
        ax_leg.text(0.5, 2, 'Softmax Policy', ha='left', va='bottom', fontsize=10, color=label_color) 
        ax_leg.text(0.5, 1.55, 'Go Right', ha='left', va='bottom', fontsize=9, color=label_color)   
        ax_leg.text(0.5, -0.55, 'Go Left', ha='left', va='top', fontsize=9, color=label_color)      
        ax_leg.text(0.27, 0.5, 'Prob.', rotation=90, ha='center', va='center', fontsize=9, color="#565555") 
        wx, wy = fwd_rot(0, 0)
        ax_leg.text(wx + 0.35, wy, 'Wait', ha='left', va='center', fontsize=9, color=label_color) 

    # Initialize dynamic grid layout based on outcomes and selected timesteps
    n_rows = len(out_dict)
    n_cols = len(selected_timesteps)
    outcome_names = list(out_dict.keys())
    
    go_keys = ["Go Success", "Go Error", "Go Missing"]
    stop_keys = ["Stop Success", "Stop Error"]
    
    total_go = sum([len(out_dict[k]) for k in go_keys if k in out_dict and out_dict[k] is not None])
    total_stop = sum([len(out_dict[k]) for k in stop_keys if k in out_dict and out_dict[k] is not None])
    
    fig_width = 7.5 
    fig_height = max(4.0, n_rows * 1.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi)
    
    if n_rows == 1: axes = np.expand_dims(axes, axis=0)
    if n_cols == 1: axes = np.expand_dims(axes, axis=1)

    for r in range(n_rows):
        out_name = outcome_names[r]
        out_data = out_dict[out_name]
        
        count = len(out_data) if out_data is not None else 0
        if out_name in go_keys:
            pct = (count / total_go * 100) if total_go > 0 else 0
        elif out_name in stop_keys:
            pct = (count / total_stop * 100) if total_stop > 0 else 0
        else:
            pct = 0.0
        
        for c in range(n_cols):
            t = selected_timesteps[c]
            ax = axes[r, c]
            
            if t < Costs.shape[0]:
                ax.imshow(img_rgb_all[t], extent=[0, 1, 0, 1], origin='lower', 
                          interpolation='nearest', aspect='equal')
                
                zetas_data, betas_data = get_belief_scatter(out_data, t)
                if zetas_data:
                    points = np.round(np.column_stack((zetas_data, betas_data)), 4)
                    unique_pts, counts = np.unique(points, axis=0, return_counts=True)
                    sizes = 2 + np.log1p(counts) * 8 
                    ax.scatter(unique_pts[:,0], unique_pts[:,1], 
                               c='#222222', s=sizes, alpha=0.75, 
                               edgecolors='none', zorder=10, clip_on=False)

            if r == 0:
                ax.set_title(f"Time {t+1}", fontsize=style['title_fontsize'], pad=8, fontweight='normal')
            
            if c == 0:
                row_label = f"{out_name}\n({pct:.1f}%)"
                ax.text(-0.95, 0.5, row_label, transform=ax.transAxes, 
                        ha='center', va='center', rotation=90, 
                        fontsize=style['label_fontsize'], fontweight='normal')

            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'], length=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if c != 0: ax.set_yticklabels([])
            else: ax.set_yticklabels(['0', '0.5', '1'])
                
            if r != n_rows - 1: ax.set_xticklabels([])
            else: ax.set_xticklabels(['0', '0.5', '1'])

    # Add global axis labels and legend placement
    fig.text(0.44, 0.05, r"Stop Signal Belief $\zeta^\nu_t$", ha='center', va='center', fontsize=style['label_fontsize'])
    fig.text(0.079, 0.5, r"Go Belief $\beta^\nu_t$", ha='center', va='center', rotation='vertical', fontsize=style['label_fontsize'])

    draw_right_triangle_legend(fig, [0.7, 0.65, 0.068, 0.068], gamma, colors, angle_deg=135)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#222222', markersize=6, alpha=0.75, 
               label='Belief States') 
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.68, 0.54), 
               frameon=False, fontsize=style['legend_fontsize'], handletextpad=1.5)
    
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.14, right=0.68, hspace=0.15, wspace=0.15)

    if saveto:
        fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()
        

def plot_policy(model, out_dict, outcome_name, saveto=None, dpi=300, gamma=0.3):
    """
    Generates a publication-quality scatter plot of the policy over time.
    
    Args:
        model: The computational model containing value arrays.
        out_dict: Dictionary containing simulation outcomes.
        outcome_name: The specific outcome key to plot (e.g., 'Go Success').
        saveto: Optional filepath to save the figure.
        dpi: Resolution for the saved figure.
        gamma: Gamma correction factor for visual clarity of probabilities.
    """
    
    # Configure PLOS-compliant typography and color palette
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    colors = {
        "Q_L": "#E3B23C",   # Gold (Left)
        "Q_R": "#3A6EA5",   # Blue (Right)
        "Q_W": "#F7F7F7",   # Light Gray (Wait)
    }
    
    rgb_L = np.array(to_rgb(colors["Q_L"]))
    rgb_R = np.array(to_rgb(colors["Q_R"]))
    rgb_W = np.array(to_rgb(colors["Q_W"]))

    style = {
        "title_fontsize": 10,
        "label_fontsize": 12, 
        "tick_fontsize": 8,
        "legend_fontsize": 10,
    }

    # Calculate the percentage of the selected outcome relative to its trial type
    go_keys = ["Go Success", "Go Error", "Go Missing"]
    stop_keys = ["Stop Success", "Stop Error"]
    
    total_go = sum([len(out_dict[k]) for k in go_keys if k in out_dict and out_dict[k] is not None])
    total_stop = sum([len(out_dict[k]) for k in stop_keys if k in out_dict and out_dict[k] is not None])
    
    out_data = out_dict.get(outcome_name, [])
    count = len(out_data) if out_data is not None else 0
    
    if outcome_name in go_keys:
        pct = (count / total_go * 100) if total_go > 0 else 0
    elif outcome_name in stop_keys:
        pct = (count / total_stop * 100) if total_stop > 0 else 0
    else:
        pct = 0.0

    # Compute action probabilities using softmax and apply gamma correction
    limit = -1
    V_L = model.value_left[:limit, :, :]
    V_R = model.value_right[:limit, :, :]
    V_W = model.value_wait[:limit, :, :]
    
    Costs = np.stack([V_L, V_R, V_W], axis=-1)
    num_timesteps = Costs.shape[0]

    inv_temp = getattr(model, 'inv_temp', 1.0)
    logits = -Costs * inv_temp
    
    max_logits = np.nanmax(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.nansum(exp_logits, axis=-1, keepdims=True)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        probs = exp_logits / sum_exp
    probs = np.nan_to_num(probs, nan=0.0)

    probs_vis = probs ** gamma
    probs_vis = probs_vis / np.sum(probs_vis, axis=-1, keepdims=True)

    img_rgb_all = (probs_vis[..., 0:1] * rgb_L.reshape(1,1,1,3) + 
                   probs_vis[..., 1:2] * rgb_R.reshape(1,1,1,3) + 
                   probs_vis[..., 2:3] * rgb_W.reshape(1,1,1,3))
    img_rgb_all = np.clip(img_rgb_all, 0.0, 1.0)

    def get_belief_scatter(data, t):
        """Helper to extract belief states (zeta, beta) for a given timestep."""
        if data is None or len(data) == 0:
            return [], []
        betas, zetas = [], []
        for trial in data:
            b_seq = trial.get('beta_seq', [])
            z_seq = trial.get('sigma_seq', [])
            
            rt = trial.get('rt')
            if rt is None:
                rt = float('inf')
            
            if t < len(b_seq) and t < len(z_seq) and t <= rt:
                betas.append(b_seq[t])
                zetas.append(z_seq[t])
                
        return zetas, betas

    def draw_right_triangle_legend(fig, rect, gamma, colors_dict, angle_deg=135):
        """Helper to draw a rotated colormap legend for the three-way policy."""
        x_orig, y_orig, w_orig, h_orig = rect
        w_new, h_new = w_orig * 3.0, h_orig * 3.0
        x_new, y_new = x_orig - w_orig * 1.0, y_orig - h_orig * 1.0
        
        ax_leg = fig.add_axes([x_new, y_new, w_new, h_new])
        ax_leg.axis('off')
        
        n = 300
        x_vals = np.linspace(-0.5, 1.5, n)
        y_vals = np.linspace(-0.5, 1.5, n)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        cx, cy = 0.5, 0.5
        theta_inv = np.deg2rad(-angle_deg)
        c_inv, s_inv = np.cos(theta_inv), np.sin(theta_inv)
        
        X_orig = (X - cx) * c_inv - (Y - cy) * s_inv + cx
        Y_orig = (X - cx) * s_inv + (Y - cy) * c_inv + cy
        
        V_R = X_orig
        V_L = Y_orig
        V_W = 1.0 - X_orig - Y_orig
        
        valid = (V_R >= -0.001) & (V_L >= -0.001) & (V_W >= -0.001)
        V_R, V_L, V_W = np.clip(V_R, 0, 1), np.clip(V_L, 0, 1), np.clip(V_W, 0, 1)
        
        r_W = np.array(to_rgb(colors_dict["Q_W"]))
        r_L = np.array(to_rgb(colors_dict["Q_L"]))
        r_R = np.array(to_rgb(colors_dict["Q_R"]))
        
        img = np.zeros((n, n, 4))
        img[valid, :3] = V_W[valid, None]*r_W + V_L[valid, None]*r_L + V_R[valid, None]*r_R
        img[valid, 3] = 1.0 
        
        ax_leg.imshow(img, origin='lower', extent=[-0.5, 1.5, -0.5, 1.5], clip_on=False)
        
        def fwd_rot(px, py):
            t = np.deg2rad(angle_deg)
            return (px - cx) * np.cos(t) - (py - cy) * np.sin(t) + cx, \
                   (px - cx) * np.sin(t) + (py - cy) * np.cos(t) + cy

        def true_to_visual(p, g):
            if p == 0: return 0.0
            if p == 1: return 1.0
            return (p**g) / (p**g + (1-p)**g)

        tick_probs = [0, 0.01, 0.1, 0.5, 1.0]
        tick_pos = [true_to_visual(p, gamma) for p in tick_probs]
        tick_labels = ['0', '.01', '.1', '.5', '1']
        
        ax_leg.plot(*zip(fwd_rot(0,0), fwd_rot(1,0)), color='black', lw=1, clip_on=False)
        ax_leg.plot(*zip(fwd_rot(0,0), fwd_rot(0,1)), color='black', lw=1, clip_on=False)
        
        base_pad = -0.22
        
        for p, lbl in zip(tick_pos, tick_labels):
            tx, ty = fwd_rot(p, 0)
            dx, dy = fwd_rot(p, -0.04)
            ax_leg.plot([tx, dx], [ty, dy], color='black', lw=0.8, clip_on=False)
            
            p_shift = p
            if lbl == '.5':   p_shift += 0.07  
            if lbl == '.01':  p_shift -= 0.07  
            
            lx, ly = fwd_rot(p_shift, base_pad)
            ax_leg.text(lx, ly, lbl, rotation=0, ha='center', va='center', fontsize=8, fontstretch='ultra-condensed')
            
            tx2, ty2 = fwd_rot(0, p)
            dx2, dy2 = fwd_rot(-0.04, p)
            ax_leg.plot([tx2, dx2], [ty2, dy2], color='black', lw=0.8, clip_on=False)
            
            p_shift_2 = p
            if lbl == '.5':   p_shift_2 += 0.08  
            if lbl == '.01':  p_shift_2 -= 0.08  

            lx2, ly2 = fwd_rot(base_pad, p_shift_2)
            ax_leg.text(lx2, ly2, lbl, rotation=0, ha='center', va='center', fontsize=8, fontstretch='ultra-condensed')
                
        label_color = "#000000"
        offset_right = 0.4
        
        gx_r, gy_r = fwd_rot(1, 0)
        ax_leg.text(gx_r + offset_right, gy_r + 0.15, 'Go Right', ha='left', va='center', fontsize=9, color=label_color)
        
        gx_l, gy_l = fwd_rot(0, 1)
        ax_leg.text(gx_l + offset_right, gy_l - 0.15, 'Go Left', ha='left', va='center', fontsize=9, color=label_color)
        
        wx, wy = fwd_rot(0, 0)
        ax_leg.text(wx + offset_right, wy, 'Wait', ha='left', va='center', fontsize=9, color=label_color) 
        
        ax_leg.text(0.27, 0.5, 'Prob.', rotation=90, ha='center', va='center', fontsize=9, color="#565555") 

    # Initialize grid layout for timesteps
    cols = 8
    rows = (num_timesteps + cols - 1) // cols
    
    fig_width = cols * 1.2
    fig_height = rows * 1.3 + 1.8  
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi)
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        
        if i < num_timesteps:
            ax.imshow(img_rgb_all[i], extent=[0, 1, 0, 1], origin='lower', 
                      interpolation='nearest', aspect='equal')
            
            zetas_data, betas_data = get_belief_scatter(out_data, i)
            if zetas_data:
                points = np.round(np.column_stack((zetas_data, betas_data)), 4)
                unique_pts, counts = np.unique(points, axis=0, return_counts=True)
                
                sizes = 2 + np.log1p(counts) * 6
                ax.scatter(unique_pts[:,0], unique_pts[:,1], 
                           c='#222222', s=sizes, alpha=0.75, 
                           edgecolors='none', zorder=10, clip_on=False)

            ax.set_title(f"Time {i+1}", fontsize=style['tick_fontsize'], pad=4)
            
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0', '0.5', '1'])
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '0.5', '1'])
            ax.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'], length=2)
            
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            row_idx = i // cols
            col_idx = i % cols
            if col_idx != 0: ax.set_yticklabels([])
            if row_idx != rows - 1: ax.set_xticklabels([])
        else:
            ax.axis("off")

    # Add global axis labels and legend placement
    fig.text(0.5, 0.02, r"Stop Signal Belief $\zeta^\nu_t$", ha='center', va='center', fontsize=style['label_fontsize'])
    fig.text(0.04, 0.5, r"Go Belief $\beta^\nu_t$", ha='center', va='center', rotation='vertical', fontsize=style['label_fontsize'])

    base_y = 0.81
    leg_w = 0.05
    leg_h = leg_w * (fig_width / fig_height)
    
    draw_right_triangle_legend(fig, [0.1, base_y - leg_h/2, leg_w, leg_h], gamma, colors, angle_deg=135)
    fig.text(0.27, base_y, 'Softmax Policy', ha='left', va='center', fontsize=10, color='#000000')

    label_str = f'Belief States ({outcome_name}, {pct:.1f}%)'
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#222222', markersize=6, alpha=0.75, 
               label=label_str)
    ]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.58, base_y), 
               frameon=False, fontsize=style['legend_fontsize'])

    plt.subplots_adjust(top=0.68, bottom=0.08, left=0.08, right=0.95, hspace=0.45, wspace=0.15)

    if saveto:
        fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()




