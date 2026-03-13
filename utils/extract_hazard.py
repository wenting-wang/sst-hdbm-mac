import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================
# Input files
LIST_FILE = '/Users/w/sst-abcd-adhd-mac/sst-hdbm-main/stats_beh_full.csv'
DATA_ROOT = '/Users/w/Desktop/data/sst_valid_base'

# Output files
OUTPUT_RAW_CSV = 'go_run_hazard_raw_data.csv'      # Raw go_run events
OUTPUT_ORDER_CSV = 'go_run_hazard_per_order.csv'   # Hazard stats per order
OUTPUT_TOTAL_CSV = 'go_run_hazard_combined.csv'    # Hazard stats combined
OUTPUT_PLOT = 'go_run_hazard_plot.png'             # Visualization

# Plotting Settings
MIN_EVENTS_FOR_PLOT = 50  # Only plot orders with at least this many go_run events
# ===============================================

def get_go_run_lengths_from_file(file_path, filename):
    """
    Parses a subject file and extracts the length of consecutive Go trials
    immediately preceding a Stop trial.
    E.g., Sequence: G, G, G, S -> Returns a go_run of length 3.
    """
    try:
        df = pd.read_csv(file_path)
        
        # 1. Define Run IDs (assuming 360 trials split into 2 runs)
        run_ids = np.zeros(len(df), dtype=int)
        if len(df) >= 360:
            run_ids[:180] = 1
            run_ids[180:] = 2
        else:
            run_ids[:] = 1 
        df['manual_run'] = run_ids
        
        # 2. Identify Trial Types
        df['is_go_stim'] = df['sst_expcon'].astype(str).str.strip() == 'GoTrial'
        df['is_stop_stim'] = ~df['is_go_stim'] 

        # 3. Define Order Signature
        expcon_list = df['sst_expcon'].apply(lambda x: 0 if str(x).strip() == 'GoTrial' else 1).tolist()
        sig = tuple(expcon_list)

        extracted_go_runs = []

        # 4. Extract go_runs per Run
        for run_id, run_df in df.groupby('manual_run'):
            run_df = run_df.copy()
            run_df['type_change'] = (run_df['is_go_stim'] != run_df['is_go_stim'].shift()).cumsum()
            blocks = list(run_df.groupby('type_change'))
            
            for i in range(len(blocks) - 1):
                curr_id, curr_data = blocks[i]
                next_id, next_data = blocks[i+1]
                
                # Check if Current is GO and Next is STOP
                if curr_data['is_go_stim'].all() and next_data['is_stop_stim'].all():
                    go_run_len = len(curr_data)
                    extracted_go_runs.append({
                        'filename': filename,
                        'run_id': run_id,
                        'go_run_length': go_run_len,
                    })

        return sig, extracted_go_runs

    except Exception as e:
        return None, None

def calculate_hazard_function(df_input):
    """
    Calculates the discrete hazard function.
    Hazard(t) = (go_runs ending at t) / (go_runs reaching at least t)
    """
    # Count occurrences of each go_run length
    # This represents 'd_t': number of events (Stops) that happened exactly after t Gos
    counts = df_input['go_run_length'].value_counts().sort_index()
    
    # We need to compute 'n_t': number of go_runs at risk (reached at least t)
    # n_t = sum(counts[k]) for all k >= t
    max_go_run = counts.index.max()
    possible_lengths = range(1, max_go_run + 1)
    
    hazard_data = []
    
    for t in possible_lengths:
        if t not in counts.index:
            d_t = 0
        else:
            d_t = counts[t]
            
        # Risk set: all go_runs that are >= t
        # (i.e. they survived t-1 Go trials and were candidates to stop at t)
        n_t = df_input[df_input['go_run_length'] >= t].shape[0]
        
        if n_t == 0:
            h_t = 0
            se = 0
        else:
            h_t = d_t / n_t
            # Standard Error (Greenwood's formula approximation for simple proportion)
            se = np.sqrt((h_t * (1 - h_t)) / n_t)
            
        hazard_data.append({
            'go_run_length': t,
            'n_at_risk': n_t,
            'n_events': d_t,
            'hazard_prob': h_t,
            'se': se
        })
        
    return pd.DataFrame(hazard_data)

def main():
    # --- STEP 1: EXTRACT DATA ---
    print(">>> STEP 1: Scanning files for go_run Lengths...")
    
    try:
        df_list = pd.read_csv(LIST_FILE)
        file_list = df_list['filename'].tolist() if 'filename' in df_list.columns else df_list.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Failed to read list file: {e}")
        return

    all_data = []
    order_map = {}
    order_counter = 1
    total_files = len(file_list)
    
    for i, fname in enumerate(file_list):
        full_path = os.path.join(DATA_ROOT, fname)
        if not os.path.exists(full_path):
             full_path = os.path.join(DATA_ROOT, 'SST/baseline_year_1_arm_1', fname)
             if not os.path.exists(full_path): continue
        
        sig, go_runs = get_go_run_lengths_from_file(full_path, fname)
        
        if sig and go_runs:
            if sig not in order_map:
                order_map[sig] = order_counter
                order_counter += 1
            order_id = order_map[sig]
            for record in go_runs:
                record['order_id'] = order_id
                all_data.append(record)
        
        if (i+1) % 500 == 0: print(f"Processed {i+1}...")

    if not all_data:
        print("No valid data extracted.")
        return

    df_raw = pd.DataFrame(all_data)
    df_raw.to_csv(OUTPUT_RAW_CSV, index=False)
    print(f"\nRaw data saved to: {OUTPUT_RAW_CSV} ({len(df_raw)} events)")

    # --- STEP 2: CALCULATE HAZARD PER ORDER ---
    print("\n>>> STEP 2: Calculating Hazard per Order...")
    
    hazard_dfs = []
    
    # Get list of orders sorted by data volume
    order_counts = df_raw['order_id'].value_counts()
    
    for order_id in order_counts.index:
        sub_df = df_raw[df_raw['order_id'] == order_id]
        h_df = calculate_hazard_function(sub_df)
        h_df['order_id'] = order_id
        h_df['total_events_in_order'] = len(sub_df)
        hazard_dfs.append(h_df)
    
    if hazard_dfs:
        df_order_hazard = pd.concat(hazard_dfs, ignore_index=True)
        # Reorder columns
        cols = ['order_id', 'total_events_in_order', 'go_run_length', 'n_at_risk', 'n_events', 'hazard_prob', 'se']
        df_order_hazard = df_order_hazard[cols]
        df_order_hazard.to_csv(OUTPUT_ORDER_CSV, index=False)
        print(f"Order-level hazard saved to: {OUTPUT_ORDER_CSV}")

    # --- STEP 3: CALCULATE COMBINED HAZARD ---
    print("\n>>> STEP 3: Calculating Combined Hazard...")
    df_total_hazard = calculate_hazard_function(df_raw)
    df_total_hazard.to_csv(OUTPUT_TOTAL_CSV, index=False)
    print(f"Combined hazard saved to: {OUTPUT_TOTAL_CSV}")
    
    # --- STEP 4: PLOTTING ---
    print("\n>>> STEP 4: Plotting Hazard Functions...")
    plt.figure(figsize=(10, 6))
    
    # Plot individual orders (Top N or all valid ones)
    plotted_count = 0
    # Plot orders with enough data to be smooth-ish
    valid_orders = order_counts[order_counts >= MIN_EVENTS_FOR_PLOT].index
    
    # Use a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_orders)))
    
    for idx, order_id in enumerate(valid_orders):
        data = df_order_hazard[df_order_hazard['order_id'] == order_id]
        # Filter out extreme tails for plotting clarity if N is small at tail
        data = data[data['n_at_risk'] > 5] 
        
        if not data.empty:
            plt.plot(data['go_run_length'], data['hazard_prob'], 
                     marker='', linestyle='-', alpha=0.3, color=colors[idx], linewidth=1)
            plotted_count += 1
            
    # Plot Combined (Thick Line)
    # Filter combined tail noise
    total_plot = df_total_hazard[df_total_hazard['n_at_risk'] > 10]
    plt.plot(total_plot['go_run_length'], total_plot['hazard_prob'], 
             marker='o', linestyle='-', color='black', linewidth=3, label='Combined (All Subjects)')
    
    # Formatting
    plt.title(f'SST Stop Hazard Function\n(Prob. of Stop given go_run Length)\nIndividual Orders (N={plotted_count}) vs Combined', fontsize=14)
    plt.xlabel('go_run Length (Consecutive Go Trials)', fontsize=12)
    plt.ylabel('Hazard Probability P(Stop | go_run)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved to: {OUTPUT_PLOT}")
    print("Done!")

if __name__ == "__main__":
    main()