"""
preprocessing.py

Data Preprocessing Module for the ABCD Stop-Signal Task (SST).

This script takes raw trial-level data (straight from the ABCD study format) and 
transforms it into the clean, standardized format required for the POMDP simulator 
and the neural network encoders. 

Key Transformations:
1. Categorizes trials into mutually exclusive behavioral outcomes (GS, GE, GM, SS, SE).
2. Discretizes continuous time variables (RT and SSD) into 25ms bins for the simulator.
3. Maps raw text strings into categorical state variables and binary responses.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

# CONSTANTS
# Resolution for discretizing time (milliseconds per simulation step)
TIME_STEP_MS = 25  


# MAIN PREPROCESSING FUNCTION

def preprocessing(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Preprocesses the raw ABCD SST trial-level data.

    Args:
        file_path (Union[str, Path]): Path to the raw subject CSV file.

    Returns:
        pd.DataFrame: A cleaned dataframe containing only the necessary columns 
        for modeling and inference:
            ['true_go_state', 'true_stop_state', 'ssd', 'ssd_real', 
             'rt', 'rt_real', 'rt_gs_real', 'lg_rt_real', 'lg_rt_gs_real', 
             'result', 'sequence', 'response']
    """
    # Ensure file_path is handled correctly by pandas
    df = pd.read_csv(str(file_path), low_memory=False)

    # --- 1. Categorize Trial Outcomes ---
    
    # GS (Go Success): Correct Go Trial, Valid RT (<= 1000ms)
    is_gs = (df['sst_expcon'] == 'GoTrial') & \
            (df['sst_choiceacc'] == 1) & \
            (df['sst_primaryrt'] <= 1000)

    # GE (Go Error): Incorrect Go Trial, Valid RT
    is_ge = (df['sst_expcon'] == 'GoTrial') & \
            (df['sst_choiceacc'] == 0) & \
            (df['sst_primaryrt'] <= 1000)

    # GM (Go Miss): Go Trial with no valid response
    # Note: Response codes 1 or 2 indicate a button press. If neither, it's a miss.
    is_gm = (df['sst_expcon'] == 'GoTrial') & \
            (df['sst_go_resp'] != 1) & \
            (df['sst_go_resp'] != 2)

    # SS (Stop Success): Stop Trial with successful motor inhibition
    is_ss = (df['sst_expcon'] == 'VariableStopTrial') & \
            (df['sst_inhibitacc'] == 1)

    # SE (Stop Error): Stop Trial with failed motor inhibition
    is_se = (df['sst_expcon'] == 'VariableStopTrial') & \
            (df['sst_inhibitacc'] == 0)

    # Consolidate masks into a single 'result' categorical column
    conditions = [is_gs, is_ge, is_gm, is_ss, is_se]
    choices = ['GS', 'GE', 'GM', 'SS', 'SE']
    df['result'] = np.select(conditions, choices, default='nan')

    # --- 2. Process Time Variables (RT & SSD) ---
    
    # Standardize column names for raw time (milliseconds)
    df = df.rename(columns={'sst_ssd_dur': 'ssd_real'})
    df['rt_real'] = df['sst_primaryrt']

    # Discretize time into steps for the POMDP simulator
    # df['ssd'] = df['ssd_real'] // TIME_STEP_MS
    # df['rt'] = df['rt_real'] // TIME_STEP_MS
    df['ssd'] = np.round(df['ssd_real'] / TIME_STEP_MS)
    df['rt'] = np.round(df['rt_real'] / TIME_STEP_MS)

    # Isolate real Reaction Times specifically for Go Success trials
    df["rt_gs_real"] = np.where(df["result"] == "GS", df["rt_real"], np.nan)

    # Create log-transformed RT variables (replacing 0s with NaN to avoid -inf)
    df['lg_rt_real'] = np.log(df['rt_real'].replace(0, np.nan))
    df['lg_rt_gs_real'] = np.log(df['rt_gs_real'].replace(0, np.nan))

    # --- 3. Encode Task States ---
    
    # Trial Type: non-stop (Go) vs stop (Stop)
    df['true_stop_state'] = df['sst_expcon'].map({
        'GoTrial': 'nonstop',
        'VariableStopTrial': 'stop'
    })

    # Stimulus Direction
    df['true_go_state'] = df['sst_stim'].map({
        'left_arrow': 'left', 
        'right_arrow': 'right'
    })

    # Sequence ID: 0 for Go Trials, 1 for Stop Trials
    df['sequence'] = df['sst_expcon'].map({
        'GoTrial': 0, 
        'VariableStopTrial': 1
    })

    # Binary Motor Response: 1 = Responded (Action), 0 = No Response (Inhibited/Missed)
    df['response'] = df['result'].map({
        'GS': 1, 'GE': 1, 'SE': 1, 
        'GM': 0, 'SS': 0
    })

    # --- 4. Final Output Formatting ---
    
    columns_to_keep = [
        'true_go_state', 
        'true_stop_state', 
        'ssd', 
        'ssd_real', 
        'rt', 
        'rt_real', 
        'rt_gs_real', 
        'lg_rt_real', 
        'lg_rt_gs_real',
        'result', 
        'sequence', 
        'response'
    ]
    
    return df[columns_to_keep]