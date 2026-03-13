from collections import Counter, defaultdict
from matplotlib import pyplot as plt
import numpy as np


from collections import Counter


def count_go_trials(sequence):
    """
    Count the number of consecutive go trials (0s) before each stop trial (1), 
    resetting the counter at trial 180 (i.e., second run).

    Args:
        sequence (list): A list of integers (0s and 1s) representing go and stop trials.

    Returns:
        dict: Sorted dictionary {go_trial_count: frequency}

     Example:
        sequence = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        response = [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
        rt_real = [100, 122, np.nan, 200, 100, 122, 150, 200, np.nan, 122, 150]
        go_trials = count_go_trials(sequence)
        print(go_trials)
        # Output: {2: 2, 4: 1}
    """
    go_counts = []
    count = 0

    for n, trial in enumerate(sequence):
        if n == 179:  # Start of second run
            count = 0

        if trial == 0:
            count += 1
        elif trial == 1:
            if count > 0:
                go_counts.append(count)
            count = 0

    # Count and sort the occurrences
    go_trials = Counter(go_counts)
    return dict(sorted(go_trials.items()))


def count_go_responses(sequence, response):
    """
    Count the number of go (1) responses for each go trial count before a stop trial (1),
    resetting the counter at trial 180 (i.e., when index == 179).

    Args:
        sequence (list): A list of integers (0s and 1s) representing go and stop trials.
        response (list): A list of integers (0s and 1s) representing responses (1 for go, 0 for no-go).

    Returns:
        dict: A dictionary where keys are go trial counts (int), 
              and values are lists of go response counts (list[int]).

    Example:
        sequence = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        response = [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
        go_responses = count_go_responses(sequence, response)
        print(go_responses)
        # First stop (index 4): previous go trial count = 4, responses = [1, 1, 0, 1] → sum = 3
        # Second stop (index 7): go count = 2, responses = [1, 1] → sum = 2
        # Third stop (index 10): go count = 2, responses = [0, 1] → sum = 1
        # Output: {2: [2, 1], 4: [3]}
    """
    assert len(sequence) == len(
        response), "Sequence and response must be the same length"

    go_responses = defaultdict(list)
    count = 0
    go_response_window = []

    for n, (seq_trial, res_trial) in enumerate(zip(sequence, response)):
        # Trial index 179 is the 180th trial → reset here
        if n == 179:
            count = 0
            go_response_window = []

        if seq_trial == 0:
            count += 1
            go_response_window.append(res_trial)
        elif seq_trial == 1:
            if count > 0:
                go_response_count = sum(go_response_window)
                go_responses[count].append(go_response_count)

            # Reset after processing a stop trial
            count = 0
            go_response_window = []

    return dict(sorted(go_responses.items()))


def count_last_go_responses(sequence, response):
    """
    Count the number of go (1) responses for only the last go trial before each stop trial (1),
    resetting the counter at trial 180 (i.e., when index == 179).

    Args:
        sequence (list): A list of integers (0s and 1s) representing go and stop trials.
        response (list): A list of integers (0s and 1s) representing responses (1 for go, 0 for no-go).

    Returns:
        dict: A dictionary where keys are go trial counts and values are the count of go responses 
              only from the last go trial before a stop trial.

    Example:
        sequence = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        response = [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
        last_go_responses = count_last_go_responses(sequence, response)
        print(last_go_responses)
        # First stop (index 4): go count = 4 → last response = 1
        # Second stop (index 7): go count = 2 → last response = 1
        # Third stop (index 10): go count = 2 → last response = 1
        # Output: {2: [1, 1], 4: [1]}
    """
    from collections import defaultdict

    last_go_responses = defaultdict(list)
    count = 0
    last_go_response = 0

    for n, (seq_trial, res_trial) in enumerate(zip(sequence, response)):
        if n == 179:  # Index 179 == trial 180 → reset for second run
            count = 0
            last_go_response = 0

        if seq_trial == 0:
            count += 1
            last_go_response = res_trial
        elif seq_trial == 1 and count > 0:
            last_go_responses[count].append(last_go_response)
            count = 0
            last_go_response = 0

    return dict(sorted(last_go_responses.items()))


def record_go_rt(sequence, rt_real):
    """
    Record the reaction times (RTs) as a list for each go trial count window before a stop trial (1),
    resetting at trial 180 (index 179).

    Args:
        sequence (list): A list of integers (0s and 1s) representing go and stop trials.
        rt_real (list): A list of reaction times corresponding to each trial.

    Returns:
        dict: A dictionary where keys are go trial counts, and values are lists of RT sequences 
              recorded before each stop trial.

    Example:
        sequence = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        rt_real  = [100,122,nan,200,100,122,150,200,nan,122,150]
        result = record_go_rt(sequence, rt_real)
        print(result)
        # Output: {2: [[122, 150], [nan, 122]], 4: [[100, 122, nan, 200]]}
    """
    go_rt = defaultdict(list)
    count = 0
    go_rt_window = []

    for n, (seq_trial, rt) in enumerate(zip(sequence, rt_real)):
        if n == 179:  # Index 179 == trial 180 → reset for second run
            count = 0
            go_rt_window = []

        if seq_trial == 0:
            count += 1
            go_rt_window.append(rt)
        elif seq_trial == 1 and count > 0:
            go_rt[count].append(go_rt_window.copy())
            count = 0
            go_rt_window = []

    return dict(sorted(go_rt.items()))


def record_last_go_rt(sequence, rt_real):
    """
    Record the reaction time (RT) of the last go trial before each stop trial (1),
    resetting at trial 180 (index 179).

    Args:
        sequence (list): A list of integers (0s and 1s) representing go and stop trials.
        rt_real (list): A list of reaction times corresponding to each trial.

    Returns:
        dict: A dictionary where keys are go trial counts, and values are lists of last RTs
              before each stop trial.

    Example:
        sequence = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        rt_real = [100, 122, np.nan, 200, 100, 122, 150, 200, np.nan, 122, 150]
        last_go_rt = record_last_go_rt(sequence, rt_real)
        print(last_go_rt)
        # Output: {2: [150, 122], 4: [200]}
    """
    last_go_rt = defaultdict(list)
    count = 0
    last_go_rt_num = 0

    for n, (seq_trial, rt) in enumerate(zip(sequence, rt_real)):
        if n == 179:  # Trial 180 starts here → reset
            count = 0
            last_go_rt_num = 0

        if seq_trial == 0:
            count += 1
            last_go_rt_num = rt  # Always overwrite with latest go RT
        elif seq_trial == 1 and count > 0:
            last_go_rt[count].append(last_go_rt_num)
            count = 0
            last_go_rt_num = 0

    return dict(sorted(last_go_rt.items()))


def record_r_pred(sequence, r_pred_seq):
    """
    Record predicted r values in each go trial window before a stop trial (1),
    resetting at trial 180 (index 179).

    Args:
        sequence (list): A list of integers (0s and 1s) representing go and stop trials.
        r_pred_seq (list): A list of predicted r values corresponding to each trial.

    Returns:
        dict: A dictionary where keys are go trial counts, and values are lists of predicted r value sequences.

    Example:
        sequence = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        r_pred_seq = [0.45, 0.67, 0.89, 0.55, 0.50, 0.60, 0.75, 0.80, 0.65, 0.70, 0.85]
        # Output: {2: [[0.6, 0.75], [0.65, 0.7]], 4: [[0.45, 0.67, 0.89, 0.55]]}
    """
    r_pred_dict = defaultdict(list)
    count = 0
    r_pred_window = []

    for n, (seq_trial, r_val) in enumerate(zip(sequence, r_pred_seq)):
        if n == 179:  # Trial 180 starts here
            count = 0
            r_pred_window = []

        if seq_trial == 0:
            count += 1
            r_pred_window.append(r_val)
        elif seq_trial == 1 and count > 0:
            r_pred_dict[count].append(r_pred_window.copy())
            count = 0
            r_pred_window = []

    return dict(sorted(r_pred_dict.items()))

# def summarize_per_go_count(data_dict, summary_func=np.mean):
#     """
#     Converts {go_count: [[...], [...], ...]} to {go_count: mean/other_statistic}.
#     """
#     return {
#         k: summary_func([item for sublist in v for item in sublist if not np.isnan(item)])
#         for k, v in data_dict.items()
#     }


def record_last_r_pred(sequence, r_pred_seq):
    """
    Record the predicted r value (r_pred) for each go trial count before a stop trial (1),
    resetting at trial 180 (index 179).

    Args:
        sequence (list): A list of integers (0s and 1s) representing go and stop trials.
        r_pred_seq (list): A list of predicted r values corresponding to each trial.

    Returns:
        dict: Keys are go trial counts, values are predicted r values from the last go trial
              before each stop trial.

    Example:
        sequence = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        r_pred_seq = [0.45, 0.67, 0.89, 0.55, 0.50, 0.60, 0.75, 0.80, 0.65, 0.70, 0.85]
        # Output: {4: [0.55], 2: [0.75, 0.70]}
    """
    last_r_pred_dict = defaultdict(list)
    count = 0
    last_r_pred_value = 0

    for n, (seq_trial, r_val) in enumerate(zip(sequence, r_pred_seq)):
        if n == 179:  # Trial 180 starts → reset second run
            count = 0
            last_r_pred_value = 0

        if seq_trial == 0:
            count += 1
            last_r_pred_value = r_val
        elif seq_trial == 1 and count > 0:
            last_r_pred_dict[count].append(last_r_pred_value)
            count = 0
            last_r_pred_value = 0

    return dict(sorted(last_r_pred_dict.items()))


def plot_bar(data, text="Value", save_path=None):
    """
    Plot a bar chart from a dictionary.

    Args:
        data (dict): Keys as x-axis values, values as y-axis values (can be list or scalar).
        text (str): Y-axis label and title text.
        save_path (str or None): If provided, the figure will be saved to this path.

    Example:
        data = {2: 2, 4: 1}
        plot_bar(data, "Frequency")
    """
    # Sort the data
    data = dict(sorted(data.items()))

    # If values are lists, take mean
    example_val = next(iter(data.values()))
    if isinstance(example_val, list):
        y_vals = [np.nanmean([v for v in vals if not (isinstance(
            v, float) and np.isnan(v))]) for vals in data.values()]
    else:
        y_vals = list(data.values())

    x_vals = list(data.keys())

    plt.figure(figsize=(8, 5))
    plt.bar(x_vals, y_vals, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("Number of Go Trials before Stop Trial")
    plt.ylabel(text)
    plt.title(f"{text} vs. Go Trials Before Stop Trial")
    plt.xticks(range(min(x_vals), max(x_vals)+1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_box(data, text="Value"):
    """
    Plot a boxplot from a dictionary where keys are x-axis labels (int),
    and values are lists of numeric values.

    Example:
        data = {2: [122, 150], 4: [100, 122, 200]}
        plot_box(data, "Reaction Time")
    """
    keys = list(range(min(data.keys()), max(data.keys()) + 1))

    values = []
    for key in keys:
        vals = data.get(key, [np.nan])
        # Optionally clean out None
        values.append([v for v in vals if v is not None])

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, patch_artist=True, showfliers=False)

    plt.xlabel("Number of Go Trials before Stop Trial")
    plt.ylabel(text)
    plt.title(f"{text} vs. Go Trials Before Stop Trial")
    plt.xticks(ticks=range(1, len(keys)+1), labels=keys)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_box_nested(data, text="Value"):
    """
    Plot a boxplot from a nested dictionary where each key represents the number of consecutive
    go trials before a stop trial, and each value is a list of lists representing trial-wise
    values (e.g., RTs or predictions) for **all** go trials in that sequence (not just the last one).

    This function flattens the nested lists to show the distribution of all go-trial values
    (e.g., all RTs) that appeared before each stop trial, regardless of their position.

    Args:
        data (dict): A nested dictionary like {2: [[122, 150], [nan, 122]], 4: [[100, 122, nan, 200]]}
                     where outer keys are go trial counts, inner lists are sequences of values
                     across each go window before a stop.
        text (str): Label for the y-axis and plot title.

    Example:
        plot_box_nested({2: [[122, 150], [nan, 122]], 4: [[100, 122, nan, 200]]}, "Reaction Time")
    """
    keys = list(range(min(data.keys()), max(data.keys()) + 1))
    flattened_data = {}

    for key, nested_lists in data.items():
        flat_list = [
            v for sublist in nested_lists for v in sublist
            if not (isinstance(v, float) and np.isnan(v))
        ]
        flattened_data[key] = flat_list if flat_list else [np.nan]

    values = [flattened_data.get(key, [np.nan]) for key in keys]

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, patch_artist=True, showfliers=False)
    plt.xlabel("Number of Go Trials before Stop Trial")
    plt.ylabel(text)
    plt.title(f"{text} vs. Go Trials Before Stop Trial")
    plt.xticks(ticks=range(1, len(keys) + 1), labels=keys)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_r_pred(r_pred_seq, sequence):
    """
    Visualize predicted r values over trials, with go/stop trials marked.

    Args:
        r_pred_seq (list or np.array): Sequence of predicted r values.
        sequence (list or np.array): Sequence of trial types (0 for go, 1 for stop).

    Plot:
        - r_pred values over time
        - green dots: go trials
        - red dots: stop trials
    """
    import numpy as np
    import matplotlib.pyplot as plt

    sequence = np.array(sequence)
    r_pred_seq = np.array(r_pred_seq)

    plt.figure(figsize=(20, 3))

    go_idx = np.where(sequence == 0)[0]
    stop_idx = np.where(sequence == 1)[0]

    plt.plot(r_pred_seq, label='r_pred', linewidth=1)
    plt.scatter(go_idx, r_pred_seq[go_idx],
                color='green', marker='.', label='Go Trials')
    plt.scatter(stop_idx, r_pred_seq[stop_idx],
                color='red', marker='.', label='Stop Trials')

    plt.xlabel("Trial Index")
    plt.ylabel("r = p(stop)")
    plt.title("Predicted Stop Probability (r_pred) Over Trials")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def collect_rt_by_go_run(sequence, rt_real):
    """
    收集每个 Go trial 的反应时，并根据它当前的连续 Go 序列长度进行分组。
    
    Args:
        sequence (list): 0 代表 Go, 1 代表 Stop。
        rt_real (list): 对应的反应时。
        
    Returns:
        dict: {go_run_length (int): [rt_values...]}
        例如: {1: [500, 520, ...], 2: [480, 490, ...]}
    """
    from collections import defaultdict
    import numpy as np
    
    go_run_data = defaultdict(list)
    count = 0
    
    for n, (trial, rt) in enumerate(zip(sequence, rt_real)):
        # ABCD 数据通常在第 180 个 trial (index 179) 结束第一个 run
        # 在进入第二个 run (index 180) 前重置计数
        if n == 180: 
            count = 0
            
        if trial == 0:  # Go Trial
            count += 1
            # 记录当前的序列长度和对应的 RT
            # 只有当 rt 是有效数值时才记录
            if not (isinstance(rt, float) and np.isnan(rt)) and rt > 0:
                go_run_data[count].append(rt)
        else:  # Stop Trial
            count = 0
            
    return go_run_data