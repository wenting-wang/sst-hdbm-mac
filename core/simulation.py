import numpy as np


def simu_trial(model, true_go_state, true_stop_state, ssd, verbose=False):
    """
    Simulate a single trial of the Stop Signal Task (SST).

    Args:
        model: The computational model (pomdp).
        true_go_state (str): The direction of the go cue ('left' or 'right').
        true_stop_state (str): The type of trial ('nonstop' or 'stop').
        ssd (int): The stop signal delay in time steps. Use -1 for go ('nonstop') trials.
        verbose (bool): If True, returns detailed sequences of beliefs and values.

    Returns:
        dict or tuple: A dictionary of detailed trial sequences if verbose is True, 
        otherwise a tuple containing (result, reaction_time).
    """
    def generate_obs_sequences(true_go_state, true_stop_state, ssd):
        """Generate observation sequences for go and stop signals."""
        # Stop trial
        if true_stop_state == 'stop':
            # Ensure ssd is a valid positive integer for stop trials
            if ssd == -1 or ssd is None or np.isnan(ssd):
                raise ValueError("A valid SSD (>= 0) must be specified for stop trials.")
            ssd = int(ssd)

            y_pre = np.random.choice([0, 1, 2], ssd, p=model.p_y_z0_pre)
            y_on = np.random.choice([0, 1, 2], mu, p=model.p_y_z1_on)
            y_post = np.random.choice([0, 1, 2], max(
                horizon - ssd - mu, 0), p=model.p_y_z0_post)
            y_seq = np.concatenate((y_pre, y_on, y_post))

            p_x_z0, p_x_z1 = {"left": (model.p_x_dL_z0, model.p_x_dL_z1),
                              "right": (model.p_x_dR_z0, model.p_x_dR_z1)}[true_go_state]
            x_pre = np.random.choice([0, 1, 2], ssd, p=p_x_z0)
            x_on_post = np.random.choice(
                [0, 1, 2], horizon - ssd, p=p_x_z1)
            x_seq = np.concatenate((x_pre, x_on_post))
            
        # Go trial
        elif true_stop_state == 'nonstop':
            if ssd != -1:
                raise ValueError("SSD must be -1 for go ('nonstop') trials.")

            y_seq = np.random.choice([0, 1, 2], horizon, p=model.p_y_z0_pre)
            x_seq = np.random.choice([0, 1, 2], horizon,
                                     p={"left": model.p_x_dL_z0,
                                        "right": model.p_x_dR_z0}[true_go_state])
            
        return x_seq, y_seq

    def get_action_and_qvals(beta, zeta, t_step):
        """Retrieve Q-values and select an action using a softmax policy."""
        beta_idx = np.searchsorted(model.beta_space, beta, side="left")
        zeta_idx = np.searchsorted(model.zeta_space, zeta, side="left")
        idx = (t_step, beta_idx, zeta_idx)

        v_left = model.value_left[idx]
        v_right = model.value_right[idx]
        v_wait = model.value_wait[idx]

        # Softmax action selection
        # Q-values are treated as costs; lower values are more favorable. 
        # Hence, we negate them for the softmax.
        q_vals = np.array([v_left, v_right, v_wait])
        q_vals = -q_vals  # cost
        q_vals -= np.max(q_vals)
        probs = np.exp(model.inv_temp * q_vals)
        probs /= np.sum(probs)
        action = np.random.choice([0, 1, 2], p=probs)

        return action, q_vals, (v_left, v_right, v_wait)

    # --- Constants ---
    horizon = model.horizon
    mu = 12 # Stop signal duration in time steps
    x_seq, y_seq = generate_obs_sequences(true_go_state, true_stop_state, ssd)

    # --- Initial beliefs ---
    beta, zeta, sigma = 0.5, 0.0, model.update_sigma(0.0, 0)
    beta_seq, zeta_seq, sigma_seq = [beta], [zeta], [sigma]
    action, q_vals, (v_left, v_right, v_wait) = get_action_and_qvals(
        beta, zeta, 0)

    if verbose:
        value_seq = [q_vals[action]]
        value_left_seq = [v_left]
        value_right_seq = [v_right]
        value_wait_seq = [v_wait]
    policy_seq = [action]

    rt, simu_action, ever_act = None, None, False

    for t_step, (x, y) in enumerate(zip(x_seq, y_seq)):
        if t_step >= horizon - 2:
            break

        zeta = model.update_zeta(zeta, y, t_step)
        sigma = model.update_sigma(zeta, t_step)
        beta = model.update_beta(beta, zeta, x)

        beta_seq.append(beta)
        zeta_seq.append(zeta)
        sigma_seq.append(sigma)

        action, q_vals, (v_left, v_right, v_wait) = get_action_and_qvals(
            beta, zeta, t_step)

        if verbose:
            value_seq.append(q_vals[action])
            value_left_seq.append(v_left)
            value_right_seq.append(v_right)
            value_wait_seq.append(v_wait)
        policy_seq.append(action)

        if action != 2 and not ever_act:
            simu_action = action
            rt = t_step
            ever_act = True

    # --- Outcome decoding ---
    simu_action = {0: 'left', 1: 'right', 2: 'wait'}.get(simu_action)
    if true_stop_state == 'nonstop':
        if simu_action == true_go_state:
            result = 'GS'
        elif simu_action in ['left', 'right']:
            result = 'GE'
        else:
            result = 'GM'
    else:
        result = 'SS' if not ever_act else 'SE'

    # --- Output ---
    if verbose:
        return {
            'result': result,
            'true_go_state': true_go_state,
            'true_stop_state': true_stop_state,
            'beta_seq': beta_seq,
            'zeta_seq': zeta_seq,
            'sigma_seq': sigma_seq,
            'value_seq': value_seq,
            'value_left_seq': value_left_seq,
            'value_right_seq': value_right_seq,
            'value_wait_seq': value_wait_seq,
            'policy_seq': policy_seq,
            'rt': rt
        }
    else:
        return (result, rt)