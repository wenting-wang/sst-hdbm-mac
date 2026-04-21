import numpy as np

class POMDP:
    """
    Partially Observable Markov Decision Process (POMDP) for the Stop Signal Task (SST).

    This model simulates within-trial belief dynamics under uncertainty about sensory inputs
    and trial type. It supports dynamic hazard modeling and cost-sensitive decision policies.
    """

    def __init__(self, q_d_n, q_d, q_s_n, q_s, cost_go_error, cost_go_missing,
                 cost_stop_error, cost_time, inv_temp, rate_stop_trial):
        """
        Initialize the model with perceptual noise and cost parameters.

        Args:
            q_d_n (float): Null level for the go-signal (chi').
            q_d (float): Precision level of the go-signal (chi).
            q_s_n (float): Null level for the stop-signal (delta').
            q_s (float): Precision level of the stop-signal (delta).
            cost_go_error (float): Cost of an incorrect directional response.
            cost_go_missing (float): Cost of a missing response on go trials.
            cost_stop_error (float): Cost of an inhibition failure (stop error).
            cost_time (float): Per-step time cost (e.g., delay penalty).
            inv_temp (float): Softmax inverse temperature for action selection.
            rate_stop_trial (float): Prior probability of stop trials (default typically 1/6).
        """
        # Hazard function parameter
        self.lamb = 0.1
        # Prior probability of a stop trial (r)
        self.rate_stop_trial = rate_stop_trial

        # Observation parameters
        self.q_d_n = q_d_n
        self.q_d = q_d
        self.q_s_n = q_s_n
        self.q_s = q_s

       # Cost parameters
        self.cost_go_error = cost_go_error
        self.cost_go_missing = cost_go_missing
        self.cost_stop_error = cost_stop_error
        self.cost_time = cost_time

        # Softmax inverse temperature
        self.inv_temp = inv_temp

        # Discretization parameters
        self.bins = 80
        self.horizon = 41
        self.beta_space = np.linspace(0, 1, self.bins + 1)  # go signal belief
        self.zeta_space = np.linspace(
            0, 1, self.bins + 1)  # stop signal belief
        self.sigma_space = np.linspace(
            0, 1, self.bins + 1)  # stop trial belief
        self.shape = (self.horizon, self.bins + 1, self.bins + 1)

        # Time-varying hazard
        self.hazard = np.array([self._generate_hazard(t_step)
                               for t_step in range(self.horizon)])

        # Observation likelihoods: p(x | d, z)
        self.p_x_dL_z0, self.p_x_dR_z0 = self._generate_observation_probs(
            q_d, q_d_n)  # Pre-stop signal
        self.p_x_dL_z1, self.p_x_dR_z1 = self._generate_observation_probs(
            q_d, q_d_n, reverse=True)  # Stop signal on and post

        # Observation likelihoods: p(y | z)
        self.p_y_z0_pre, self.p_y_z1_pre = self._generate_observation_probs(
            q_s, q_s_n)
        self.p_y_z0_on, self.p_y_z1_on = self._generate_observation_probs(
            q_s, q_s_n)
        self.p_y_z0_post, self.p_y_z1_post = self._generate_observation_probs(
            q_s, q_s_n, reverse=True)

        # Action space: 0 = left, 1 = right, 2 = wait
        self.action_space = [0, 1, 2]

        # Value and policy tables
        self.policy = np.full(self.shape, np.nan)
        self.value = np.full(self.shape, np.nan)
        self.value_wait = np.full(self.shape, np.nan)
        self.value_left = np.full(self.shape, np.nan)
        self.value_right = np.full(self.shape, np.nan)

    def _generate_observation_probs(self, precision, null, reverse=False):
        """
        Generate observation probabilities P(obs | signal).

        Args:
            precision (float): Precision level of the signal.
            null (float): Null (ambiguity) level of the signal.
            reverse (bool): If True, simulates degraded observation after the signal disappears.

        Returns:
            tuple: Two lists representing probabilities for condition 0 and condition 1.
                   For go-signal: P(x | d, z) where x in {left, right, null} maps to [0, 1, 2].
                   For stop-signal: P(y | z) where y in {absent, present, null} maps to [0, 1, 2].
        """
        if not reverse:
            p0 = [(1 - null) * precision, (1 - null) * (1 - precision), null]
            p1 = [(1 - null) * (1 - precision), (1 - null) * precision, null]
        else:
            # Post-signal: signal disappears, residual effect remains (null influence more)
            p0 = [null * precision, null * (1 - precision), 1 - null]
            p1 = [null * (1 - precision), null * precision, 1 - null]

        return p0, p1

    def _generate_hazard(self, t_step):
        """
        Compute the probability (hazard) of stop signal onset at a given time step.

        Args:
            t_step (int): The current time step in the trial.

        Returns:
            float: The probability that the stop signal appears at time t_step.
        """
        survival_prob = (1 - self.q_s) ** (t_step - 1)
        numerator = self.rate_stop_trial * survival_prob * self.lamb
        denominator = self.rate_stop_trial * \
            survival_prob + (1 - self.rate_stop_trial)

        return numerator / denominator

    def update_beta(self, beta, zeta, x):
        """
        Update the belief over the go cue direction (beta).

        Args:
            beta (float): Prior belief that the direction is "right" (d = R).
            zeta (float): Belief that the stop signal has already appeared (z = 1).
            x (int): Observed direction cue (0 = left, 1 = right, 2 = null).

        Returns:
            float: The updated belief that the direction is "right".
        """
        # Marginalize likelihoods over the stop signal status (z)
        # p(x|d=R,z=0) + p(x|d=R,z=1)
        p_dR = (1 - zeta) * self.p_x_dR_z0[x] + zeta * self.p_x_dR_z1[x]
        # p(x|d=L,z=0) + p(x|d=L,z=1)
        p_dL = (1 - zeta) * self.p_x_dL_z0[x] + zeta * self.p_x_dL_z1[x]

        numerator = beta * p_dR
        denominator = beta * p_dR + (1 - beta) * p_dL

        return numerator / denominator if denominator > 0 else beta

    def update_zeta(self, zeta, y, t_step):
        """
        Update the belief over stop signal onset (zeta).

        Args:
            zeta (float): Prior belief that the stop signal has occurred (z = 1).
            y (int): Stop-signal observation (0 = absent, 1 = present, 2 = null).
            t_step (int): The current time step.

        Returns:
            float: The updated belief over stop signal onset.
        """
        # Assume the subject imagines the stop signal remains subjectively active since onset
        h = self.hazard[t_step]

        # Compute belief updates
        zeta_already = self.p_y_z1_on[y] * (zeta + (1 - zeta) * h)
        zeta_future = self.p_y_z0_pre[y] * (1 - zeta) * (1 - h)

        denominator = zeta_already + zeta_future
        return zeta_already / denominator

    def update_sigma(self, zeta, t_step):
        """
        Compute the belief that the current trial is a stop trial (sigma).

        Args:
            zeta (float): Belief over stop-signal onset (z = 1).
            t_step (int): The current time step.

        Returns:
            float: The overall belief that this is a stop trial.
        """
        # Prior probability of being a stop trial, conditional on not having seen it yet
        numerator = (1 - self.lamb) ** t_step * self.rate_stop_trial
        denominator = numerator + (1 - self.rate_stop_trial)

        sigma_future = numerator / denominator

        # Compute updated belief
        return zeta + (1 - zeta) * sigma_future

    def immediate_cost(self, action, beta, zeta, t_step):
        """
        Compute the immediate expected cost of taking an action.

        Args:
            action (int): 0 = left, 1 = right, 2 = wait.
            beta (float): Belief that the direction is "right" (P(d = R)).
            zeta (float): Belief that there is a stop signal (P(z_t = 1)) at time t_step.
            t_step (int): The current time step.

        Returns:
            float: The expected immediate cost of the action.
        """
        # Calculate the belief that this is a stop trial (sigma) based on zeta and time step
        sigma = self.update_sigma(zeta, t_step)
        # Immediate time cost for taking any action (including wait)
        # i.e. All outcomes (GS, GE, GM, SS, SE) incur time cost
        time_cost = self.cost_time * t_step

        # Expected cost of a go error (incurs only on go trials)
        # i.e. Go Error (GE)
        if action in [0, 1]:
            p_error = beta if action == 0 else 1 - beta  # wrong direction probability
            go_error_cost = (1 - sigma) * p_error * self.cost_go_error
        else:
            go_error_cost = 0

        # Expected cost of violating a stop instruction (acting on a stop trial)
        # i.e. Stop Error (SE)
        stop_error_cost = sigma * \
            self.cost_stop_error if action in [0, 1] else 0

        return time_cost + go_error_cost + stop_error_cost

    def terminal_cost(self, zeta):
        """
        Compute the terminal cost if the agent waits until the horizon without responding.

        Args:
            zeta (float): Belief over stop signal onset.

        Returns:
            float: The final accumulated cost for not responding.
        """
        # Compute the belief that this is a stop trial (sigma) based on zeta and time step
        sigma = self.update_sigma(zeta, self.horizon - 1)

        # Time cost for waiting until the end of the trial
        time_cost = self.cost_time * (self.horizon - 1)

        # Penalty for missing a response if the trial was actually a go trial
        # i.e. Go Missing (GM)
        go_missing_cost = (1 - sigma) * self.cost_go_missing

        return time_cost + go_missing_cost

    def p_trans_x_(self, beta, x, z):
        """
        Compute the marginalized observation likelihood P(x | beta, z).

        Args:
            beta (float or ndarray): Belief that the go cue is "right".
            x (int): Observed cue (0 = left, 1 = right, 2 = null).
            z (int): Stop signal status (0 = not yet appeared, 1 = appeared).

        Returns:
            float or ndarray: Marginalized likelihood of observing cue x given
            belief beta and stop signal status z.
        """
        if z == 0:
            return beta * self.p_x_dR_z0[x] + (1 - beta) * self.p_x_dL_z0[x]
        elif z == 1:
            return beta * self.p_x_dR_z1[x] + (1 - beta) * self.p_x_dL_z1[x]

    def p_trans_y_(self, zeta, y, t_step, z):
        """
        Compute the joint transition probability P(y | z) * P(z | zeta).

        Args:
            zeta (float or ndarray): Belief that the stop signal has occurred.
            y (int): Observed stop signal (0 = absent, 1 = present, 2 = null).
            t_step (int): The current time step.
            z (int): Stop signal status (0 = not yet appeared, 1 = appeared).

        Returns:
            float or ndarray: The joint probability of observing y given z and
            the belief zeta about stop signal onset.
        """
        # Hazard probability
        hazard_prob = self.hazard[t_step + 1]

        if z == 0:
            return (1 - zeta) * (1 - hazard_prob) * self.p_y_z0_pre[y]
        elif z == 1:
            # Assume subject treats stop signal as actively sustained once triggered
            return (zeta + (1 - zeta) * hazard_prob) * self.p_y_z1_on[y]

    def future_value_tensor_(self, beta, zeta, t_step):
        """
        Bilinear-interpolate future values V_{t+1}(beta, zeta) on the discrete grid.

        Args:
            beta (ndarray): Belief after transition for beta-branches, shape (nbeta, nzeta, 3).
            zeta (ndarray): Belief after transition for zeta-branches, shape (nzeta, 3).
            t_step (int): The target time step.

        Returns:
            ndarray: Interpolated future values of shape (nbeta, nzeta, 3, 3).
        """
        V = self.value[t_step]                          # Shape: (bins+1, bins+1)
        nbeta = beta.shape[0]
        nzeta = zeta.shape[0]

        # Map coordinates to grid indices [0, bins]
        B = np.clip(beta, 0.0, 1.0) * self.bins         # Shape: (nbeta, nzeta, 3)
        Z = np.clip(zeta, 0.0, 1.0) * self.bins         # Shape: (nzeta, 3)

        # Expand for broadcasting
        B = B[..., None]                                # Shape: (nbeta, nzeta, 3, 1)
        Z = Z[None, :, None, :]                         # Shape: (1, nzeta, 1, 3)

        # Identify cell corners (indices)
        b0 = np.floor(B).astype(int)
        z0 = np.floor(Z).astype(int)
        b1 = np.clip(b0 + 1, 0, self.bins)
        z1 = np.clip(z0 + 1, 0, self.bins)

        # Compute interpolation weights in [0, 1)
        wb = B - b0                                      # Shape: (nbeta, nzeta, 3, 1)
        wz = Z - z0                                      # Shape: (1, nzeta, 1, 3)

        # Extract values at the four corners
        V00 = V[b0, z0]
        V01 = V[b0, z1]
        V10 = V[b1, z0]
        V11 = V[b1, z1]

        # Apply bilinear interpolation
        Vinterp = ((1 - wb) * (1 - wz) * V00 +
                   (1 - wb) * wz * V01 +
                   wb * (1 - wz) * V10 +
                   wb * wz * V11)

        return Vinterp

    def value_iteration_tensor(self):
        """
        Perform value iteration over the full tensorized belief space.

        This method computes the optimal policy and value function over time by 
        iteratively updating state-action values backward from the final trial step. 
        It accounts for
        - Per-step immediate cost (for all actions)
        - Future value under 'wait' action based on belief updates and transition probabilities
        - Uncertainty in both go cue direction and stop signal onset
        """
        # Initialize terminal values: The cost of doing nothing until the end of the trial
        self.value[-1, :, :] = self.terminal_cost(self.zeta_space)

        # Backward iteration over time steps
        for t_step in range(self.horizon - 2, -1, -1):
            # Precompute zeta index
            zeta_idx = (self.zeta_space * self.bins).astype(int)

            # --- Belief state updates ---
            # Compute zeta(t+1) under all y observations; shape: (81, 3)
            new_zeta = np.array([[self.update_zeta(zeta, y, t_step)
                                  for y in range(3)] for zeta in self.zeta_space])

            # Compute beta(t+1) under all x observations; shape: (81, 81, 3)
            new_beta = np.array([[[self.update_beta(beta, zeta, x)
                                   for x in range(3)]
                                  for zeta in self.zeta_space]
                                 for beta in self.beta_space])

            # --- Transition probabilities P(x, y | beta, zeta) ---
            # Marginalize over latent variable z in {0, 1}
            p_trans_x_z0 = np.array([[self.p_trans_x_(beta, x, 0)
                                     for x in range(3)] for beta in self.beta_space])
            p_trans_y_z0 = np.array([[self.p_trans_y_(zeta, y, t_step, 0)
                                     for y in range(3)] for zeta in self.zeta_space])
            p_trans_z0 = np.einsum('ik,jl->ijkl', p_trans_x_z0, p_trans_y_z0)

            p_trans_x_z1 = np.array([[self.p_trans_x_(beta, x, 1)
                                     for x in range(3)] for beta in self.beta_space])
            p_trans_y_z1 = np.array([[self.p_trans_y_(zeta, y, t_step, 1)
                                     for y in range(3)] for zeta in self.zeta_space])
            p_trans_z1 = np.einsum('ik,jl->ijkl', p_trans_x_z1, p_trans_y_z1)

            # Combine probabilities; final shape: (81, 81, 3, 3)
            p_trans = p_trans_z0 + p_trans_z1

            # --- Expected value of future WAIT action ---
            # shape: (81,81,3,3)
            future_value = self.future_value_tensor_(
                new_beta, new_zeta, t_step + 1)

            # shape: (81,81)
            expected_value = (p_trans * future_value).sum(axis=(-1, -2))

            # shape: (81,81)
            self.value_wait[t_step, np.arange(
                self.bins + 1)[:, None], zeta_idx] = expected_value

            # --- Expected costs for LEFT and RIGHT actions ---
            action_costs = np.array([[[self.immediate_cost(action, beta, zeta, t_step)
                                     for zeta in self.zeta_space]
                                     for beta in self.beta_space]
                                     for action in [0, 1]])
            self.value_left[t_step] = action_costs[0]
            self.value_right[t_step] = action_costs[1]
            
            # --- Determine optimal action and value ---
            all_action_values = np.stack(
                [self.value_left[t_step], self.value_right[t_step], self.value_wait[t_step]])

            minv = np.min(all_action_values, axis=0)
            tol = 1e-9
            is_best = np.isclose(
                all_action_values, minv[None, ...], atol=tol, rtol=0)
            best = np.argmin(all_action_values, axis=0)

            # Tie-breaking rules
            # 1. Left and Right tie (Wait is sub-optimal): Decide based on beta threshold
            lr_tie = (is_best[0] & is_best[1]) & (~is_best[2])
            beta_grid = self.beta_space[:, None]
            best[(beta_grid < 0.5) & lr_tie] = 0
            best[(beta_grid > 0.5) & lr_tie] = 1

            # 2. Assign 'Wait' if it ties with optimal active response
            best[(is_best[2]) & (is_best.sum(axis=0) > 1)] = 2

            self.value[t_step] = minv
            self.policy[t_step] = best
