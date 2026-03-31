import numpy as np

class HDBM:
    """
    Hybrid Dynamic Belief Model (HDBM) for the Stop Signal Task.
    
    Updated version using parametric Beta distribution updating with distinct
    learning/forgetting rates for Go and Stop trials, and flexible fusion rules.
    """

    def __init__(self, alpha_go=0.85, alpha_stop=0.85, eta=5.0, rho=0.6, gamma=0.8, 
                 fusion_type='additive_2', a0=5.0, b0=1.0):
        """
        Initialize the HDBM model with free parameters.
        
        Parameters:
            alpha_go (float): Memory retention rate after Go trials.
            alpha_stop (float): Cognitive reset (retention) rate after Stop trials.
            eta (float): Habitual impulse increment added after each Go trial.
            rho (float): Weight of the hazard function in 'additive' fusion.
            gamma (float): Amplification factor of the hazard in 'multiplicative' fusion.
            fusion_type (str): 'additive_1', 'additive_2', or 'multiplicative'.
            a0 (float): Initial prior parameter alpha (representing Go counts).
            b0 (float): Initial prior parameter beta (representing Stop counts).
        """
        self.alpha_go = alpha_go
        self.alpha_stop = alpha_stop
        self.eta = eta
        self.rho = rho
        self.gamma = gamma
        self.fusion_type = fusion_type
        self.a0 = a0
        self.b0 = b0

        # Standard Weibull Hazard values (20 trials max)
        self.hazard_values = np.array([
            0.0723, 0.1439, 0.1924, 0.2315, 0.2649,
            0.2943, 0.3206, 0.3445, 0.3664, 0.3867,
            0.4055, 0.4232, 0.4397, 0.4553, 0.4701,
            0.4841, 0.4974, 0.5100, 0.5221, 0.5336
        ])

    def _get_hazard(self, run_length):
        """Return hazard value for a given number of consecutive go trials."""
        idx = min(run_length, len(self.hazard_values) - 1)
        return self.hazard_values[idx]

    def simu_task(self, sequence, block_size=None, return_details=False):
        """
        Run the HDBM simulation over a trial sequence.

        Args:
            sequence (list or np.array): Sequence of trials (0 = go, 1 = stop).
            block_size (int, optional): Resets the run length counter at block boundaries.
            return_details (bool): If True, returns (r_traj, Er_traj, h_traj) instead of just r_traj.

        Returns:
            np.array: Predicted r value for each trial (r_traj).
        """
        a, b = self.a0, self.b0
        run_length = 0
        
        r_traj = []
        Er_traj = []
        h_traj = []

        for n, trial in enumerate(sequence):
            # 1. Pre-trial: Calculate current subjective expectation
            Er = b / (a + b)
            h = self._get_hazard(run_length)

            # 2. Pre-trial: Fuse decision signals based on chosen mechanism
            if self.fusion_type == 'additive_1':
                r_raw = (1 - self.rho) * Er + self.rho * h
            elif self.fusion_type == 'additive_2':
                r_raw = Er + self.rho * h
            elif self.fusion_type == 'multiplicative':
                r_raw = Er * (1 + self.gamma * h)
            else:
                raise ValueError("fusion_type must be either 'additive_1', 'additive_2', or 'multiplicative'")

            # POMDP safe clipping
            r_final = np.clip(r_raw, 0, 1-1e-4)

            # Store trajectories
            r_traj.append(r_final)
            Er_traj.append(Er)
            h_traj.append(h)

            # 3. Post-trial: Physical stimulus appears, update internal parameters
            if trial == 0:  # Go Trial
                a = (1 - self.alpha_go) * self.a0 + self.alpha_go * (a + self.eta)
                b = (1 - self.alpha_go) * self.b0 + self.alpha_go * (b + 0)
                run_length += 1
            else:           # Stop Trial
                a = (1 - self.alpha_stop) * self.a0 + self.alpha_stop * (a + 0)
                b = (1 - self.alpha_stop) * self.b0 + self.alpha_stop * (b + 1)
                run_length = 0  # Streak broken, reset hazard

            # 4. Handle Block Resets
            if block_size is not None and (n + 1) % block_size == 0:
                run_length = 0
                a, b = self.a0, self.b0

        # Output formatting
        if return_details:
            return np.array(r_traj), np.array(Er_traj), np.array(h_traj)
        
        return np.array(r_traj)

