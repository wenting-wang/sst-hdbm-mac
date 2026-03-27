import numpy as np

class HDBM:
    """
    Hybrid Dynamic Belief Model (HDBM) for the Stop Signal Task.
    
    Updated version incorporating Learning, Hazard, and Fatigue components.
    The final stop prior is a linear combination of these three factors,
    with weights summing to 1.
    """

    def __init__(self, alpha_go=0.75, alpha_stop=0.75, k_go=5.0, 
                 w_hazard=0.4, w_fatigue=0.2, fatigue_shape=2.0, a0=5.0, b0=1.0):
        """
        Parameters:
            alpha_go, alpha_stop: Retention rates.
            k_go: Learning rate/Habitual impulse.
            w_hazard: Weight for the objective hazard function.
            w_fatigue: Weight for the fatigue/vigilance decrement.
            fatigue_shape: Curvature of fatigue (e.g., >1 means accelerates later).
        """
        self.alpha_go = alpha_go
        self.alpha_stop = alpha_stop
        self.k_go = k_go
        
        # Enforce weight sum to 1
        self.w_hazard = w_hazard
        self.w_fatigue = w_fatigue
        self.w_learning = max(0.0, 1.0 - w_hazard - w_fatigue)
        
        self.fatigue_shape = fatigue_shape
        self.a0 = a0
        self.b0 = b0

        # Standard Weibull Hazard values (20 trials max)
        self.hazard_values = np.array([
            0.0723, 0.1439, 0.1924, 0.2315, 0.2649,
            0.2943, 0.3206, 0.3445, 0.3664, 0.3867,
            0.4055, 0.4232, 0.4397, 0.4553, 0.4701,
            0.4841, 0.4974, 0.5100, 0.5221, 0.5336
        ])
        self.max_streak = len(self.hazard_values)

    def _get_hazard(self, run_length):
        idx = min(run_length, self.max_streak - 1)
        return self.hazard_values[idx]

    def _get_fatigue(self, run_length):
        # Normalized to [0, 1] then shaped
        norm_streak = min(run_length, self.max_streak - 1) / (self.max_streak - 1)
        return norm_streak ** self.fatigue_shape

    def simu_task(self, sequence, block_size=None, return_details=False):
        a, b = self.a0, self.b0
        run_length = 0
        
        r_traj, Er_traj, h_traj, f_traj = [], [], [], []

        for n, trial in enumerate(sequence):
            # 1. Calculate the three base components
            Er = b / (a + b)
            h = self._get_hazard(run_length)
            f = self._get_fatigue(run_length)

            # 2. Linear combination: Learning + Hazard - Fatigue
            r_raw = self.w_learning * Er + self.w_hazard * h - self.w_fatigue * f
            r_final = np.clip(r_raw, 0, 1-1e-10) # Keep valid probability bounds

            # Store trajectories
            r_traj.append(r_final)
            Er_traj.append(Er)
            h_traj.append(h)
            f_traj.append(f)

            # 3. Post-trial update
            if trial == 0:  # Go
                a = (1 - self.alpha_go) * self.a0 + self.alpha_go * (a + self.k_go)
                b = (1 - self.alpha_go) * self.b0 + self.alpha_go * (b + 0)
                run_length += 1
            else:           # Stop
                a = (1 - self.alpha_stop) * self.a0 + self.alpha_stop * (a + 0)
                b = (1 - self.alpha_stop) * self.b0 + self.alpha_stop * (b + 1)
                run_length = 0

            # Block Resets
            if block_size is not None and (n + 1) % block_size == 0:
                run_length = 0

        if return_details:
            return np.array(r_traj), np.array(Er_traj), np.array(h_traj), np.array(f_traj)
        
        return np.array(r_traj)