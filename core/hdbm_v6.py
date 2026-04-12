import numpy as np

class HDBM:
    """
    Hybrid Dynamic Belief Model (HDBM) for the Stop Signal Task.
    
    Updated version (v5): Uses parametric Weibull hazard function with a free 
    shape parameter 'k'. E[r] drop and Hazard rise are now fully decoupled.
    """

    def __init__(self, alpha_go=0.85, alpha_stop=0.85, eta=5.0, rho=0.6, k=2.5, gamma=0.8, 
                 fusion_type='additive_2', a0=5.0, b0=1.0):
        self.alpha_go = alpha_go
        self.alpha_stop = alpha_stop
        self.eta = eta
        self.rho = rho
        self.k = k            # NEW: Weibull shape parameter
        self.gamma = gamma
        self.fusion_type = fusion_type
        self.a0 = a0
        self.b0 = b0

        # 动态计算 Weibull Hazard: h(t) = (t / max_trials) ^ (k - 1)
        max_trials = 20
        self.hazard_values = np.zeros(max_trials + 1)
        for i in range(1, max_trials + 1):
            # i=0 保持为 0，防止 k<1 时出现 0的负数次方报错
            self.hazard_values[i] = (i / max_trials) ** (self.k - 1)

    def _get_hazard(self, run_length):
        idx = min(run_length, len(self.hazard_values) - 1)
        return self.hazard_values[idx]

    def simu_task(self, sequence, block_size=None, return_details=False):
        a, b = self.a0, self.b0
        run_length = 0
        
        r_traj, Er_traj, h_traj = [], [], []

        for n, trial in enumerate(sequence):
            Er = b / (a + b)
            h = self._get_hazard(run_length)

            if self.fusion_type == 'additive_1':
                r_raw = (1 - self.rho) * Er + self.rho * h
            elif self.fusion_type == 'additive_2':
                r_raw = Er + self.rho * h
            elif self.fusion_type == 'multiplicative':
                r_raw = Er * (1 + self.gamma * h)
            else:
                raise ValueError("Unknown fusion_type")

            r_final = np.clip(r_raw, 0, 1-1e-4)

            r_traj.append(r_final)
            Er_traj.append(Er)
            h_traj.append(h)

            if trial == 0:  
                a = (1 - self.alpha_go) * self.a0 + self.alpha_go * (a + self.eta)
                b = (1 - self.alpha_go) * self.b0 + self.alpha_go * (b + 0)
                run_length += 1
            else:           
                a = (1 - self.alpha_stop) * self.a0 + self.alpha_stop * (a + 0)
                b = (1 - self.alpha_stop) * self.b0 + self.alpha_stop * (b + 1)
                run_length = 0  

            if block_size is not None and (n + 1) % block_size == 0:
                run_length = 0
                a, b = self.a0, self.b0

        if return_details:
            return np.array(r_traj), np.array(Er_traj), np.array(h_traj)
        return np.array(r_traj)