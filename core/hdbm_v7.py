import numpy as np

class HDBM:
    """
    Hybrid Dynamic Belief Model (HDBM) v7.
    - Global Bayesian updates for E[r].
    - Local strict Weibull hazard h(t) resetting upon Stop signals.
    - Added free parameter: scale_lambda.
    """

    def __init__(self, alpha_go=0.85, alpha_stop=0.85, eta=5.0, rho=0.6, k=2.5, gamma=0.8, 
                 fusion_type='additive_2', a0=5.0, b0=1.0, scale_lambda=4.5):
        self.alpha_go = alpha_go
        self.alpha_stop = alpha_stop
        self.eta = eta
        self.rho = rho
        self.k = k            
        self.gamma = gamma
        self.fusion_type = fusion_type
        self.a0 = a0
        self.b0 = b0
        self.scale_lambda = scale_lambda

        # 预先计算 Hazard，最大支持连续 20 个 Go Trial
        max_possible_run = 20
        self.hazard_values = np.zeros(max_possible_run + 1)
        for i in range(1, max_possible_run + 1):
            # 严格的 Weibull Hazard 公式
            val = (self.k / self.scale_lambda) * ((i / self.scale_lambda) ** (self.k - 1))
            self.hazard_values[i] = val

    def _get_hazard(self, run_length):
        idx = min(run_length, len(self.hazard_values) - 1)
        return self.hazard_values[idx]

    def simu_task(self, sequence, block_size=None, return_details=False):
        a, b = self.a0, self.b0
        run_length = 0
        
        r_traj = []
        if return_details:
            Er_traj, h_traj, r_raw_traj = [], [], []

        for n, trial in enumerate(sequence):
            # 1. Bayesian Expectation
            Er = b / (a + b)
            
            # 2. Weibull Hazard
            h = self._get_hazard(run_length)

            # 3. Fusion Logic
            if self.fusion_type == 'additive_1':
                r_raw = (1 - self.rho) * Er + self.rho * h
            elif self.fusion_type == 'additive_2':
                r_raw = Er + self.rho * h
            elif self.fusion_type == 'multiplicative':
                r_raw = Er * (1 + self.gamma * h)
            else:
                raise ValueError("Unknown fusion_type")

            # 最终的预期概率截断
            r_final = np.clip(r_raw, 0, 1 - 1e-4)
            r_traj.append(r_final)

            if return_details:
                r_raw_traj.append(r_raw)
                Er_traj.append(Er)
                h_traj.append(h)

            # 4. State Update
            if trial == 0:  # Go trial
                a = (1 - self.alpha_go) * self.a0 + self.alpha_go * (a + self.eta)
                b = (1 - self.alpha_go) * self.b0 + self.alpha_go * (b + 0)
                run_length += 1
            else:           # Stop trial
                a = (1 - self.alpha_stop) * self.a0 + self.alpha_stop * (a + 0)
                b = (1 - self.alpha_stop) * self.b0 + self.alpha_stop * (b + 1)
                run_length = 0  

            # 5. Block Reset (根据你的 e2e 脚本，通常是 180 个 trial 清零一次)
            if block_size is not None and (n + 1) % block_size == 0:
                run_length = 0
                a, b = self.a0, self.b0

        if return_details:
            return np.array(r_traj), np.array(Er_traj), np.array(h_traj), np.array(r_raw_traj)
        
        return np.array(r_traj)