import numpy as np

class HDBM:
    """
    Hybrid Dynamic Belief Model (HDBM) for the Stop Signal Task.
    
    Updated version using parametric Beta distribution updating with distinct
    learning/forgetting rates for Go and Stop trials, and flexible fusion rules.
    """

    def __init__(self, alpha_go=0.75, alpha_stop=0.75, k_go=5.0, rho=0.6, gamma=0.8, 
                 fusion_type='additive', a0=5.0, b0=1.0):
        """
        Initialize the HDBM model with free parameters.
        
        Parameters:
            alpha_go (float): Memory retention rate after Go trials.
            alpha_stop (float): Cognitive reset (retention) rate after Stop trials.
            k_go (float): Habitual impulse increment added after each Go trial.
            rho (float): Weight of the hazard function in 'additive' fusion.
            gamma (float): Amplification factor of the hazard in 'multiplicative' fusion.
            fusion_type (str): 'additive' or 'multiplicative'.
            a0 (float): Initial prior parameter alpha (representing Go counts).
            b0 (float): Initial prior parameter beta (representing Stop counts).
        """
        self.alpha_go = alpha_go
        self.alpha_stop = alpha_stop
        self.k_go = k_go
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
                raise ValueError("fusion_type must be either 'additive' or 'multiplicative'")

            # POMDP safe clipping
            r_final = np.clip(r_raw, 0, 1-1e-4)

            # Store trajectories
            r_traj.append(r_final)
            Er_traj.append(Er)
            h_traj.append(h)

            # 3. Post-trial: Physical stimulus appears, update internal parameters
            if trial == 0:  # Go Trial
                a = (1 - self.alpha_go) * self.a0 + self.alpha_go * (a + self.k_go)
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



import numpy as np
import matplotlib.pyplot as plt

# 假设 HDBM 类已经定义在上方

if __name__ == "__main__":
    # 1. 生成测试序列: 360 trials, 大约 25% Stop (1), 75% Go (0)
    np.random.seed(42) # 固定随机种子以保证结果可复现
    sequence = np.random.choice([0, 1], size=360, p=[0.75, 0.25])
    
    # 2. 初始化模型
    model = HDBM(a0=5.0, b0=1.0, fusion_type='additive')
    
    # 3. 运行带有 block_size=180 的仿真
    r_traj, Er_traj, h_traj = model.simu_task(sequence, block_size=180, return_details=True)
    
    # 4. 打印 Block 交界处的数据验证 (Python 是 0-index)
    # Index 179 -> Block 1 的最后一个 trial (Trial 180)
    # Index 180 -> Block 2 的第一个 trial (Trial 181)
    print("--- Block 交界处数值检查 ---")
    print(f"Trial 180 (Block 1 结束) -> Er: {Er_traj[179]:.4f}, h: {h_traj[179]:.4f}")
    print(f"Trial 181 (Block 2 开始) -> Er: {Er_traj[180]:.4f}, h: {h_traj[180]:.4f}")
    print(f"Prior 理论值应为 b0/(a0+b0) = 1/(5+1) = {1/6:.4f}")
    print("----------------------------\n")

    # 5. 可视化轨迹
    plt.figure(figsize=(14, 6))
    
    # 画出三条轨迹
    plt.plot(r_traj, label='Final Expectation (r)', color='black', linewidth=2)
    plt.plot(Er_traj, label='Subjective Prior (Er)', color='blue', alpha=0.7)
    plt.plot(h_traj, label='Hazard Function (h)', color='red', alpha=0.5, linestyle='--')
    
    # 画出 Block 分界线
    plt.axvline(x=179.5, color='gray', linestyle=':', linewidth=2, label='Block Boundary')
    
    # 装饰图表
    plt.title('HDBM Trajectories across 2 Blocks (360 Trials)')
    plt.xlabel('Trial Number')
    plt.ylabel('Probability / Expectation')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()