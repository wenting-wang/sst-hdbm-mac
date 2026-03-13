import numpy as np
from scipy.stats import beta
np.random.seed(42)

class HDBM:
    """
    Hybrid Dynamic Belief Model (HDBM) for the Stop Signal Task.

    Models trial-by-trial estimates of stop probability r = P(stop) based on Bayesian belief updating
    and empirical hazard function.

    Parameters:
        alpha (float): Weight of Bayesian prior updating (decay rate).
        rho (float): Weight of hazard-based prediction.
        a (float): Beta distribution parameter alpha (prior shape).
        b (float): Beta distribution parameter beta (prior shape).
    """

    def __init__(self, alpha, rho, a=1, b=5):
        # Prior Beta distribution parameters
        self.alpha = alpha
        self.rho = rho
        self.a = a
        self.b = b

        # Discretize the probability space
        self.n_bin = 100
        self.bin_width = 1.0 / self.n_bin
        self.r_space = np.linspace(0, 1, self.n_bin)
        self.prior_beta = beta(self.a, self.b).pdf(self.r_space)

        # Fitted hazard values from fit_hazard.csv (Discrete Weibull: q=0.9277, beta=1.6186)
        # Represents P(Stop | Run Length) for lengths 1 to 20
        self.hazard_values = np.array([
            0.07230000, 0.14393138, 0.19246705, 0.23159678, 0.26497388,
            0.29431999, 0.32062351, 0.34451875, 0.36644358, 0.38671633,
            0.40557752, 0.42321449, 0.43977670, 0.45538575, 0.47014227,
            0.48413068, 0.49742263, 0.51007964, 0.52215493, 0.53369498
        ])

    def _get_hazard(self, num_go_trials):
        """Return hazard value for a given number of consecutive go trials."""
        # Clamp to the last available hazard value if run exceeds modeled length
        idx = min(num_go_trials, len(self.hazard_values) - 1)
        return self.hazard_values[idx]

    def simu_task(self, sequence, block_size=None):
        """
        Run the HDBM simulation over a trial sequence.

        Args:
            sequence (list or np.array): Sequence of trials (0 = go, 1 = stop)
            block_size (int, optional): Number of trials per block. If provided, 
                                        the run counter resets at the start of each block.
            e.g. len(sequence) = 360, block_size = 180 -> resets after 180 trials.

        Returns:
            list of float: Predicted r value for each trial.
        """
        r_pred_seq = [1/6]  # Initial prediction based on prior mean (a/(a+b))
        go_run_length = 0
        posterior = self.prior_beta.copy()

        for n, trial in enumerate(sequence):
            # 1. Update prior: blend of previous posterior and fixed prior (forgetting)
            #    prior_t = alpha * posterior_{t-1} + (1 - alpha) * prior_0
            prior = self.alpha * posterior + (1 - self.alpha) * self.prior_beta
            prior /= np.sum(prior * self.bin_width)

            # 2. Predict stop probability using hazard and expected belief
            #    r_pred = rho * Hazard(cnt) + (1 - rho) * E[prior]
            expected_r = np.sum(prior * self.r_space * self.bin_width)
            r_pred = self.rho * self._get_hazard(go_run_length) + (1 - self.rho) * expected_r
            r_pred_seq.append(r_pred)

            # 3. Bayesian update: posterior = likelihood * prior
            #    Likelihood: r if stop (trial=1), (1-r) if go (trial=0)
            likelihood = (self.r_space ** trial) * ((1 - self.r_space) ** (1 - trial))
            posterior = likelihood * prior
            
            # Avoid divide by zero if posterior collapses (numerical stability)
            norm = np.sum(posterior * self.bin_width)
            if norm > 0:
                posterior /= norm
            else:
                posterior = prior # Fallback if data is impossible under current belief

            # 4. Update go trial count (Run Length)
            if trial == 0:
                go_run_length += 1
            else:
                go_run_length = 0

            # 5. Handle Block Resets
            if block_size is not None and (n + 1) % block_size == 0:
                go_run_length = 0

        return [float(r) for r in r_pred_seq[:-1]]