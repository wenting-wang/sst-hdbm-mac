import torch
import torch.nn as nn
import zuko
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. THE RL SIMULATOR (2-Armed Bandit)
# ==========================================
def simulate_q_learning(z_params, n_trials=50):
    """
    Simulates a 2-armed bandit task.
    z_params: Unconstrained parameters [z_alpha, z_beta]
    """
    # Transform unbounded z to bounded RL parameters
    alpha = torch.sigmoid(z_params[0])            # alpha in (0, 1)
    beta = 10.0 * torch.sigmoid(z_params[1])      # beta in (0, 10)
    
    # Reward probabilities for the two arms
    payout_probs = [0.8, 0.2]
    Q = torch.zeros(2)
    
    trial_data = []
    
    for _ in range(n_trials):
        # Action selection using Softmax
        logits = beta * Q
        probs = torch.softmax(logits, dim=0)
        action = torch.multinomial(probs, 1).item()
        
        # Environment returns reward
        reward = 1.0 if torch.rand(1).item() < payout_probs[action] else 0.0
        
        # Q-value update
        Q[action] = Q[action] + alpha * (reward - Q[action])
        
        # Feature representation: [is_arm_0, is_arm_1, reward]
        feature = [1.0 if action == 0 else 0.0, 
                   1.0 if action == 1 else 0.0, 
                   reward]
        trial_data.append(feature)
        
    return torch.tensor(trial_data, dtype=torch.float32)

def generate_dataset(n_agents=2000, n_trials=50):
    print("Simulating dataset...")
    # Generate random true parameters in unconstrained space (Normal distribution)
    true_z = torch.randn(n_agents, 2) 
    
    sequences = []
    for i in range(n_agents):
        seq = simulate_q_learning(true_z[i], n_trials=n_trials)
        sequences.append(seq)
        
    X = torch.stack(sequences) # Shape: (n_agents, n_trials, 3)
    Y = true_z                 # Shape: (n_agents, 2)
    return X, Y

# ==========================================
# 2. YOUR MODEL (Slightly adapted for dims)
# ==========================================
class AmortizedInferenceNet(nn.Module):
    def __init__(self, trial_feature_dim=3, d_model=64, n_heads=4, n_layers=2, param_dim=2):
        super().__init__()
        self.embedding = nn.Linear(trial_feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=128
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Zuko Flow: param_dim is what we are predicting, context is the transformer output
        self.flow = zuko.flows.MAF(features=param_dim, context=d_model, hidden_features=[64, 64])
        
    def forward(self, x_seq, true_params_scaled=None):
        emb = self.embedding(x_seq)
        out_seq = self.transformer(emb)
        
        # Mean pooling across the sequence (alternatively, use a [CLS] token)
        context = out_seq.mean(dim=1) 
        
        # Condition the flow on the sequence context
        dist = self.flow(context)
        
        if true_params_scaled is not None:
            # Return Negative Log-Likelihood for training
            return -dist.log_prob(true_params_scaled)
        
        # Return the distribution object for sampling/inference later
        return dist

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model():
    # 1. Get Data
    X, Y = generate_dataset(n_agents=5000, n_trials=50)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 2. Init Model
    model = AmortizedInferenceNet(trial_feature_dim=3, param_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Train
    epochs = 10
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            # Forward pass returns the Negative Log Likelihood
            loss = model(batch_x, batch_y).mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss (NLL): {epoch_loss/len(loader):.4f}")
        
    return model

# ==========================================
# 4. INFERENCE & VISUALIZATION
# ==========================================
def plot_posterior(model):
    model.eval()
    
    # Generate a single unseen test subject
    true_z = torch.randn(2)
    test_seq = simulate_q_learning(true_z, n_trials=50).unsqueeze(0) # Add batch dim
    
    with torch.no_grad():
        # Get the conditional distribution
        predicted_dist = model(test_seq)
        
        # Draw 2000 samples from the predicted posterior
        samples = predicted_dist.sample((2000,))
        samples = samples.squeeze(1).numpy() # Remove context batch dim
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.hist2d(samples[:, 0], samples[:, 1], bins=40, density=True, cmap='Blues')
    plt.colorbar(label='Posterior Density')
    
    # Mark the ground truth
    plt.scatter(true_z[0].item(), true_z[1].item(), color='red', marker='*', s=200, label='True Parameters')
    
    plt.title('Posterior Distribution learned by Normalizing Flow')
    plt.xlabel('Unconstrained $\\alpha$ ($z_0$)')
    plt.ylabel('Unconstrained $\\beta$ ($z_1$)')
    plt.legend()
    plt.show()

# Run the demo
if __name__ == "__main__":
    trained_model = train_model()
    plot_posterior(trained_model)