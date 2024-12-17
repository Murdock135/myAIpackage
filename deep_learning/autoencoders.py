import torch
import torch.nn as nn
import torch.optim as optim

# Define the Sparse Autoencoder (SAE) class
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        
        # Encoder: input -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder: hidden -> output (reconstruction of input)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # Encoding
        hidden = self.encoder(x)
        # Decoding (reconstruction)
        reconstruction = self.decoder(hidden)
        return reconstruction, hidden

# L1 regularization function for sparsity
def l1_penalty(hidden_activations, lambda_l1):
    return lambda_l1 * torch.sum(torch.abs(hidden_activations))