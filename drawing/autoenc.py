import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DropBlock2d
from typing import Callable, Optional
import copy
from drawing.projectUtils import move_dict_to_device
from power_spherical import HypersphericalUniform, MarginalTDistribution, PowerSpherical
from torchvision.transforms.functional import resize

class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=128, droprate=0.3):
        super().__init__()
        # Encoder
        self.activation = nn.ReLU
        self.encoder = nn.Sequential(
            # Input: 1x224x224
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # 64x112x112
            nn.BatchNorm2d(64),
            self.activation(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x56x56
            nn.BatchNorm2d(128),
            self.activation(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256x28x28
            nn.BatchNorm2d(256),
            self.activation(),
            nn.Dropout2d(p=droprate),
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            self.activation(),
            nn.Linear(1024, bottleneck_dim)  # Bottleneck (e.g., 128D)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 1024),
            self.activation(),
            nn.Linear(1024, 256 * 28 * 28),
            self.activation(),
            nn.Unflatten(1, (256, 28, 28)),  # 256x28x28
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            self.activation(),

            # Second upsampling block: from 128 channels to 64 channels
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.activation(),

            # Third upsampling block: from 64 channels to 1 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            self.activation(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        #latent = torch.tanh(latent)
        reconstructed = self.decoder(latent)
        return reconstructed



class ConvAutoencoderR(nn.Module):
    def __init__(self, bottleneck_dim=128, droprate=0.3):
        super().__init__()
        # Encoder
        self.activation = nn.ELU
        self.encoder = nn.Sequential(
            # Input: 1x224x224
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # 64x112x112
            nn.BatchNorm2d(64),
            self.activation(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x56x56
            nn.BatchNorm2d(128),
            self.activation(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256x28x28
            nn.BatchNorm2d(256),
            self.activation(),
            nn.Dropout2d(p=droprate),
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            self.activation(),
            nn.Linear(1024, bottleneck_dim)  # Bottleneck (e.g., 128D)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 1024),
            self.activation(),
            nn.Linear(1024, 256 * 28 * 28),
            self.activation(),
            nn.Unflatten(1, (256, 28, 28)),  # 256x28x28
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            self.activation(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            self.activation(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            self.activation(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        latent = latent/torch.norm(latent, dim=-1, keepdim=True)
        reconstructed = self.decoder(latent)
        return reconstructed





class ConvSVAE(nn.Module):
    def __init__(self, bottleneck_dim=128, droprate=0.3):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        # Encoder: Extract features from input image
        self.encoder = nn.Sequential(
            # Input: 1 x 224 x 224
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # 64 x 112 x 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 56 x 56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256 x 28 x 28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=droprate),
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(),
        )
        # Two linear layers to produce mean and log variance of the latent distribution
        self.fc_mu = nn.Linear(1024, bottleneck_dim)
        self.fc_var = nn.Linear(1024, 1)
            
        # Decoder: Convert latent vector back to image
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 28 * 28),
            nn.ReLU(),
            nn.Unflatten(1, (256, 28, 28)),  # 256 x 28 x 28
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    
    def reparameterize(self, mu, var):
        q_z = PowerSpherical(mu, var)
        p_z = HypersphericalUniform(self.bottleneck_dim, device=mu.device)
        return q_z, p_z
    
    def forward(self, x):
        # Encode input image to a 1024-dim feature vector
        h = self.encoder(x)
        # Produce mean and log variance for the latent space
        mu = self.fc_mu(h)
        var = self.fc_var(h).squeeze(-1)
        mu = mu / mu.norm(dim=-1, keepdim=True) # Make sure on hypersphere
        var = F.softplus(var) + 1.0 # Numerical stability
        

        # Sample a latent vector using the reparameterization trick
        q_z, p_z = self.reparameterize(mu, var)

        z = q_z.rsample()
        # Decode the latent vector to reconstruct the image
        reconstructed = self.decoder(z)
        return reconstructed, mu, var, q_z, p_z
    

class ConvVAE(nn.Module):
    def __init__(self, bottleneck_dim=128, droprate=0.3):
        super().__init__()
        # Encoder: Extract features from input image
        self.encoder = nn.Sequential(
            # Input: 1 x 224 x 224
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # 64 x 112 x 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 56 x 56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256 x 28 x 28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=droprate),
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(),
        )
        # Two linear layers to produce mean and log variance of the latent distribution
        self.fc_mu = nn.Linear(1024, bottleneck_dim)
        self.fc_logvar = nn.Linear(1024, bottleneck_dim)
        
        # Decoder: Convert latent vector back to image
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 28 * 28),
            nn.ReLU(),
            nn.Unflatten(1, (256, 28, 28)),  # 256 x 28 x 28
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    
    def reparameterize(self, mu, logvar):
        # Compute standard deviation from log variance
        std = torch.exp(0.5 * logvar)
        # Sample epsilon from a standard normal distribution
        eps = torch.randn_like(std)
        # Reparameterize
        return mu + eps * std
    
    def forward(self, x):
        # Encode input image to a 1024-dim feature vector
        h = self.encoder(x)
        # Produce mean and log variance for the latent space
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Sample a latent vector using the reparameterization trick
        z = self.reparameterize(mu, logvar)
        # Decode the latent vector to reconstruct the image
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

class Conv1DEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, droprate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            # (B, T, d) → (B, 64, T/2)
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(droprate),
            
            # (B, 64, T/2) → (B, 128, T/4)
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(droprate),
            
            # (B, 128, T/4) → (B, 256, T/8)
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout1d(droprate),
            
            nn.AdaptiveAvgPool1d(1),  # Global pooling → (B, 256, 1)
            nn.Flatten(),
            nn.Linear(256, latent_dim))
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, d) → (B, d, T) for Conv1d
        x = self.net(x)
        x = x / torch.norm(x,dim=-1,keepdim=True)
        return x

class Conv1DDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 256 * (seq_len // 8)),
            nn.Unflatten(1, (256, seq_len // 8))
        )
        self.net = nn.Sequential(
            # (B, 256, T/8) → (B, 128, T/4)
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # (B, 128, T/4) → (B, 64, T/2)
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # (B, 64, T/2) → (B, d, T)
            nn.ConvTranspose1d(64, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.AdaptiveAvgPool1d(seq_len),
        )
    
    def forward(self, z):
        z = self.projection(z)
        x = self.net(z)
        return x.permute(0, 2, 1)  # (B, d, T) → (B, T, d)

class TemporalAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim=32, droprate=0.2):
        super().__init__()
        self.encoder = Conv1DEncoder(input_dim, latent_dim, droprate=droprate)
        self.decoder = Conv1DDecoder(latent_dim, input_dim, seq_len)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
class MLPAutoencoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 encoder_dims, 
                 latent_dim, 
                 decoder_dims=None, 
                 final_activation=nn.Sigmoid(),
                 auto_flatten=True,
                 output_shape=None):
        """
        Parameters:
            input_dim (int): Size of the flattened input.
            encoder_dims (list of int): Sizes of the encoder hidden layers.
            latent_dim (int): Dimension of the latent space.
            decoder_dims (list of int, optional): Sizes of the decoder hidden layers.
                                                  If None, uses the reverse of encoder_dims.
            auto_flatten (bool): If True, automatically flatten inputs with more than 2 dimensions.
            output_shape (tuple, optional): Desired output shape (excluding batch dimension). 
                                            If provided, used to unflatten the decoder output.
        """
        super(MLPAutoencoder, self).__init__()
        self.auto_flatten = auto_flatten
        self.output_shape = output_shape  # If None, will use the input's original shape
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim
        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror encoder by default)
        if decoder_dims is None:
            decoder_dims = encoder_dims[::-1]
            
        decoder_layers = []
        prev_dim = latent_dim
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # Check if we need to flatten
        if self.auto_flatten and x.ndim > 2:
            batch_size = x.size(0)
            original_shape = x.shape[1:]
            x = x.view(batch_size, -1)
        else:
            original_shape = None
        
        # Pass through encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Unflatten if necessary
        if self.auto_flatten and original_shape is not None:
            # Use provided output_shape if available; otherwise, use original shape
            unflatten_shape = self.output_shape if self.output_shape is not None else original_shape
            decoded = decoded.view(x.size(0), *unflatten_shape)
        return decoded


class AutoencoderTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_best: bool = True,
        key: str = "data",
        hook = None,
        weight = 0.01
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.save_best = save_best
        self.key = key
        self.hook = hook
        self.weight = 0.01
        
        # Tracking history
        self.train_losses = []
        self.val_losses = []

        # Best model tracking
        self.best_model_weights = None
        self.best_val_loss = float('inf')
        self.best_epoch = -1

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            if self.hook:
                self.hook.clear()
            # Move entire batch to the specified device
            batch = move_dict_to_device(batch, self.device)
            # For autoencoders, the input is the target
            input_data = batch[self.key]
            
            self.optimizer.zero_grad()
            # Forward pass: obtain the reconstruction from the autoencoder
            reconstructed = self.model(input_data)
            loss = self.loss_fn(reconstructed, input_data)

            if self.hook:
                loss += self.weight*self.hook.activation_loss
            
            # Backward pass and optimizer step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track loss (weighted by batch size)
            batch_size = input_data.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            
        avg_loss = total_loss / total_items
        return avg_loss

    def _validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_items = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = move_dict_to_device(batch, self.device)
                input_data = batch[self.key]
                reconstructed = self.model(input_data)
                loss = self.loss_fn(reconstructed, input_data)
                
                batch_size = input_data.size(0)
                total_loss += loss.item() * batch_size
                total_items += batch_size

        avg_loss = total_loss / total_items
        return avg_loss

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
    ):
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation phase (if provided)
            val_loss = 0.0
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.val_losses.append(val_loss)

                # Save best model weights based on validation loss
                if self.save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = epoch + 1

            # Step the scheduler (if any)
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Print progress
            msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4e}"
            if val_loader is not None:
                msg += f" | Val Loss: {val_loss:.4e}"
            if epoch + 1 == self.best_epoch:
                msg += " 🔖"
            print(msg)

    def restore_best_model(self):
        """Load the weights of the best performing model."""
        if self.best_model_weights is None:
            raise RuntimeError("No best model weights available. Did you run validation during training?")
        self.model.load_state_dict(self.best_model_weights)


class ActivationL1LossHook:
    def __init__(self, model):
        self.model = model
        self.activation_loss = 0.0
        self.hooks = []
        self.register_hooks()

    def hook_fn(self, module, input, output):
        # Accumulate the L1 loss (sum of absolute activations)
        self.activation_loss += output.abs().mean()

    def register_hooks(self):
        # Register hook for every ReLU layer in the model
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)

    def clear(self):
        # Reset the activation loss (to be called before each forward pass)
        self.activation_loss = 0.0

    def remove_hooks(self):
        # Remove all hooks if no longer needed
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def vae_loss(outputs, x,beta=1.0):
    xhat,mu,logvar = outputs

    # Reconstruction loss: using MSE loss as an example
    # If your images are normalized between 0 and 1, you might consider using BCE loss
    recon_loss = F.mse_loss(xhat, x, reduction='mean')
    
    # KL Divergence loss:
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss:
    return recon_loss + beta * kl_loss

def vmf_loss(outputs, x, beta=1.0):
    xhat, mu, var, q_z, p_z = outputs
    recon_loss = F.mse_loss(xhat, x, reduction='mean')
    loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    loss = recon_loss + beta*loss_KL
    return loss
