import torch
from torch import nn
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(self, backbonemodel, encoder, noise_scheduler, trajmax, trajmin, sketchprocessor = None, projprocessor = None):
        super().__init__()
        self.model = backbonemodel
        self.encoder = encoder
        self.noise_scheduler = noise_scheduler
        self.inference_model = None
        self.device = next(self.model.parameters()).device # Hacky way to get device that backbone is on
        self.diffusion_steps = noise_scheduler.config.num_train_timesteps
        self.embed_dim = self.model.cond_dim
        # Register buffers for constant tensors
        self.register_buffer("trajmax", trajmax.to(torch.float32))
        self.register_buffer("trajmin", trajmin.to(torch.float32))
        self.sketchprocessor = sketchprocessor
        self.projprocessor = projprocessor

    def forward(self, images, sketches, guidance_scale=1.0, condval = 0.8): # This is not really a forward method. It samples.
        self.encoder.eval()
        self.inference_model.eval()
        device = next(self.model.parameters()).device # Hacky way to get device that backbone is on
        if self.sketchprocessor is not None:
            autosketch = self.sketchprocessor(sketches)
        else:
            autosketch = sketches.clone()
        with torch.no_grad():
            # initialize action from Gaussian noise
            noise = torch.randn([len(autosketch),*self.shape], device=device, dtype=torch.float32)
            ntrajs = noise

            nconds = torch.ones((len(ntrajs),self.embed_dim), dtype=torch.float32, device=device)
            nconds[:,:-1] = self.encoder(autosketch, images)
            nconds[:,-1] = condval

            # init scheduler
            self.noise_scheduler.set_timesteps(self.diffusion_steps)
            for k in self.noise_scheduler.timesteps:
                # Cat to predict both condtioned and unconditioned noise
                tntrajs = torch.cat([ntrajs,ntrajs], dim=0)
                tnconds = torch.cat([nconds,0*nconds], dim=0)
                # predict noise
                noise_pred = self.inference_model(
                    sample=tntrajs,
                    timestep=k,
                    global_cond=tnconds)

                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred_guided = noise_pred_uncond + guidance_scale*(noise_pred_cond - noise_pred_uncond)

                # inverse diffusion step (remove noise)
                ntrajs = self.noise_scheduler.step(
                    model_output=noise_pred_guided,
                    timestep=k,
                    sample=ntrajs
                ).prev_sample
        return self.unnormaliseTraj(ntrajs)

    def stochastic_forward(self, images, sketches, guidance_scale=1.0, costgrad=None, costgrad_scale=0.0, M=2):
        self.encoder.eval()
        self.inference_model.eval()
        device = next(self.model.parameters()).device

        if self.sketchprocessor is not None:
            autosketch = self.sketchprocessor(sketches)
        else:
            autosketch = sketches.clone()

        with torch.no_grad():
            # Initialize plan τ_N ~ N(0, I)
            ntrajs = torch.randn([len(autosketch), *self.shape], device=device)

            # Conditioning
            nconds = torch.ones((len(ntrajs),self.embed_dim), dtype=torch.float32, device=device)
            nconds[:,:-1] = self.encoder(autosketch, images)
            nconds[:,-1] = 0.0

            # Setup scheduler
            self.noise_scheduler.set_timesteps(self.diffusion_steps)
            timesteps = self.noise_scheduler.timesteps

            for i, timestep in enumerate(timesteps):
                for j in range(M):
                    # ε ← π_θ(τᵢ)
                    noise_pred = self.inference_model(
                        sample=ntrajs,
                        timestep=timestep,
                        global_cond=nconds,
                    )

                    # δ ← ∇ξ(τᵢ, z)
                    if costgrad is not None:
                        fulltraj = self.unnormaliseTraj(ntrajs)
                        delta = costgrad(fulltraj, autosketch, images) * (self.trajmax - self.trajmin) # scaling correction
                    else:
                        delta = 0.0

                    # Scale alignment gradient
                    beta_i = guidance_scale
                    delta = beta_i * delta

                    # Reverse step (denoising)
                    reverse_input = noise_pred + costgrad_scale * delta
                    next_step = timestep if j < M - 1 else timesteps[i + 1] if i + 1 < len(timesteps) else timestep
                    ntrajs = self.noise_scheduler.step(
                        model_output=reverse_input,
                        timestep=timestep if j < M - 1 else next_step,
                        sample=ntrajs,
                    ).prev_sample

        return self.unnormaliseTraj(ntrajs)


    def pred_noise(self, noisy_actions, timesteps, images, sketches, condswitches):
        # Take image, sketch and cond number and encode them.
        # Do tanh
        # add cond number to end.
        if self.projprocessor is not None:
            sketches = self.projprocessor(sketches)
        else:
            sketches = sketches.clone()

        device = next(self.model.parameters()).device # Hacky way to get device that backbone is on
        self.shape = noisy_actions[0].shape[-2:]
        nconds = torch.ones((len(noisy_actions),self.embed_dim), dtype=torch.float32, device=device)
        nconds[:,:-1] = self.encoder(sketches, images)
        nconds[:,-1] = condswitches
        nconds = (nconds.T * (condswitches != 0)).T
        noise_pred = self.model(noisy_actions, timesteps, global_cond=nconds)
        return noise_pred
    
    def freezeNonCond(self):
        self.model.freezeNonCond()
        self.encoder.freezeconv()

    def normaliseTraj(self, trajs):
        return 2*(trajs - self.trajmin)/(self.trajmax-self.trajmin) - 1

    def unnormaliseTraj(self, normtrajs):
        return self.trajmin + (normtrajs + 1) * (self.trajmax - self.trajmin) / 2
