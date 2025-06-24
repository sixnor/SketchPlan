import torch
import torch.nn as nn
from transformers import AutoModel

class DinoEncoder(nn.Module):
    def __init__(self, output_dim, dropout_prob=0.0, dinomodel="facebook/dinov2-with-registers-small", cls_only=True):
        """
        Args:
            output_dim (int): The final output dimension.
            dinov2_model (nn.Module): A pretrained DINOv2 model (e.g., loaded via AutoModel.from_pretrained).
        """
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained(dinomodel, attn_implementation="sdpa", torch_dtype=torch.float32, mask_ratio=0.0)
        # Freeze dinov2 if you don't want to update its parameters:
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        # This linear layer is applied patch-wise to convert 384-dim tokens to 20-dim.
        self.patch_linear = nn.Linear(768, 20)
        self.relu = nn.ReLU()
        # Dropout layer applied after patch-wise transformation.
        self.dropout_patch = nn.Dropout(dropout_prob)
        # Final linear layer maps flattened patches (256*20) to the desired output_dim.
        self.final_linear = nn.Linear(196 * 20, output_dim)
        self.dropout_final = nn.Dropout(dropout_prob)
        # ImageNet normalization buffers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, 1, 224, 224) with float32 values.
        Returns:
            Tensor: Output tensor of shape (B, output_dim)
        """
        with torch.no_grad():
            # Step 1: Replicate channel dimension from 1 to 3.
            x = x.repeat(1, 3, 1, 1)  # now (B,3,224,224)
            
            # Step 2: Normalize image using ImageNet mean and std.
            x = (x - self.mean) / self.std
            
            # Step 3: Pass through dinov2. The dinov2 model is expected to take a keyword argument "pixel_values".
            outputs = self.dinov2(pixel_values=x)
            # outputs.last_hidden_state is assumed to be of shape (B, N, 384).
            tokens = outputs.last_hidden_state
            
            # Step 4: Discard the last 5 tokens, so that we end up with 256 tokens.
            tokens = tokens[:, 1:, :]  # shape becomes (B, 256, 384)
            
        # Step 5: Apply the same linear layer + ReLU to each patch token.
        # This is done by applying it to the last dimension.
        patch_features = self.patch_linear(tokens)  # (B, 256, 20)
        patch_features = self.relu(patch_features)
        patch_features = self.dropout_patch(patch_features)
        
        # Step 6: Flatten patch features and apply final linear layer.
        # Flatten from (B, 256, 20) to (B, 256*20)
        flattened = patch_features.view(patch_features.size(0), -1)
        flattened = self.dropout_final(flattened)
        output = self.final_linear(flattened)  # (B, output_dim)
        
        return output
    
class DinoCLSEncoder(nn.Module):
    def __init__(self, dinomodel="facebook/dinov2-with-registers-small", mode="cls"):
        """
        Args:
            output_dim (int): The final output dimension.
            dinov2_model (nn.Module): A pretrained DINOv2 model (e.g., loaded via AutoModel.from_pretrained).
        """
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained(dinomodel, attn_implementation="sdpa", torch_dtype=torch.float32)
        self.mode = mode
        # Freeze dinov2 if you don't want to update its parameters:
        for param in self.dinov2.parameters():
            param.requires_grad = False
        

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, 1, 224, 224) with float32 values.
        Returns:
            Tensor: Output tensor of shape (B, output_dim)
        """
        datadevice = x.device
        modeldevice = next(self.dinov2.parameters()).device # Hacky way to get device that model is on
        x = x.to(modeldevice)
        with torch.no_grad():
            # Step 1: Replicate channel dimension from 1 to 3.
            x = x.repeat(1, 3, 1, 1)  # now (B,3,224,224)
            # Step 3: Pass through dinov2. The dinov2 model is expected to take a keyword argument "pixel_values".
            outputs = self.dinov2(pixel_values=x)
            # outputs.last_hidden_state is assumed to be of shape (B, N, 384).
            tokens = outputs.last_hidden_state
            if self.mode == "cls":
            # Step 4: Discard the last 5 tokens, so that we end up with 256 tokens.
                tokens = tokens[:, 0, :]  # shape becomes (B, 768)
            elif self.mode == "mean":
                tokens = torch.mean(tokens[:, 5:, :],dim=-2)  # shape becomes (B, 768)
            elif self.mode == "all":
                tokens = torch.cat([tokens[:,0:1], tokens[:,5:]],dim=-2)
            else:
                raise NotImplementedError
        tokens = tokens.to(datadevice)
        return tokens



class GeneralEnc(nn.Module):
    def __init__(self, sketchEnc, imageEnc):
        super().__init__()
        self.sketchEnc = sketchEnc
        self.imageEnc = imageEnc
    def forward(self, sketches, images):
        return torch.cat([self.sketchEnc(sketches), self.imageEnc(images)],dim=-1)
    
class flatEnc(nn.Module): # just flattens
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.flatten(inp,start_dim=1)