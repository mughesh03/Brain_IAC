import torch
import torch.nn as nn
from monai.networks.nets import ViT


class ViTBackboneNet(nn.Module):
    def __init__(self, simclr_ckpt_path):
        super(ViTBackboneNet, self).__init__()

        self.backbone = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=True,
        )

        ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                new_key = key[9:]
                backbone_state_dict[new_key] = value

        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        print("Backbone weights loaded!!")

    def forward(self, x):
        features = self.backbone(x)
        # First token. This checkpoint has no separate class token.
        return features[0][:, 0]
