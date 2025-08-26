import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_shape, num_channels, patch_size, depth, hidden_dim, num_heads, mlp_dim, num_classes):
        super(VisionTransformer, self).__init__()

        num_patches = (img_shape // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size # rgb*ps*ps

        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1,num_patches+1,hidden_dim))

        self.transformer = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, num_heads=num_heads),
                'norm1': nn.LayerNorm(hidden_dim),
                'mlp':  nn.Sequantial(
                    nn.Linear(hidden_dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, hidden_dim)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(depth)
        ])

