import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_shape, patch_size, depth, hidden_dim, num_heads, mlp_dim, num_classes):
        super(VisionTransformer, self).__init__()

        self.img_shape = img_shape
        self.patch_size = patch_size

        num_patches = (img_shape // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size # rgb*ps*ps

        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1,num_patches+1,hidden_dim))

        self.transformer = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_dim),
                'attn': nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True),               'norm1': nn.LayerNorm(hidden_dim),
                'mlp':  nn.Sequential(
                    nn.Linear(hidden_dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, hidden_dim)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(depth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes) # to logits vector
        )
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        ps  = self.patch_size

        B = x.shape[0]
        x = x.unfold(2,ps,ps).unfold(3,ps,ps)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, ps*ps*3)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls_tokens, x], dim=1)

        x += self.pos_embed[:, : x.size(1)]

        for layer in self.transformer:
               residual = x
               x = layer['norm1'](x)
               attn_output, _ = layer['attn'](x, x, x)
               x = residual + attn_output
               residual = x
               x = layer['norm2'](x)
               x = residual + layer['mlp'](x)
        x = self.mlp_head(x[:, 0])

        return x
