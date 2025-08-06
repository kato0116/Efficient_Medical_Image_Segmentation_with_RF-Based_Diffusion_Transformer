import torch
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch import nn
from models.modules import CrossAttention


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim,dropout=0, useconv=False):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        
        if in_ch == out_ch:
            self.skip_connection = nn.Identity()
        elif useconv:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(out_ch, out_ch, 3, padding=1)
            ),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_ch),
        )

        self.time_emb = nn.Linear(time_embed_dim, 2*out_ch)

    def forward(self, x, v):
        h = self.in_layer(x)
        if v is not None:
            emb_out = self.time_emb(v)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            
            out_norm, out_rest = self.out_layer[0], self.out_layer[1:]
            shift, scale = emb_out.chunk(2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layer(h)
        return h+self.skip_connection(x)
    
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.down  = nn.MaxPool2d(2)
        self.convs = ResBlock(in_ch, out_ch, time_embed_dim)
    
    def forward(self, x, v):
        x = self.down(x)
        x = self.convs(x, v)
        return x
    
class UnetEnc(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim, hiddn_dim=128):
        super().__init__()
        self.conv_in  = nn.Conv2d(in_ch, hiddn_dim, 3, padding=1)
        self.in_layer = ResBlock(hiddn_dim, hiddn_dim, time_embed_dim)       
        self.down1 = DownBlock(hiddn_dim, hiddn_dim*2, time_embed_dim)
        self.down2 = DownBlock(hiddn_dim*2, hiddn_dim*4, time_embed_dim)
        self.down3 = DownBlock(hiddn_dim*4, hiddn_dim*8, time_embed_dim)
    def forward(self, x):
        v  = None
        x  = self.conv_in(x) 
        x0 = self.in_layer(x, v) 
        x1 = self.down1(x0, v)
        x2 = self.down2(x1, v)
        x = self.down3(x2, v)
        return  x    

  
#################################################################################
#                                 Core SSDiT Model                                #
#################################################################################

class SSDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False, cross_attn=False,shared_step=True, **block_kwargs):
        super().__init__()
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None
        self.norm1       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn        = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_attn  = CrossAttention(hidden_size, hidden_size, num_heads=num_heads, qkv_bias=True, dropout=0.) if cross_attn else None
        self.norm2       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim   = int(hidden_size * mlp_ratio)
        approx_gelu      = lambda: nn.GELU(approximate="tanh")
        self.mlp         = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,out_features=hidden_size, act_layer=approx_gelu, drop=0)
        # 正規化パラメータの共有するかどうか
        if shared_step:
            self.adaLN_modulation = None
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
    def forward(self, x, t, y, skip=None):
        if self.adaLN_modulation is not None:
            t = self.adaLN_modulation(t) # 正規化パラメータを計算
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t.chunk(6, dim=1)

        if self.skip_linear is not None:
            y = self.skip_linear(torch.cat([y, skip], dim=-1))
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # 条件付け部
        if self.cross_attn is not None:
            x = x + self.cross_attn(x, y)
        else:
            x = x + y
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(torch.cat([modulate(self.norm2(x), shift_mlp, scale_mlp),y],dim=-1)) 
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) 

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear     = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SSDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        cond_in_channels=3,
        hidden_size=1152,
        unet_hidden_size=128,
        time_embed_dim=256,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        skip_flag = False,
        cross_attn_flag = False,
        shared_step = True,
        y_diff_flag = False,
    ):
        super().__init__()
        self.learn_sigma  = learn_sigma
        self.in_channels  = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size   = patch_size
        self.num_heads    = num_heads
        self.skip_flag    = skip_flag
        self.cross_attn_flag = cross_attn_flag
        self.y_diff_flag  = y_diff_flag
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if self.y_diff_flag:
            cond_in_channels = cond_in_channels * 2
        self.y_encoder  = UnetEnc(cond_in_channels, in_channels, time_embed_dim=hidden_size, hiddn_dim=unet_hidden_size)
        self.y_embedder = PatchEmbed(input_size, patch_size, unet_hidden_size*8, hidden_size, bias=True)
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        cond_patches = self.y_embedder.num_patches
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, cond_patches, hidden_size), requires_grad=False) # 追加
        self.shared_step = shared_step
        # 正規化パラメータの共有
        if shared_step:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = None
        
        self.encoder_blocks = nn.ModuleList([
            SSDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,skip=False, cross_attn=cross_attn_flag, shared_step=shared_step) for _ in range(depth//2)
        ])
        self.middle_block = SSDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,skip=False,cross_attn=cross_attn_flag, shared_step=shared_step)
        self.decoder_blocks = nn.ModuleList([
            SSDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,skip=skip_flag, cross_attn=cross_attn_flag, shared_step=shared_step) for _ in range(depth//2)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        # y 用

        cond_pos_embed = get_2d_sincos_pos_embed(self.cond_pos_embed.shape[-1], int(self.y_embedder.num_patches ** 0.5))
        self.cond_pos_embed.data.copy_(torch.from_numpy(cond_pos_embed).float().unsqueeze(0)) # 追加
        
        w = self.y_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.y_embedder.proj.bias, 0)
        
        # # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # 正規化パラメータを共有する場合
        if self.shared_step:
            # Zero-out adaLN modulation layer
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.encoder_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.middle_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.middle_block.adaLN_modulation[-1].bias, 0)
            for block in self.decoder_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        if self.cross_attn_flag:
            # cross_attnの出力層をゼロにする
            for block in self.encoder_blocks:
                nn.init.constant_(block.cross_attn.to_out[0].weight, 0)
                nn.init.constant_(block.cross_attn.to_out[0].bias, 0)
            nn.init.constant_(self.middle_block.cross_attn.to_out[0].weight, 0)
            nn.init.constant_(self.middle_block.cross_attn.to_out[0].bias, 0)
            for block in self.decoder_blocks:
                nn.init.constant_(block.cross_attn.to_out[0].weight, 0)
                nn.init.constant_(block.cross_attn.to_out[0].bias, 0)
        
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, conditioned_image,y_diff=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of mask images
        t: (N,) tensor of diffusion timesteps
        y: (N, C, H, W) tensor of medical images
        """
        t = self.t_embedder(t)                   # (N, D)
        # 正規化パラメータを一度のみ計算 (全ブロックで共有)
        if self.shared_step:
            shared_t = self.adaLN_modulation(t)          # (N, 6D)
        else:
            shared_t = t
        if self.y_diff_flag:
            y = torch.cat([conditioned_image, y_diff], dim=1)
        else:
            y = conditioned_image
        y = self.y_encoder(y)  # enc_hidden×32×32
    
        y = self.y_embedder(y) + self.cond_pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        x = self.x_embedder(x) + self.pos_embed       # (N, T, D), where T = H * W / patch_size ** 2
        if self.skip_flag:
            skip_x = []
        for block in self.encoder_blocks:
            x = block(x, shared_t, y, None)
            if self.skip_flag:
                skip_x.append(x)
        x = self.middle_block(x, shared_t, y, None)
        for block in self.decoder_blocks:
            if self.skip_flag:
                x = block(x, shared_t, y, skip_x.pop(-1))      # (N, T, D)
            else:
                x = block(x, shared_t, y, None)                # (N, T, D)

        x = self.final_layer(x, t)         # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)             # (N, out_channels, H, W)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

Model_Size = {
    "S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16}
}

# def main(args):
#     # image_encoder = get_network(args)
#     # model = SSDiT(image_encoder=image_encoder)
#     model = SSDiT()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     a = torch.randn(1, 4, 32, 32).to(device)
#     y = torch.randn(1, 3, 256, 256).to(device)
#     t = torch.tensor([1]).to(device)

#     out = model(a, t, y)
#     print(out.shape)

# import argparse
# if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # # samの設定
    # parser.add_argument('--use_sam_encoder', type=bool, default=True)
    # parser.add_argument('--sam_net', type=str, default='efficient_sam')              # sam or efficient_sam
    # parser.add_argument('--encoder', type=str, default='vit_s', help='encoder name') # 'default','vit_b','vit_l','vit_h' # effisient_sam: vit_s
    # parser.add_argument('--sam_ckpt', type=str, default="/root/volume/sam/efficient_sam_vits.pt",help='checkpoint for sam')  # /root/volume/sam/efficient_sam_vits.pt
    # parser.add_argument('--mod', type=str, default='sam_adpt') # sam_adpt or normal
    # parser.add_argument('--mid_dim', type=int, default=None)
    # parser.add_argument('--thd', type=bool, default=False)
    # parser.add_argument('--sam_input_size', type=int, default=512)
    # args = parser.parse_args()
    # main(args)