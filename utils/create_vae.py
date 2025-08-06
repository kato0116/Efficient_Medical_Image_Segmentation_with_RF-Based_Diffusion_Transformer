from diffusers import AutoencoderKL
import torch
import torch.nn as nn
from utils.lora import insert_lora_adapters

def create_vae(args):
    # 学習済みモデル
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    in_out_channel = args.vae_in_out_ch  
        
    if in_out_channel != 3:
        # エンコーダ部の調整
        vae.encoder.conv_in = nn.Conv2d(in_out_channel, 128, kernel_size=3, stride=1, padding=1)
        # デコーダ部の調整
        vae.decoder.conv_out = nn.Conv2d(128, in_out_channel, kernel_size=3, stride=1, padding=1)
    # LoRAの挿入
    if args.vae_mode == "adapter":
        vae = insert_lora_adapters(vae, rank=args.lora_rank, alpha=args.lora_alpha)
    # モデル全体のパラメータをフリーズ
    for param in vae.parameters():
        param.requires_grad = False
    return vae
    
        
