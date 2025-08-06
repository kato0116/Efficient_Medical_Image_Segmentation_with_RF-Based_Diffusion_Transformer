import os
import matplotlib.pyplot as plt
import torch
import torchvision
from collections import OrderedDict

def create_folder(dir_path):
    """
    in
    ==============================
    dir_path:   ディレクトリのパス

    out
    ==============================
    log記録用のディレクトリ作成
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    imgs_dir_path = os.path.join(dir_path,"imgs")
    if not os.path.exists(imgs_dir_path):
        os.mkdir(imgs_dir_path)
        
    weights_dir_path = os.path.join(dir_path,"weights")
    if not os.path.exists(weights_dir_path):
        os.mkdir(weights_dir_path)
        
def save_imgs(imgs, epoch,time, dir_path, nrow=3):
    """
    in
    ==============================
    imgs: 画像
    epoch: 現在のエポック数
    dir_path: ディレクトリのパス
    
    out
    ==============================
    画像保存
    """
    grid_imgs = torchvision.utils.make_grid(imgs, nrow=nrow,padding=2,normalize=True)
    grid_imgs = grid_imgs.cpu()
    plt.imshow(grid_imgs.permute(1,2,0))
    plt.axis('off')
    plt.show()
    imgs_path = os.path.join(dir_path,"imgs",f"epoch_{epoch}_time_{time}.png")
    plt.savefig(imgs_path)

def save_model(model,epoch,dir_path, optimizer=None):
    """
    in
    ===============================
    model: モデルの重み
    epoch: 現在のエポック数
    dir_path: ディレクトリを指定
    
    out
    ===============================
    重み保存
    """
    model_path = os.path.join(dir_path,"weights",f"weight_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    if optimizer is not None:
        optimizer_path = os.path.join(dir_path,"weights",f"optimizer_epoch_{epoch}.pth")
        torch.save(optimizer.state_dict(), optimizer_path)

def load_model(model,model_path):
    """
    in
    ===============================
    model: モデルの重み
    mode_path: モデルのパス
    """
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v  # 'module.' を取り除いて新しい辞書に追加
        else:
            new_state_dict[k] = v  # 'module.' が含まれていない場合、そのまま新しい辞書に追加
            
    model.load_state_dict(new_state_dict)
    return model