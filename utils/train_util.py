import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch as th
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import torch.nn as nn
from guided_diffusion.dist_util import cleanup, init_process
from utils.utils import requires_grad, update_ema
from utils.strage import save_model, load_model
from metric.metric import dice, jaccard, sensitivity, specificity, accuracy
from skimage.filters import threshold_otsu
from copy import deepcopy
import wandb
from diffusers.models import AutoencoderKL
import torch.nn.functional as F
import numpy as np
import random
from dataset.refuge2_dataset import fundus_inv_map_mask
from utils.create_vae import create_vae
from zigzag_segmentation import z_sampling
def set_seed(seed):
    th.manual_seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # GPUの乱数も固定（必要なら）
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)  # マルチGPU用
        th.backends.cudnn.deterministic = True  # CuDNNの動作を固定
        th.backends.cudnn.benchmark = False  # 再現性を優先

class Trainer:
    def __init__(
        self,
        *,  # *以降は呼び出す際にキーワード引数で指定する必要がある(ex Trainer(a=1, b=2))
        model,
        diffuser,
        optimizer,
        train_set,
        val_set,
        test_set,
        args,
        dir_path,
        scheduler=None,
        mode="latent",
        parameter_num=None,
        trainable_param=None,
        multi_gpu=False,
    ):
        # シードを固定
        # set_seed(args.seed)
        self.multi_gpu     = multi_gpu
        self.microbatch    = args.microbatch
        if multi_gpu:
            self.rank, self.device, self.seed = init_process(args)
            self.model = DDP(
                model.to(self.device),
                device_ids=[self.rank],
            )
            self.train_sampler, self.train_loader = self.get_loader(dataset=train_set, shuffle=True,  seed=args.global_seed, batch_size=args.global_batch_size, num_workers=args.num_workers, drop_lat=True)
            self.val_sampler, self.val_loader     = self.get_loader(dataset=val_set,   shuffle=False, seed=args.global_seed, batch_size=args.microbatch, num_workers=args.num_workers, drop_lat=False)
            self.test_sampler, self.test_loader   = self.get_loader(dataset=test_set,  shuffle=False, seed=args.global_seed, batch_size=1, num_workers=args.num_workers, drop_lat=False)
        else:
            self.rank   = 0
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
            self.model  = model.to(self.device)
            self.train_loader = DataLoader(train_set, batch_size=args.global_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            self.val_loader   = DataLoader(val_set, batch_size=args.microbatch, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
            self.test_loader  = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        
        self.diffuser_type = args.diffuser_type # diffusion, rectified_flow, flow_matching
        self.diffuser      = diffuser

        self.img_size      = args.img_size
        self.in_channels   = args.in_channels
        self.num_ensemble  = args.num_ensemble
        self.test_ensemble = args.test_ensemble
        self.optimizer     = optimizer
        self.lr            = args.lr
        self.scheduler     = scheduler
        self.epochs        = args.epochs

        self.wandb_flag        = args.wandb_flag
        self.wandb_num_images  = args.wandb_num_images
        self.save_n_model      = args.save_n_model
        self.val_step_num      = args.val_step_num
        self.dir_path          = dir_path

        self.mode              = mode
        self.use_ema           = args.use_ema
        self.clip_grad         = args.clip_grad
        self.best_dice         = 0
        
        self.scaling_factor    = args.scaling_factor

        self.best_path         = os.path.join(dir_path,"weights",f"weight_epoch_best_dice.pth")
        self.ema_best_path     = os.path.join(dir_path,"weights",f"weight_epoch_best_dice_ema.pth")
        self.last_path         = os.path.join(dir_path,"weights",f"weight_epoch_last.pth")
        self.ema_last_path     = os.path.join(dir_path,"weights",f"weight_epoch_last_ema.pth")

        self.train_size        = args.train_size
        self.val_size          = args.val_size
        self.test_size         = args.test_size

        self.use_stability_vae = args.use_stability_vae
        self.use_tanh          = args.use_tanh
        self.parameter_num     = parameter_num
        self.trainable_param   = trainable_param
        self.vae_in_out_ch     = args.vae_in_out_ch
        # vaeの用意
        if self.mode == "latent":
            if self.use_stability_vae:
                vae = create_vae(args)
            if args.vae_checkpoint:
                print("Reading the VAE checkpoint...")
                print(args.vae_path)
                vae = load_model(vae,args.vae_path)
            requires_grad(vae, False)
            self.vae = vae.to(self.device)

        if self.use_ema:
            print("Using EMA Model")
            self.ema = deepcopy(self.model).to(self.device)  # Create an EMA of the model for use after training
            requires_grad(self.ema, False)
        
        self.checkpoint = args.checkpoint
        self.checkpoint_epoch = 0
        if self.checkpoint:
            print("Reading the checkpoint...")
            self.checkpoint_epoch = args.checkpoint_epoch
            checkpoint_dir_path = os.path.join(args.path,"log","fold:"+args.fold+"_"+args.model_name+"_epochs_"+str(args.checkpoint_epoch)+":"+args.dataset)
            checkpoint_path     = os.path.join(checkpoint_dir_path,"weights",f"weight_epoch_{args.checkpoint_epoch}.pth")
            self.model          = load_model(model, checkpoint_path).to(self.device)
            self.ema_checkpoint_path = os.path.join(checkpoint_dir_path,"weights",f"weight_epoch_{args.checkpoint_epoch}_ema.pth")

    def train(self, args):
        if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1"  or args.dataset == "REFUGE1_Crop":
            name = args.model_name + "_"+str(args.img_size)
        else:
            name = args.model_name + "_fold:" + args.fold
        if self.wandb_flag:
            if self.rank == 0:
                wandb.init(
                    project=args.project_name,
                    name=name,
                    tags=[args.dataset, args.cond, self.mode,args.model_size,"fold:"+args.fold],
                    notes=args.notes,
                    config={
                        "model":            args.model_name,
                        "epochs":           args.epochs,
                        "image_size":       args.img_size,
                        "channel":          args.in_channels,
                        "cond_channel":     args.cond_channels,
                        "batch_size":       args.global_batch_size,
                        "learning_rate":    args.lr,
                        "predict_xstart":   args.predict_xstart,
                        "noise_schedule":   args.noise_schedule,
                        "learn_sigma":      args.learn_sigma,
                        "ddim":             args.ddim,
                        "num_ensemble":     args.num_ensemble,
                        "patch_size":       args.patch_size,
                                    
                        "clip_grad":        args.clip_grad,
                        
                                    
                        "checkpoint":       args.checkpoint,
                        "checkpoint_epoch": args.checkpoint_epoch,
                                    
                        "use_ema":          self.use_ema,
                        "use_tanh":         self.use_tanh,
                        
                        "skip_flag":        args.skip_flag,
                        "cross_attn_flag":  args.cross_attn_flag,
                        "shared_step":      args.shared_step,
                        "image_diff":       args.image_diff,

                        "diffuser_type":    args.diffuser_type,
                           
                        "use_stability_vae": self.use_stability_vae,
                        "vae_in_out_ch":     args.vae_in_out_ch,
                        "vae_checkpoint":    args.vae_checkpoint,
                        "vae_path":          args.vae_path,
                        "vae_mode":          args.vae_mode,
                        "latent_sample":     args.latent_sample,
                        
                        "lora_rank":         args.lora_rank,
                        "lora_alpha":        args.lora_alpha,
                        "scaling_factor":    args.scaling_factor,
                                    
                        "euler_step":        args.euler_step,
                        "euler_ensemble":    args.euler_ensemble,
                        "sampler_type":      args.sampler_type if args.diffuser_type == "rectified_flow" else "DDIM",

                        "fold":              args.fold,
                        "param_num":         self.parameter_num,
                        "trainable_param":   self.trainable_param,
                        "train_size":        args.train_size,
                        "val_size":          args.val_size,
                        "test_size":         args.test_size,
                        "seed":              args.seed, 
                        
                        "time_sampler":      args.time_sampler,
                    }
                )


        if self.use_ema:        # EMAを使う場合
            if self.checkpoint: # checkpointがある場合
                print("Reading the EMA checkpoint...")
                self.ema = load_model(self.ema, self.ema_checkpoint_path).to(self.device)
                requires_grad(self.ema, False)
            else:
                update_ema(self.ema, self.model, decay=0)
            self.ema.eval()
            
        self.model.train()
        for epoch in range(self.checkpoint_epoch+1, self.epochs + 1):
            if self.rank == 0:
                print(f"epoch:{epoch}/{self.epochs}")
                train_losses       = []
                val_dices_otsu     = []
                val_ious_otsu      = []
                
                val_disc_dice_otsu = []
                val_disc_iou_otsu  = []
                val_cup_dice_otsu  = []
                val_cup_iou_otsu   = []
                # val_sensitivities_otsu = []
                # val_specificities_otsu = []
                # val_accuracies_otsu = []
            # 学習  
            for batch in tqdm(self.train_loader):
                image, mask = batch
                self.optimizer.zero_grad() # 勾配の初期化
                for i in range(0, mask.shape[0], self.microbatch):
                    micro_mask  = mask[i:i+self.microbatch]
                    micro_image = image[i:i+self.microbatch]

                    micro_mask  = micro_mask.to(self.device)
                    x_start     = micro_mask.to(self.device)  # mask画像
                    y = micro_image.to(self.device)
                    if args.image_diff:
                        with th.no_grad():
                            z_y = self.vae.encode(x = y).latent_dist.sample()
                            recon_y = self.vae.decode(z_y).sample
                            y_diff = (y-recon_y)**2
                        model_kwargs = dict(conditioned_image=y, y_diff=y_diff)
                    else:
                        model_kwargs = dict(conditioned_image=y)
                    # 潜在空間への変換
                    if   self.mode == "latent": 
                        with th.no_grad():
                            if self.use_stability_vae:
                                x_start = self.vae.encode(x = x_start).latent_dist
                                if args.latent_sample == "mean":
                                    x_start = x_start.mode()
                                elif args.latent_sample == "sample":
                                    x_start = x_start.sample()
                            if self.use_tanh:
                                x_start = th.tanh(x_start)
                            else:
                                x_start = x_start.mul_(self.scaling_factor)            
                    if   self.diffuser_type == "diffusion":
                        t = th.randint(0, self.diffuser.num_timesteps, (x_start.shape[0],), device=self.device)
                        loss_dict = self.diffuser.training_losses(self.model, x_start, t, model_kwargs)
                        loss = loss_dict["loss"]
                    elif self.diffuser_type == "rectified_flow":
                        noise = th.randn_like(x_start)
                        loss = self.diffuser.train_losses(self.model, z0=noise, z1=x_start,model_kwargs=model_kwargs)
                    elif self.diffuser_type == "flow_matching":
                        print("Not implemented yet.")
                        exit()
                    loss.backward() # 勾配の計算
                    if self.rank == 0:
                        train_losses.append(loss.item()*len(x_start))
                if self.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 勾配クリッピング
                self.optimizer.step()
                if self.use_ema:
                    update_ema(self.ema, self.model)
            
            if self.scheduler is not None:
                    self.scheduler.step()
            if self.multi_gpu:
                self.val_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)
                dist.barrier()

            # 検証
            if self.use_ema:
                self.ema.eval()
            self.model.eval()
            if epoch % self.val_step_num == 0:
                for image, mask in tqdm(self.val_loader):
                    x_start = mask.to(self.device)
                    y = image.to(self.device)
                    if args.image_diff:
                        with th.no_grad():
                            z_y = self.vae.encode(x = y).latent_dist.sample()
                            recon_y = self.vae.decode(z_y).sample
                            y_diff = (y-recon_y)**2
                        model_kwargs = dict(conditioned_image=y, y_diff=y_diff)
                    else:   
                        model_kwargs = dict(conditioned_image=y)
                    pred_x_start_list = []
                    # アンサンブルの平均を計算
                    for _ in range(self.num_ensemble):
                        if self.mode == "latent":
                            with th.no_grad():
                                if self.use_stability_vae:
                                    z_start = self.vae.encode(x = x_start).latent_dist.sample()
                                x_end = th.randn_like(z_start).to(self.device)
                        else:
                            x_end = th.randn_like(x_start).to(self.device)
            
                        if self.diffuser_type == "diffusion":
                            pred_x_start = self.diffuser.ddim_sample_loop(
                                self.ema if self.use_ema else self.model, # EMAを使うかどうか
                                x_end.shape,
                                noise=x_end,
                                model_kwargs=model_kwargs,
                                clip_denoised=True,
                            )
                        elif self.diffuser_type == "rectified_flow":
                            pred_x_start = self.diffuser.sampler(
                                self.ema if self.use_ema else self.model, 
                                x_end.shape, 
                                self.device, 
                                model_kwargs=model_kwargs
                            )
                        elif self.diffuser_type == "flow_matching":
                            print("Not implemented yet.")
                            exit()
                        
                        if self.mode == "latent":
                            with th.no_grad():
                                if self.use_stability_vae:
                                    if self.use_tanh:
                                        pred_x_start = self.vae.decode(pred_x_start).sample
                                    else:
                                        pred_x_start = self.vae.decode(pred_x_start/self.scaling_factor).sample
                                if args.ch1_ch3:
                                    pred_x_start = th.mean(pred_x_start, dim=1)
                                    pred_x_start = pred_x_start.unsqueeze(1)
                        pred_x_start_list.append(pred_x_start)
                    
                    # 平均、標準偏差を計算
                    mean_x_start = th.mean(th.stack(pred_x_start_list), dim=0)  # emsembleの平均
                    std_x_start  = th.std(th.stack(pred_x_start_list), dim=0)  # emsembleの標準偏差
                    if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                        if args.split_ODOC == "optic_cup" or args.split_ODOC == "optic_disc":
                            # 3チャネルを平均
                            mean_x_start = th.mean(mean_x_start, dim=1).unsqueeze(1)
                            # 閾値処理
                            threshold                = threshold_otsu(mean_x_start.cpu().numpy())
                            mean_x_start_binary_otsu = (mean_x_start > threshold).float()
                        else:
                            mean_x0_start = mean_x_start[:,0]
                            mean_x1_start = mean_x_start[:,1]
                            mean_x2_start = mean_x_start[:,2]
                            th0 = threshold_otsu(mean_x0_start.cpu().numpy())
                            th1 = threshold_otsu(mean_x1_start.cpu().numpy())
                            th2 = threshold_otsu(mean_x2_start.cpu().numpy())
                            mean_x0_start_binary_otsu = (mean_x0_start > th0).float()
                            mean_x1_start_binary_otsu = (mean_x1_start > th1).float()
                            mean_x2_start_binary_otsu = (mean_x2_start > th2).float()
                    else:    
                        # 閾値処理
                        threshold                = threshold_otsu(mean_x_start.cpu().numpy())
                        mean_x_start_binary_otsu = (mean_x_start > threshold).float()
                    
                    if self.multi_gpu:
                        mean_x_start_binary_otsu = gather_tensors(mean_x_start_binary_otsu)
                        x_start                  = gather_tensors(x_start)
                        std_x_start              = gather_tensors(std_x_start)
                        y                        = gather_tensors(y)
                    if self.rank == 0:
                        if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                            if args.split_ODOC == "optic_cup" or args.split_ODOC == "optic_disc":
                                mean_x_start_binary_otsu = mean_x_start_binary_otsu.cpu().numpy()
                                x_start                  = x_start[:,0].unsqueeze(1).cpu().numpy()
                                th_otsu_metric           = calc_metric(mean_x_start_binary_otsu, x_start)
                                val_dices_otsu.append(th_otsu_metric["dice"])
                                val_ious_otsu.append(th_otsu_metric["iou"])
                            else:
                                mean_x_start_binary_otsu = th.stack([mean_x0_start_binary_otsu, mean_x1_start_binary_otsu, mean_x2_start_binary_otsu], dim=1)
                                mean_x0_start_binary_otsu = mean_x0_start_binary_otsu.cpu().numpy()
                                mean_x1_start_binary_otsu = mean_x1_start_binary_otsu.cpu().numpy()
                                mean_x2_start_binary_otsu = mean_x2_start_binary_otsu.cpu().numpy()
                                

                                x_start  = x_start.cpu().numpy()
                                mean_x_start_binary_otsu = mean_x_start_binary_otsu.cpu().numpy()
                                th1_otsu_metric = calc_metric(mean_x1_start_binary_otsu, x_start[:,1])
                                th2_otsu_metric = calc_metric(mean_x2_start_binary_otsu, x_start[:,2])
                                
                                x_start = fundus_inv_map_mask(x_start)
                                mean_x_start_binary_otsu = fundus_inv_map_mask(mean_x_start_binary_otsu)
                                
                                val_disc_dice_otsu.append(th1_otsu_metric["dice"])
                                val_disc_iou_otsu.append(th1_otsu_metric["iou"])
                                val_cup_dice_otsu.append(th2_otsu_metric["dice"])
                                val_cup_iou_otsu.append(th2_otsu_metric["iou"])
                        else:
                            mean_x_start_binary_otsu = mean_x_start_binary_otsu.cpu().numpy()
                            if args.ch1_ch3:
                                x_start = x_start[:,0,:,:].unsqueeze(1).cpu().numpy()
                            else:
                                x_start = x_start.cpu().numpy()
                            th_otsu_metric = calc_metric(mean_x_start_binary_otsu, x_start)

                            val_dices_otsu.append(th_otsu_metric["dice"])
                            val_ious_otsu.append(th_otsu_metric["iou"])
                        # val_sensitivities_otsu.append(th_otsu_metric["sensitivity"])
                        # val_specificities_otsu.append(th_otsu_metric["specificity"])
                        # val_accuracies_otsu.append(th_otsu_metric["accuracy"])
            if self.wandb_flag and self.rank == 0:
                train_avg_loss = sum(train_losses) / self.train_size
                if epoch % self.val_step_num == 0:
                    wandb_num_images  = min(self.wandb_num_images, y.shape[0], x_start.shape[0], mean_x_start_binary_otsu.shape[0], std_x_start.shape[0])
                    wandb_y           = [wandb.Image(y[i]) for i in range(wandb_num_images)]
                    wandb_x           = [wandb.Image(x_start[i]) for i in range(wandb_num_images)]
                    wandb_pred_x_otsu = [wandb.Image(mean_x_start_binary_otsu[i]) for i in range(wandb_num_images)]
                    wandb_std_x = [wandb.Image(std_x_start[i]) for i in range(wandb_num_images)]
                    
                    if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                        if args.split_ODOC == "optic_cup" or args.split_ODOC == "optic_disc":
                            avg_metric = {
                                "dice": sum(val_dices_otsu) / self.val_size,
                                "iou":  sum(val_ious_otsu) / self.val_size,
                            }
                            print(f"best_dice:{self.best_dice}")
                            print(f"now_dice:{avg_metric['dice']}")
                            if self.best_dice < avg_metric["dice"]:
                                self.best_dice = avg_metric["dice"]
                                print(f"best_dice:{self.best_dice}更新")
                                save_model(self.model, "best_dice", self.dir_path)
                                if self.use_ema:
                                    save_model(self.ema, "best_dice_ema", self.dir_path)
                        else:
                            avg_metric = {
                                "disc_dice": sum(val_disc_dice_otsu) / self.val_size,
                                "disc_iou":  sum(val_disc_iou_otsu) / self.val_size,
                                "cup_dice":  sum(val_cup_dice_otsu) / self.val_size,
                                "cup_iou":   sum(val_cup_iou_otsu) / self.val_size,
                            }
                            print(f"best_cup_dice:{self.best_dice}")
                            print(f"now_cup_dice:{avg_metric['cup_dice']}")
                            # best modelの保存
                            
                            if self.best_dice < (avg_metric["cup_dice"]+avg_metric["disc_dice"])/2:
                                self.best_dice = (avg_metric["cup_dice"]+avg_metric["disc_dice"])/2
                                print(f"best_dice:{self.best_dice}更新")
                                save_model(self.model, "best_dice", self.dir_path)
                                if self.use_ema:
                                    save_model(self.ema, "best_dice_ema", self.dir_path)
                    else:
                        avg_metric = {
                            "dice": sum(val_dices_otsu) / self.val_size,
                            "iou":  sum(val_ious_otsu) / self.val_size,
                            # "sensitivity": sum(val_sensitivities_otsu) / self.val_size,
                            # "specificity": sum(val_specificities_otsu) / self.val_size,
                            # "accuracy":    sum(val_accuracies_otsu) / self.val_size,
                        }

                        print(f"best_dice:{self.best_dice}")
                        print(f"now_dice:{avg_metric['dice']}")

                        # best modelの保存
                        if self.best_dice < avg_metric["dice"]:
                            self.best_dice = avg_metric["dice"]
                            print(f"best_dice:{self.best_dice}更新")
                            save_model(self.model, "best_dice", self.dir_path)
                            if self.use_ema:
                                save_model(self.ema, "best_dice_ema", self.dir_path)
                    wandb_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        "train_loss": train_avg_loss,
                        "otsu": avg_metric,
                        "y": wandb_y,
                        "x": wandb_x,
                        "pred_x_otsu": wandb_pred_x_otsu,
                        "std_x": wandb_std_x,
                        "lr": wandb_lr
                    })
                else:
                    wandb_lr = self.optimizer.param_groups[0]['lr']
                    print(f"train_loss:{train_avg_loss}, not send image")
                    wandb.log({
                        "train_loss": train_avg_loss,
                        "lr": wandb_lr
                    })
            if self.multi_gpu:
                dist.barrier()
            if self.rank == 0:
                if epoch % self.save_n_model == 0:
                    save_model(self.model, epoch, self.dir_path)
                    if self.use_ema:
                        print("save ema model...")
                        save_model(self.ema, f"{epoch}_ema", self.dir_path)
                        
        if self.multi_gpu:
            dist.barrier()
            cleanup()

        if self.rank == 0:
            save_model(self.model, "last", self.dir_path)
            if self.use_ema:
                save_model(self.ema, "last_ema", self.dir_path)
            if self.wandb_flag:
                wandb.finish()


    def test(self, model, args):
        from utils.staple import staple
        staple_dice_list      = []
        staple_jaccad_list    = []
        test_dice_list        = []
        test_jaccad_list      = []
        test_disc_dice_list   = []
        test_disc_jaccad_list = []
        test_cup_dice_list    = []
        test_cup_jaccad_list  = []

        # best modelを読み込む
        if args.use_best_model:
            if args.use_ema:
                print("Reading the best EMA Model...")
                print(self.ema_best_path)
                path = self.best_path
            else:
                print("Reading the best model...")
                print(self.best_path)
                path = self.best_path
        else:
            if args.use_ema:
                print("Reading the last EMA Model...")
                print(self.ema_last_path)
                path = self.ema_last_path
            else:
                print("Reading the last model...")
                print(self.last_path)
                path = self.last_path
        model = load_model(model, path)

        if self.multi_gpu:
            self.rank, self.device, self.seed = init_process(args)
            self.model = DDP(
                model.to(self.device),
                device_ids=[self.rank],
            )
        else:
            self.rank = 0
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
            self.model = model.to(self.device)
        
        # vaeの用意
        if self.mode == "latent":
            if self.use_stability_vae:
                vae = create_vae(args)
            if args.vae_checkpoint:
                print("Reading the VAE checkpoint...")
                print(args.vae_path)
                vae = load_model(vae,args.vae_path)
            requires_grad(vae, False)
            self.vae = vae.to(self.device)
            
        if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
            name = args.model_name + "_"+str(args.img_size)
        else:
            name = args.model_name + "_fold:" + args.fold
        if args.use_best_model:
            name = name + "(best)"
        else:
            name = name + "(last)"
        
        if args.z_sampling:
            name = name + "_z_sampling"
            # from diffusers import DDIMScheduler, DDIMInverseScheduler
            # scheduler = DDIMScheduler(prediction_type="sample",)
            # inverse_scheduler = DDIMInverseScheduler(prediction_type="sample",)
            print("Using z_sampling")
        

        # wandbの初期化
        if self.wandb_flag:
            if self.rank == 0:
                wandb.init(
                    project=args.test_project_name,
                    name=name,
                    tags=[args.dataset, args.cond, self.mode,args.model_size,"fold:"+args.fold],
                    notes=args.notes,
                    config={
                        "model":         args.model_name,
                        "dataset":       args.dataset,
                        "image_size":    args.img_size,
                        "channel":       args.in_channels,

                        "epochs":           args.epochs,
                        "cond_channel":     args.cond_channels,
                        "batch_size":       args.global_batch_size,
                        "learning_rate":    args.lr,
                        
                        # 共通設定
                        "diffuser_type": args.diffuser_type,
                        "num_ensemble": args.test_ensemble,
                        # 拡散モデルの設定
                        "ddim": args.ddim,
                        "predict_xstart": args.predict_xstart,
                        "noise_schedule": args.noise_schedule,
                        "learn_sigma": args.learn_sigma,
                        # Rectified Flowの設定
                        "euler_step": args.euler_step,
                        "euler_ensemble": args.euler_ensemble,
                        "sampler_type":      args.sampler_type if args.diffuser_type == "rectified_flow" else "DDIM",                       
                        
                        "clip_grad":        args.clip_grad,
                        "use_ema":          self.use_ema,
                        
                        # モデル設定
                        "patch_size": args.patch_size,
                        "unet_hidden_size": args.unet_hidden_size,
                        "skip_flag":        args.skip_flag,
                        "cross_attn_flag":  args.cross_attn_flag,
                        "shared_step":      args.shared_step,
                        "image_diff":       args.image_diff,
                        
                        # VAEの設定
                        "use_stability_vae": self.use_stability_vae,
                        "vae_in_out_ch":     args.vae_in_out_ch,
                        "vae_checkpoint":    args.vae_checkpoint,
                        "vae_path":          args.vae_path,
                        "vae_mode":          args.vae_mode,
                        "latent_sample":     args.latent_sample,
                        
                        "lora_rank":         args.lora_rank,
                        "lora_alpha":        args.lora_alpha,
                        "scaling_factor":    args.scaling_factor,
                        
                        "z_sampling":        args.z_sampling,
                        "lambda_step":       args.lambda_step,
                        
                        "model_path": path,
                        "seed":     args.seed, 
                        
                        "fold": args.fold,
                        "param_num": self.parameter_num,
                        "trainable_param": self.trainable_param,
                        "train_size":args.train_size,
                        "val_size":args.val_size,
                        "test_size":args.test_size,
                        "time_sampler": args.time_sampler,           
                }
            )
        print("Evaluating on test dataset...")
        self.model.eval()
        for image, mask in tqdm(self.test_loader):
            x_start = mask.to(self.device)
            y       = image.to(self.device)
            
            if args.image_diff:
                with th.no_grad():
                    z_y = self.vae.encode(x = y).latent_dist.sample()
                    recon_y = self.vae.decode(z_y).sample
                    y_diff = (y-recon_y)**2
                model_kwargs = dict(conditioned_image=y, y_diff=y_diff)
            else:            
                model_kwargs = dict(conditioned_image=y)
            pred_x_start_list = []
            if self.mode == "latent":
                with th.no_grad():
                    if self.use_stability_vae:
                        x_start = self.vae.encode(x = x_start).latent_dist.sample()    
            # アンサンブル
            for _ in range(self.test_ensemble):
                x_end = th.randn_like(x_start).to(self.device)
                if self.diffuser_type == "diffusion":
                    if args.z_sampling:
                        pred_x_start = self.diffuser.z_sample_loop(
                             self.model,
                            x_end.shape,
                            noise=x_end,
                            model_kwargs=model_kwargs,
                            clip_denoised=True,
                            lambda_step=args.lambda_step,
                        )
                    else:
                        pred_x_start = self.diffuser.ddim_sample_loop(
                            self.model,
                            x_end.shape,
                            noise=x_end,
                            model_kwargs=model_kwargs,
                            clip_denoised=True,
                        )
                elif self.diffuser_type == "rectified_flow":
                    pred_x_start = self.diffuser.sampler(
                        self.model, 
                        x_end.shape, 
                        self.device, 
                        model_kwargs=model_kwargs
                    )
                elif self.diffuser_type == "flow_matching":
                    print("Not implemented yet.")
                    exit()
                if self.mode == "latent":
                    with th.no_grad():
                        if self.use_stability_vae:
                            if self.use_tanh:
                                pred_x_start = self.vae.decode(pred_x_start).sample
                            else:
                                pred_x_start = self.vae.decode(pred_x_start/self.scaling_factor).sample
                    if args.ch1_ch3:
                        pred_x_start = th.mean(pred_x_start, dim=1)
                        pred_x_start = pred_x_start.unsqueeze(1)
                            
                pred_x_start_list.append(pred_x_start)
            mean_x_start = th.mean(th.stack(pred_x_start_list), dim=0)  # emsembleの平均
            staple_x_start = staple(th.stack(pred_x_start_list,dim=0)).squeeze(0)
            std_x_start = th.std(th.stack(pred_x_start_list), dim=0)  # emsembleの標準偏差

            if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                if args.split_ODOC == "optic_cup" or args.split_ODOC == "optic_disc":
                    # 3チャネルを平均
                    mean_x_start = th.mean(mean_x_start, dim=1)
                    # 閾値処理
                    threshold                = threshold_otsu(mean_x_start.cpu().numpy())
                    mean_x_start_binary_otsu = (mean_x_start > threshold).float()
                else:
                    mean_x0_start = mean_x_start[:,0]
                    mean_x1_start = mean_x_start[:,1]
                    mean_x2_start = mean_x_start[:,2]
                    th0 = threshold_otsu(mean_x0_start.cpu().numpy())
                    th1 = threshold_otsu(mean_x1_start.cpu().numpy())
                    th2 = threshold_otsu(mean_x2_start.cpu().numpy())
                    mean_x0_start_binary_otsu = (mean_x0_start > th0).float()
                    mean_x1_start_binary_otsu = (mean_x1_start > th1).float()
                    mean_x2_start_binary_otsu = (mean_x2_start > th2).float()
                    mean_x_start_binary_otsu = th.stack([mean_x0_start_binary_otsu, mean_x1_start_binary_otsu, mean_x2_start_binary_otsu], dim=1)
            else:
                threshold = threshold_otsu(mean_x_start.cpu().numpy())
                mean_x_start_binary_otsu = (mean_x_start > threshold).float()
            
                threshold = threshold_otsu(staple_x_start.cpu().numpy())
                staple_x_start_binary_otsu = (staple_x_start > threshold).float()

            if self.rank == 0:
                if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                    if args.split_ODOC == "optic_cup" or args.split_ODOC == "optic_disc":
                        mean_x_start_binary_otsu = mean_x_start_binary_otsu.unsqueeze(1).cpu().numpy()
                        mask = mask[:,0].unsqueeze(1).cpu().numpy()
                        th_otsu_metric = calc_metric(mean_x_start_binary_otsu, mask)
                        test_dice_list.append(th_otsu_metric["dice"])
                        test_jaccad_list.append(th_otsu_metric["iou"])
                    else:
                        mean_x1_start_binary_otsu = mean_x1_start_binary_otsu.cpu().numpy()
                        mean_x2_start_binary_otsu = mean_x2_start_binary_otsu.cpu().numpy()
                        mask = mask.cpu().numpy()
                        mean_x0_start_binary_otsu = mean_x0_start_binary_otsu.cpu().numpy()
                        
                        th1_otsu_metric = calc_metric(mean_x1_start_binary_otsu, mask[:,1])
                        th2_otsu_metric = calc_metric(mean_x2_start_binary_otsu, mask[:,2])
                        mask = fundus_inv_map_mask(mask)
                        mean_x_start_binary_otsu = fundus_inv_map_mask(mean_x_start_binary_otsu)
                        
                        test_disc_dice_list.append(th1_otsu_metric["dice"])
                        test_disc_jaccad_list.append(th1_otsu_metric["iou"])
                        test_cup_dice_list.append(th2_otsu_metric["dice"])
                        test_cup_jaccad_list.append(th2_otsu_metric["iou"])
                else:
                    mean_x_start_binary_otsu   = mean_x_start_binary_otsu.cpu().numpy()
                    staple_x_start_binary_otsu = staple_x_start_binary_otsu.cpu().numpy()
                    if args.ch1_ch3:
                        mask = mask[:,0,:,:].unsqueeze(1).cpu().numpy()
                    else:
                        mask = mask.cpu().numpy()
            
                    th_otsu_metric = calc_metric(mean_x_start_binary_otsu, mask)
                    test_dice_list.append(th_otsu_metric["dice"])
                    test_jaccad_list.append(th_otsu_metric["iou"])
            
                    staple_otsu_metric = calc_metric(staple_x_start_binary_otsu, mask)
                    staple_dice_list.append(staple_otsu_metric["dice"])
                    staple_jaccad_list.append(staple_otsu_metric["iou"])
            
            if self.wandb_flag and self.rank == 0:
                wandb_num_images = min(self.wandb_num_images, y.shape[0], x_start.shape[0], mean_x_start_binary_otsu.shape[0], std_x_start.shape[0])
                wandb_y = [wandb.Image(y[i].cpu()) for i in range(wandb_num_images)]
                wandb_x = [wandb.Image(mask[i]) for i in range(wandb_num_images)]
                wandb_pred_x_otsu = [wandb.Image(mean_x_start_binary_otsu[i]) for i in range(wandb_num_images)]
                # wandb_staple_x_otsu = [wandb.Image(staple_x_start_binary_otsu[i]) for i in range(wandb_num_images)]
                wandb_std_x = [wandb.Image(std_x_start[i].cpu()) for i in range(wandb_num_images)]

                if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                    if args.split_ODOC == "optic_cup" or args.split_ODOC == "optic_disc":
                        wandb.log({
                            "image": wandb_y,
                            "mask": wandb_x,
                            "pred_otsu_mask": wandb_pred_x_otsu,
                            # "staple_otsu_mask": wandb_staple_x_otsu,
                            "std_mask": wandb_std_x,
                            "disc_dice": th_otsu_metric["dice"],
                            "disc_iou": th_otsu_metric["iou"],
                        })
                    else:
                        wandb.log({
                            "image": wandb_y,
                            "mask": wandb_x,
                            "pred_otsu_mask": wandb_pred_x_otsu,
                            # "staple_otsu_mask": wandb_staple_x_otsu,
                            "std_mask": wandb_std_x,
                            "disc_dice": th1_otsu_metric["dice"],
                            "disc_iou": th1_otsu_metric["iou"],
                            "cup_dice": th2_otsu_metric["dice"],
                            "cup_iou": th2_otsu_metric["iou"],
                        })
                else:            
                    wandb.log({
                        "image": wandb_y,
                        "mask": wandb_x,
                        "pred_otsu_mask": wandb_pred_x_otsu,
                        # "staple_otsu_mask": wandb_staple_x_otsu,
                        "std_mask": wandb_std_x,
                        # "staple_dice": staple_otsu_metric["dice"],
                        # "staple_iou": staple_otsu_metric["iou"],
                        "dice": th_otsu_metric["dice"],
                        "iou": th_otsu_metric["iou"],
                        # "sensitivity": th_otsu_metric["sensitivity"],
                        # "specificity": th_otsu_metric["specificity"],
                        # "accuracy": th_otsu_metric["accuracy"],
                    })
        
        if self.wandb_flag and self.rank == 0:
            if args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop" or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
                if args.split_ODOC == "optic_cup" or args.split_ODOC == "optic_disc":
                    test_dice        = sum(test_dice_list) / len(test_dice_list)
                    test_jaccad      = sum(test_jaccad_list) / len(test_jaccad_list)
                    print(f"test_dice:{test_dice}")
                    print(f"test_jaccad:{test_jaccad}")
                    wandb.log({
                        "test_dice": test_dice,
                        "test_iou": test_jaccad,
                    })
                else:
                    test_disc_dice   = sum(test_disc_dice_list) / len(test_disc_dice_list)
                    test_cup_dice    = sum(test_cup_dice_list) / len(test_cup_dice_list)
                    test_disc_jaccad = sum(test_disc_jaccad_list) / len(test_disc_jaccad_list)
                    test_cup_jaccad  = sum(test_cup_jaccad_list) / len(test_cup_jaccad_list)
                    print(f"test_disc_dice:{test_disc_dice}")
                    print(f"test_cup_dice:{test_cup_dice}")
                    wandb.log({
                        "test_disc_dice": test_disc_dice,
                        "test_cup_dice": test_cup_dice,
                        "test_disc_iou": test_disc_jaccad,
                        "test_cup_iou": test_cup_jaccad,
                    })
            else:
                test_dice        = sum(test_dice_list) / len(test_dice_list)
                test_sta_dice    = sum(staple_dice_list) / len(staple_dice_list)
                test_sta_jaccad  = sum(staple_jaccad_list) / len(staple_jaccad_list)
                test_jaccad      = sum(test_jaccad_list) / len(test_jaccad_list)
                # test_sensitivity = sum(test_sensitivity_list) / len(test_sensitivity_list)
                # test_specificity = sum(test_specificity_list) / len(test_specificity_list)
                # test_accuracy = sum(test_accuracy_list) / len(test_accuracy_list)
            
                print(f"test_dice:{test_dice}")
                print(f"test_jaccad:{test_jaccad}")
                # print(f"test_sensitivity:{test_sensitivity}")
                # print(f"test_specificity:{test_specificity}")
                # print(f"test_accuracy:{test_accuracy}")
                wandb.log({
                    "test_staple_dice": test_sta_dice,
                    "test_staple_jaccad": test_sta_jaccad,
                    "test_dice": test_dice,
                    "test_jaccad": test_jaccad,
                    # "test_sensitivity": test_sensitivity,
                    # "test_specificity": test_specificity,
                    # "test_accuracy": test_accuracy,
                })
            wandb.finish()
            
        if self.multi_gpu:
            dist.barrier()
            cleanup()

def gather_tensors(tensor):
    """
    全GPUのテンソルを集約します。
    """
    tensors_gather = [th.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor)
    return th.cat(tensors_gather, dim=0)

def calc_metric(pred_x, x):
    dice_list = []
    iou_list = []
    sensitivity_list = []
    specificity_list = []
    accuracy_list = []
    batch = x.shape[0]
    for i in range(batch):
        target = x[i]
        pred = pred_x[i]
        dice_list.append(dice(pred, target))
        iou_list.append(jaccard(pred, target))
        sensitivity_list.append(sensitivity(pred, target))
        specificity_list.append(specificity(pred, target))
        accuracy_list.append(accuracy(pred, target))
        
    metric = {
        "dice": sum(dice_list),
        "iou": sum(iou_list),
        "specificity": sum(specificity_list),
        "sensitivity": sum(sensitivity_list),
        "accuracy": sum(accuracy_list),
    }

    return metric