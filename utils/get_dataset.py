import pandas as pd
import os
import torch.nn.functional as F
import torchvision
import os
import torch
from torchvision.transforms import InterpolationMode

def get_transform(args, version="v1"):
    # シードを固定
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    # GPUの乱数も固定（必要なら）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # マルチGPU用
        torch.backends.cudnn.deterministic = True  # CuDNNの動作を固定
        torch.backends.cudnn.benchmark = False  # 再現性を優先
      
    if version=="v1":
        from torchvision import transforms
        print("use transform v1")
        if args.dataset == 'REFUGE2' or args.dataset == 'REFUGE2_Crop' or args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
            transform = {
                "image": transforms.Compose([
                    transforms.Resize(size=(args.img_size,args.img_size),interpolation=InterpolationMode.BILINEAR),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    # transforms.RandomRotation(degrees=(0, 360)),
                    transforms.RandomAffine(
                        degrees=45,                   # 回転範囲 (-45, 45)
                        translate=(0.0625, 0.0625),   # 平行移動範囲 (-6.25%, +6.25%)
                        scale=(0.9, 1.1),             # 拡大縮小範囲 (90% - 110%)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]),
                
                "mask": transforms.Compose([
                    transforms.Resize(size=(args.img_size,args.img_size),interpolation=InterpolationMode.NEAREST),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    # transforms.RandomRotation(degrees=(0, 360)),
                    transforms.RandomAffine(
                        degrees=45,                   # 回転範囲 (-45, 45)
                        translate=(0.0625, 0.0625),   # 平行移動範囲 (-6.25%, +6.25%)
                        scale=(0.9, 1.1),             # 拡大縮小範囲 (90% - 110%)
                    ),
                ])
            }
            
            val_transform = {
                "image": transforms.Compose([
                    transforms.Resize(size=(args.img_size,args.img_size),interpolation=InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ]),
                "mask": transforms.Compose([
                    transforms.Resize(size=(args.img_size,args.img_size),interpolation=InterpolationMode.NEAREST),
                    ])
            }        
        else:
            transform = {
                "image": transforms.Compose([
                    transforms.Resize(size=(args.img_size,args.img_size),interpolation=InterpolationMode.BILINEAR),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    # transforms.RandomRotation(degrees=(0, 360)),
                    transforms.RandomAffine(
                        degrees=45,                   # 回転範囲 (-45, 45)
                        translate=(0.0625, 0.0625),   # 平行移動範囲 (-6.25%, +6.25%)
                        scale=(0.9, 1.1),             # 拡大縮小範囲 (90% - 110%)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]),
                
                "mask": transforms.Compose([
                    transforms.Resize(size=(args.img_size,args.img_size),interpolation=InterpolationMode.NEAREST),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    # transforms.RandomRotation(degrees=(0, 360)),
                    transforms.RandomAffine(
                        degrees=45,                   # 回転範囲 (-45, 45)
                        translate=(0.0625, 0.0625),   # 平行移動範囲 (-6.25%, +6.25%)
                        scale=(0.9, 1.1),             # 拡大縮小範囲 (90% - 110%)
                    ),
                    transforms.ToTensor(),
                ])
            }
            
            val_transform = {
                "image": transforms.Compose([
                    transforms.Resize(size=(args.img_size,args.img_size),interpolation=InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ]),
                "mask": transforms.Compose([
                    transforms.Resize(size=(args.img_size,args.img_size),interpolation=InterpolationMode.NEAREST),
                    transforms.ToTensor(),
                    ])
            }
    else:
        from torchvision.transforms import v2
        
        # OMP_NUM_THREADSの警告を抑制するための設定
        os.environ["OMP_NUM_THREADS"] = "1"
        torchvision.disable_beta_transforms_warning()
        print("use transform v2")
        transform = {
            "image": v2.Compose([
                v2.RandomResizedCrop(size=(args.img_size,args.img_size), antialias=True, interpolation=InterpolationMode.BILINEAR),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.3),
                v2.ToImageTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # v2.ToTensor(),
                ]),
            
            "mask": v2.Compose([
                v2.RandomResizedCrop(size=(args.img_size,args.img_size), antialias=True, interpolation=InterpolationMode.NEAREST),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToImageTensor(),
                v2.ToDtype(torch.float32, scale=True),
                ])
            }     
        val_transform = {
            "image": v2.Compose([
                v2.Resize(size=(args.img_size,args.img_size), antialias=True,interpolation=InterpolationMode.BILINEAR),
                v2.ToImageTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
            "mask": v2.Compose([
                v2.Resize(size=(args.img_size,args.img_size), antialias=True,interpolation=InterpolationMode.NEAREST),
                v2.ToImageTensor(),
                v2.ToDtype(torch.float32, scale=True),
                ])
        }
    return transform, val_transform

def get_dataset(args):
    # # シードを固定
    # torch.manual_seed(args.seed)
    # torch.manual_seed(args.seed)
    # # GPUの乱数も固定（必要なら）
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)  # マルチGPU用
    #     torch.backends.cudnn.deterministic = True  # CuDNNの動作を固定
    #     torch.backends.cudnn.benchmark = False  # 再現性を優先
        
    transform, val_transform = get_transform(args, version=args.transoform_version)
    
    if args.dataset == 'ISIC2017':
        from dataset.isic2017_dataset import ISICDataset
        train_df  = pd.read_csv(os.path.join(args.data_path, 'train_id' + '.csv'), encoding='gbk')
        val_df    = pd.read_csv(os.path.join(args.data_path, 'validation_id' + '.csv'), encoding='gbk')
        test_df   = pd.read_csv(os.path.join(args.data_path, 'test_id' + '.csv'), encoding='gbk')
        
        train_set = ISICDataset(args.data_path, df=train_df, transform=transform, mode='Train')
        val_set   = ISICDataset(args.data_path, df=val_df,  transform=val_transform, mode='Validation')
        test_set  = ISICDataset(args.data_path, df=test_df, transform=val_transform, mode='Test')
    
    elif args.dataset == 'ISIC2018':
        # if args.reflow_flag:
        #     print("use reflow dataset")
        #     from dataset.isic2018_dataset_reflow import ISICDataset
        #     train_set = ISICDataset(args.data_path, fold=args.fold, split='train',reflow_dir=args.reflow_dir, transform=transform)
        # else:
            # from dataset.isic2018_dataset import ISICDataset
            # train_set = ISICDataset(args.data_path, fold=args.fold, split='train', transform=transform)
        from dataset.isic2018_dataset import ISICDataset
        train_set = ISICDataset(args.data_path, fold=args.fold, split='train', transform=transform, ch1_ch3=args.ch1_ch3)
        val_set   = ISICDataset(args.data_path, fold=args.fold, split='val', transform=val_transform, ch1_ch3=args.ch1_ch3)
        test_set  = ISICDataset(args.data_path, fold=args.fold, split='val', transform=val_transform,  ch1_ch3=args.ch1_ch3)
        
    elif args.dataset == 'ISIC2016':
        from dataset.isic_dataset import ISICDataset
        train_df = pd.read_csv(os.path.join(args.data_path, 'csv_fold/fold_'+args.fold+'_train_id.csv'), encoding='gbk')
        val_df   = pd.read_csv(os.path.join(args.data_path, 'csv_fold/fold_'+args.fold+'_validation_id.csv'), encoding='gbk')
        test_df  = pd.read_csv(os.path.join(args.data_path, 'test_id.csv'), encoding='gbk')

        train_set = ISICDataset(args.data_path, df=train_df, transform=transform, mode='Training')
        val_set   = ISICDataset(args.data_path, df=val_df,  transform=val_transform, mode='Validation')
        test_set  = ISICDataset(args.data_path, df=test_df, transform=val_transform, mode='Test')

    elif args.dataset == 'REFUGE-Cup':
        from dataset.refuge_dataset import REFUGEDataset
        train_set = REFUGEDataset(args.data_path, mode='Training', transform=transform, type='cup')
        val_set   = REFUGEDataset(args.data_path, mode='Validation', transform=val_transform, type='cup')
        test_set  = REFUGEDataset(args.data_path, mode='Test', transform=val_transform, type='cup')
    
    elif args.dataset == 'REFUGE-Disc':
        from dataset.refuge_dataset import REFUGEDataset
        train_set = REFUGEDataset(args.data_path, mode='Training', transform=transform, type='disc')
        val_set   = REFUGEDataset(args.data_path, mode='Validation', transform=val_transform, type='disc')
        test_set  = REFUGEDataset(args.data_path, mode='Test', transform=val_transform, type='disc')
    elif args.dataset == "REFUGE2" or args.dataset == "REFUGE2_Crop":
        from dataset.refuge2_dataset import REFUGEDataset
        train_set = REFUGEDataset(args.data_path, mode='Train', transform=transform, crop=args.crop, center=args.center, split_ODOC=args.split_ODOC)
        val_set   = REFUGEDataset(args.data_path, mode='Validation', transform=val_transform, crop=args.crop, center=args.center, split_ODOC=args.split_ODOC)
        test_set  = REFUGEDataset(args.data_path, mode='Test', transform=val_transform, crop=args.crop, center=args.center, split_ODOC=args.split_ODOC)
    elif args.dataset == "REFUGE1" or args.dataset == "REFUGE1_Crop":
        from dataset.refuge_dataset import REFUGEDataset
        train_set = REFUGEDataset(args.data_path, mode='Train', transform=transform, crop=args.crop, center=args.center, split_ODOC=args.split_ODOC)
        val_set   = REFUGEDataset(args.data_path, mode='Validation', transform=val_transform, crop=args.crop, center=args.center, split_ODOC=args.split_ODOC)
        test_set  = REFUGEDataset(args.data_path, mode='Test', transform=val_transform, crop=args.crop, center=args.center, split_ODOC=args.split_ODOC)
    return train_set, val_set, test_set