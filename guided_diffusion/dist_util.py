import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()
    
def init_process(args):
    """
    Initialize DDP training.
    """
    assert torch.cuda.is_available(), "CUDA is not available. Exiting."
    dist.init_process_group('nccl')
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size be divisible by world size."

    rank   = dist.get_rank()
    device = rank%torch.cuda.device_count()
    seed   = args.global_seed+rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed = {seed}, world_size={dist.get_world_size()}.")
    return rank, device, seed
    