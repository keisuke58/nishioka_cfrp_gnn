"""トレーニング関連のユーティリティ関数（GNN_zscore_sub_noise_defect_free.pyを基に）"""
import os
import random
import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed=42):
    """再現性のためのシード設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup(rank, world_size):
    """プロセスグループの初期化"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )


def cleanup(device=None, rank=0, do_barrier=True):
    """Safely cleanup distributed process group.
    
    IMPORTANT:
    - Only call a final `dist.barrier()` if *all ranks* will also call it.
      If rank-0 continues into single-process post-work while other ranks exit,
      those exiting ranks must set `do_barrier=False` to avoid deadlock/timeouts.
    
    Args:
        device: CUDA device (optional, auto-detected if None)
        rank: Process rank (default: 0)
        do_barrier: Whether to call barrier before cleanup (default: True)
    """
    try:
        # Check if process group is initialized
        if not dist.is_initialized():
            return
        
        # Get device if not provided (use default)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # First, ensure all CUDA operations are complete
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                if rank == 0:
                    print(f"[WARNING] CUDA sync failed during cleanup: {e}")
        
        # Optional: synchronize all ranks before destroying.
        # WARNING: calling barrier here is only safe if all ranks participate.
        if do_barrier:
            try:
                if dist.is_initialized():
                    try:
                        # Barrier is more reliable than all_reduce for final sync
                        dist.barrier()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except (RuntimeError, Exception) as barrier_error:
                        # If barrier fails (e.g., communicator aborted), that's OK
                        if rank == 0 and "aborted" not in str(barrier_error).lower():
                            print(f"[WARNING] Final barrier before cleanup failed (continuing anyway): {barrier_error}")
            except Exception as sync_error:
                if rank == 0:
                    print(f"[WARNING] Final barrier before cleanup failed (continuing anyway): {sync_error}")
        
        # Now destroy the process group
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as destroy_error:
            # If destroy fails, that's OK - process will exit anyway
            if rank == 0:
                print(f"[WARNING] Process group destroy failed (this is OK): {destroy_error}")
        
    except Exception as e:
        # Log error but don't raise - allow process to exit gracefully
        if rank == 0:
            print(f"[WARNING] Error during cleanup: {e}")
        # Force destroy if normal destroy fails
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass  # Ignore errors during forced cleanup


def get_distributed_info():
    """分散学習の環境情報を取得"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')
    return rank, world_size, master_addr, master_port