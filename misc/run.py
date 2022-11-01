"""
Run this script as `python reserve_gpu.py --gpu_ids 0 1`
"""
from math import prod

from tqdm import trange

import torch


def reserve_gpu(gpu_id: int) -> bool:
    free_memory_in_bytes, total_memory_in_bytes = torch.cuda.mem_get_info(device=gpu_id)  # On the order of ~ 10000MiB
    if free_memory_in_bytes / total_memory_in_bytes < 0.4:
        return False

    base_layer_size = [10000, 10000]
    base_layer_size_in_bytes = 4 * prod(base_layer_size)  # Should be ~ 400MiB
    num_bytes_to_reserve = 2 * 10**9

    max_number_of_layers = (free_memory_in_bytes - num_bytes_to_reserve) // base_layer_size_in_bytes

    # Randomly add or take away a few layers
    n_layers = max_number_of_layers + torch.randint(-3, 1, (1,))

    total_size = [n_layers] + base_layer_size

    t = torch.ones(total_size)
    t = t.cuda(gpu_id)

    return True


if __name__ == "__main__":
    gpus_reserved = []
    for gpu_id in trange(torch.cuda.device_count(), desc="Reserving GPUs", leave=False):
        if reserve_gpu(gpu_id):
            gpus_reserved.append(gpu_id)

    print(f"Reserved {len(gpus_reserved)} gpus: {gpus_reserved}")

    while input("Press q + Enter to quit\n") != "q":
        pass

    torch.cuda.empty_cache()
