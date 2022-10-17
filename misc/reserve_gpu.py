"""
Run this script as `python -i reserve_gpu.py --gpu_ids 0 1`
"""
from argparse import ArgumentParser

from math import prod

import torch


def reserve_gpu(gpu_id: int) -> None:
    free_memory_in_bytes, _ = torch.cuda.mem_get_info(device=gpu_id)  # On the order of ~ 10000MiB
    base_layer_size = [10000, 10000]
    base_layer_size_in_bytes = 4 * prod(base_layer_size)  # Should be ~ 400MiB
    num_bytes_to_reserve = 2 * 10**9

    max_number_of_layers = (free_memory_in_bytes - num_bytes_to_reserve) // base_layer_size_in_bytes
    total_size = [max_number_of_layers] + base_layer_size

    t = torch.ones(total_size)
    t = t.cuda(gpu_id)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu_ids", nargs="+", type=int)
    args = parser.parse_args()

    for gpu_id in args.gpu_ids:
        reserve_gpu(gpu_id)

    while input("Press q + Enter to quit") != "q":
        pass

    torch.cuda.empty_cache()
