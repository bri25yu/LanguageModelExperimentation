import os

from tqdm import trange

from torch import int64, zeros
from torch.distributed import all_gather, barrier, init_process_group


os.environ.update({
    "NCCL_SOCKET_IFNAME": "ens2f1",
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_DISABLE": "1",
    "MASTER_ADDR": "10.11.49.60",
    "MASTER_PORT": "29500",
})

WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
WORLD_SIZE = 4

init_process_group(
    backend="nccl",
    world_size=WORLD_SIZE,
    rank=WORLD_RANK,
)

NUM_ITERS = 10 ** 2
for i in trange(NUM_ITERS):
    local_tensor = zeros((8, 1024, 1024), dtype=int64, device=LOCAL_RANK)
    tensor_list = [local_tensor.clone() for _ in range(WORLD_SIZE)]

    all_gather(tensor_list, local_tensor)

barrier()
