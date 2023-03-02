# Setup
## Deepspeed env
```bash
cp scripts/multi_node/.deepspeed_env ~/.deepspeed_env
```

## Deepspeed library
```bash
# Deepspeed code path example
~/miniconda3/envs/LME/lib/python3.9/site-packages/deepspeed
```
Make sure the values for your IP addresses are correct.
- comm/comm.py: 678
- launcher/runner.py: 449

Make sure the values for your network interfaces are correct.
- launcher/multinode_runner.py: 154

# Running tests
## Deepspeed setup
```bash
deepspeed --hostfile=scripts/multi_node/hostfile.txt --launcher=openmpi scripts/multi_node/deepspeed_test.py
```

## Connection reset by peer error
```bash
mpirun --hostfile scripts/multi_node/hostfile.txt ~/miniconda3/envs/LME/bin/python nccl_test.py
```
