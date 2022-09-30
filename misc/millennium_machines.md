As of September 2022

```bash
# Command to get GPU name
nvidia-smi --query-gpu=name --format=csv,noheader
```

All hostnames end with `.millennium.berkeley.edu` e.g. `a0.millennium.berkeley.edu`.

All hostnames > a28, b13, and c139 do not exist.

a1-15, a27, b5, c27-28, c30: Connection closed

a16, a27, b1-4, b6-12, c1-4, c7, c11-24, c31-138: Connection timed out

c5-6, c8-10, c25-26, c29: Permission denied


|ssh prefix|GPU name           |GPU RAM (GB)|Count|
|----------|-------------------|------------|-----|
|a18       |GeForce GTX TITAN X|          12|    8|
|a19       |Tesla M40          |          12|    8|
|a20       |Tesla M40          |          12|    8|
|a21       |Tesla K80          |          12|   16|
|a22       |Tesla K80          |          12|   16|
|a23       |Titan V            |          12|    8|
|a24       |Titan RTX          |          24|    8|
|a25       |Titan RTX          |          24|    8|
|a26       |Titan RTX          |          24|    8|


|GPU name           |Arch   |
|-------------------|-------|
|GeForce GTX TITAN X|Maxwell|
|Tesla M40          |Maxwell|
|Tesla K80          |      ?|
|Titan V            |  Volta|
|Titan RTX          | Turing|


|Arch   |bf16?|
|-------|-----|
|Maxwell|   no|
|  Volta|   no|
| Turing|   no|
