Just a short recording of exps plans so I don't lose them


zero shot nllb 600m
zero shot nllb 1.3b

finetune nllb 600m 1e-4, 2e-4, 3e-4
finetune nllb 1.3b 1e-4, 2e-4, 3e-4

finetune mt5 base 1e-3
finetune mt5 large 1e-3

finetune mix tibetan/translation examples proportional mt5 base 1e-3
finetune mix tibetan/translation examples proportional mt5 large 1e-3

finetune mix tibetan/translation 1:9 mt5 base 1e-3
finetune mix tibetan/translation 1:9 mt5 large 1e-3

finetune mix tibetan/translation 1:3 mt5 base 1e-3
finetune mix tibetan/translation 1:3 mt5 large 1e-3

finetune mix tibetan/translation 1:1 mt5 base 1e-3
finetune mix tibetan/translation 1:1 mt5 large 1e-3

pretrain mix tibetan/chinese/english/translation examples proportional 1e-4 finetune 1e-3 mt5 base
pretrain mix tibetan/chinese/english/translation examples proportional 1e-4 finetune 1e-3 mt5 large
