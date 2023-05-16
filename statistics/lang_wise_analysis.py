import pandas as pd
from typing import List

import json

from sacrebleu import CHRF
from sacrebleu.utils import sum_of_lists

chrf = CHRF(6, 2, 2)  # character n-gram order 6, word n-gram order 2, beta 2

def chrf_unreduced_str_to_aggregate(strs: List[str]) -> float:
    return chrf._compute_f_score(sum_of_lists([json.loads(s) for s in strs]))


# Expecting dataframe input as input.pkl
# which has columns ['source_lang', 'target_lang', 'chrf_unreduced']
# output will be a dataframe with columns ['label', 'chrF++']
# and 202 * 1012 * 202 = 41,293,648 rows. 

# This script will gather analysis for the language wise pairs.

# ----------------------------
# Lang to lang analysis:
# ----------------------------
# Read in data
df = pd.read_pickle('input.pkl')

# Group by source lang 
df_source = df.groupby(['source_lang'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index("chrF++")
df_source['source_lang'] = df_source['source_lang'].apply(func=lambda x: f"{x}-xx")
df_source.rename(columns={'source_lang': 'label'}, inplace=True)

# Group by target lang
df_target = df.groupby(['target_lang'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index("chrF++")
df_target['target_lang'] = df_target['target_lang'].apply(func=lambda x: f"xx-{x}")
df_target.rename(columns={'target_lang': 'label'}, inplace=True)

# Rename raw language to language data and include
df_pair_labeled = df.copy()
df_pair_labeled['label'] = df_pair_labeled['source_lang'] + '-' + df_pair_labeled['target_lang']
df_pair_labeled['chrF++'] = df_pair_labeled['chrf_unreduced'].apply(func=lambda x: chrf_unreduced_str_to_aggregate([x]))
# save xx-yy for later
xx_yy_chrf = df.where(df['source_lang'] != 'eng_Latn').where(df['target_lang'] != 'eng_Latn')['chrF++'].mean()
df_pair_labeled.drop(columns=['source_lang', 'target_lang', 'source_pretrain', 'target_pretrain'], inplace=True)

# concat all results together by label
df_all_lang = pd.concat([df_source, df_target, df_pair_labeled], ignore_index=True)

# save to csv
df_all_lang.to_csv('lang_to_lang_analysis.csv', index=False)


# ----------------------------
# Avg stats:
# ----------------------------
# Group by source lang and target lang pretrain level
# Input needs to turn into df with columns ['source_lang', 'source_pretrain', 'target_lang', 'target_pretrain', 'chrf_unreduced']
mc4_df = pd.read_csv('mc4_sizes_fixed.csv')
mc4_pretrain_langs = set(mc4_df['lang_code'])
# Add pretrain columns
df_pretrain_group = df.copy()
df_pretrain_group['source_pretrain'] = df_pretrain_group['source_lang'].apply(func=lambda x: x in mc4_pretrain_langs)
df_pretrain_group['target_pretrain'] = df_pretrain_group['target_lang'].apply(func=lambda x: x in mc4_pretrain_langs)
# Group by pretrain
df_pretrain_group = df.groupby(['source_pretrain', 'target_pretrain'], as_index=False)['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index("chrF++")
df_pretrain_group['label'] = df_pretrain_group['source_pretrain'] + '-' + df_pretrain_group['target_pretrain']
df_pretrain_group.drop(columns=['source_pretrain', 'target_pretrain'], inplace=True)

# create gloabl average row (including english)
avg_chrf = df_pair_labeled['chrF++'].mean()

# add xx-yy row and global average row
df_avg_stats = df_pretrain_group.copy()
df_avg_stats = df_avg_stats.append({'label': 'xx-yy', 'chrF++': xx_yy_chrf}, ignore_index=True)
df_avg_stats = df_avg_stats.append({'label': 'avg', 'chrF++': avg_chrf}, ignore_index=True)

# save to csv
df_avg_stats.to_csv('avg_stats.csv', index=False)


# ----------------------------
# Pretraining mt5 analysis:
# ----------------------------
# merge mt5 pretraining info with chrf scores
# mc4 pretraining info will come of the form of:
# ['lang_code', 'size']
# merge on lang code source
mc4_df = pd.read_csv('mc4_sizes_fixed.csv')
mc4_df['lang_code'] = mc4_df['lang_code'].apply(func=lambda x: f"{x}-xx")
mc4_df = mc4_df.rename(columns={'lang_code': 'label'})
mt5_df_source_pretrain = df_source.merge(mc4_df, on='label', how='right') # merge on source lang, if it has no pretraining then remove it

# and target
mc4_df = pd.read_csv('mc4_sizes_fixed.csv')
mc4_df['lang_code'] = mc4_df['lang_code'].apply(func=lambda x: f"xx-{x}")
mc4_df = mc4_df.rename(columns={'lang_code': 'label'})
mt5_df_target_pretrain = df_target.merge(mc4_df, on='label', how='right') # merge on target lang, if it has no pretraining then remove it

# combine source and target
mt5_df_pretrain = mt5_df_source_pretrain.append(mt5_df_target_pretrain, ignore_index=True)

# save to csv
mt5_df_pretrain.to_csv('mt5_pretrain.csv', index=False)


# TODO nllb analysis later
# # ----------------------------
# # Pretraining nllb analysis:
# # ----------------------------
# # merge nllb training info with chrf scores
# # nllb pretraining info will come of the form of:
# # ['lang_pair', 'size']
# # merge on lang pair
# nllb_df = pd.read_csv('nllb_sizes.csv')
# nllb_df = nllb_df.rename(columns={'lang_pair': 'label'})
# nllb_df_pair_labeled = df_pair_labeled.merge(nllb_df, on='label', how='right') # merge on source lang, if it has no pretraining then remove it

# # also on lang code source
# nllb_df_source = nllb_df.copy()
# nllb_df_source['label'] = nllb_df_source['label'].apply(func=lambda x: x.split('-')[0])
# # group by label and sum
# nllb_df_source = nllb_df_source.groupby(['label'], as_index=False).sum()
# # merge with df_source
# nllb_df_source_pretrain = df_source.merge(nllb_df_source, on='label', how='right')

# # and target
# nllb_df_target = nllb_df.copy()
# nllb_df_target['label'] = nllb_df_target['label'].apply(func=lambda x: x.split('-')[1])
# # group by label and sum
# nllb_df_target = nllb_df_target.groupby(['label'], as_index=False).sum()
# # merge with df_target
# nllb_df_target_pretrain = df_target.merge(nllb_df_target, on='label', how='right')
