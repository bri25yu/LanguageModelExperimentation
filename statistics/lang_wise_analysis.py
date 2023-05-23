import pandas as pd
from typing import List
import os

import json

from sacrebleu import CHRF
from sacrebleu.utils import sum_of_lists

from datasets import load_dataset

chrf = CHRF(6, 2, 2)  # character n-gram order 6, word n-gram order 2, beta 2

def chrf_unreduced_str_to_aggregate(strs: List[str]) -> float:
    return chrf._compute_f_score(sum_of_lists([json.loads(s) for s in strs]))


# Expecting input as dataset from HF.
# which has columns ['id', 'source_lang', 'target_lang', 'chrf_unreduced', 'source', 'target', 'prediction']
# output will be a dataframe with columns ['label', 'chrF++']
# and 202 * 1012 * 202 = 41,293,648 rows. 

# This script will gather analysis for the language wise pairs.

HF_PATH = 'hlillemark/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MC4_PATH = os.path.join(BASE_DIR, 'mc4_sizes_fixed.csv')


# ----------------------------
# Lang to lang analysis:
# ----------------------------
def lang_to_lang(OUTPUT_PATH, df):
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'lang_to_lang')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # Group by source lang 
    df_source = df[['source_lang', 'chrf_unreduced']].groupby(['source_lang'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index()
    df_source['source_lang'] = df_source['source_lang'].apply(func=lambda x: f"{x}-xx")
    df_source.rename(columns={'source_lang': 'label', 'chrf_unreduced': 'chrF++'}, inplace=True)

    # save to csv
    df_source.to_csv(os.path.join(OUTPUT_PATH,'lang_to_xx.csv'), index=False)

    # Group by target lang
    df_target = df[['target_lang', 'chrf_unreduced']].groupby(['target_lang'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index()
    df_target['target_lang'] = df_target['target_lang'].apply(func=lambda x: f"xx-{x}")
    df_target.rename(columns={'target_lang': 'label', 'chrf_unreduced': 'chrF++'}, inplace=True)

    # save to csv
    df_target.to_csv(os.path.join(OUTPUT_PATH, 'xx_to_lang.csv'), index=False)

    # Rename raw language to language data and include
    df_pair_labeled = df.copy()
    # group by pair and compute chrf
    df_pair_labeled = df_pair_labeled.groupby(['source_lang', 'target_lang'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index()
    df_pair_labeled.rename(columns={'chrf_unreduced': 'chrF++'}, inplace=True)

    df_pair_labeled['label'] = df_pair_labeled['source_lang'] + '-' + df_pair_labeled['target_lang']
    df_pair_labeled = df_pair_labeled[['label', 'chrF++']]

    # save to csv
    df_pair_labeled.to_csv(os.path.join(OUTPUT_PATH, 'lang_to_lang.csv'), index=False)


# ----------------------------
# Avg stats:
# ----------------------------
def avg_stats(OUTPUT_PATH, df):
    
    # Group by source lang and target lang pretrain level
    # Input needs to turn into df with columns ['source_lang', 'source_pretrain', 'target_lang', 'target_pretrain', 'chrf_unreduced']

    mc4_df = pd.read_csv(MC4_PATH)
    mc4_pretrain_langs = set(mc4_df['lang_code'])
    # Add pretrain columns
    df_pretrain_group = df.copy()
    df_pretrain_group['source_pretrain'] = df_pretrain_group['source_lang'].apply(func=lambda x: 'in' if x in mc4_pretrain_langs else 'out')
    df_pretrain_group['target_pretrain'] = df_pretrain_group['target_lang'].apply(func=lambda x: 'in' if x in mc4_pretrain_langs else 'out')
    # Group by pretrain
    df_pretrain_group = df_pretrain_group.groupby(['source_pretrain', 'target_pretrain'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index()
    df_pretrain_group['label'] = df_pretrain_group['source_pretrain'] + '-' + df_pretrain_group['target_pretrain']
    df_pretrain_group.rename(columns={'chrf_unreduced': 'chrF++'}, inplace=True)
    df_pretrain_group = df_pretrain_group[['label', 'chrF++']]

    # Rename raw language to language data and include
    df_pair_labeled = df.copy()
    # group by pair and compute chrf
    df_pair_labeled = df_pair_labeled.groupby(['source_lang', 'target_lang'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index()
    df_pair_labeled.rename(columns={'chrf_unreduced': 'chrF++'}, inplace=True)
    # make xx-yy
    xx_yy_chrf = df_pair_labeled.where(df_pair_labeled['source_lang'] != 'eng_Latn').where(df_pair_labeled['target_lang'] != 'eng_Latn')['chrF++'].mean()

    # create gloabl average row (including english)
    avg_chrf = df_pair_labeled['chrF++'].mean()

    # add xx-yy row and global average row
    df_avg_stats = df_pretrain_group.copy()
    df_avg_stats = pd.concat([df_avg_stats, pd.DataFrame({'label': ['xx-yy', 'avg'], 'chrF++': [xx_yy_chrf, avg_chrf]})])

    # save to csv
    df_avg_stats.to_csv(os.path.join(OUTPUT_PATH, 'avg_stats.csv'), index=False)


# ----------------------------
# Pretraining mt5 analysis:
# ----------------------------
def pretraining_mt5_analysis(OUTPUT_PATH):
    BASE_OUTPUT_PATH = OUTPUT_PATH
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'pretraining')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # merge mt5 pretraining info with chrf scores
    # mc4 pretraining info will come of the form of:
    # ['lang_code', 'size']
    # merge on lang code source
    mc4_df = pd.read_csv(MC4_PATH)
    mc4_df['lang_code'] = mc4_df['lang_code'].apply(func=lambda x: f"{x}-xx")
    mc4_df = mc4_df.rename(columns={'lang_code': 'label'})
    df_source = (pd.read_csv(os.path.join(BASE_OUTPUT_PATH, 'lang_to_lang', 'lang_to_xx.csv')))
    mt5_df_source_pretrain = df_source.merge(mc4_df, on='label', how='right') # merge on source lang, if it has no pretraining then remove it

    # save to csv
    mt5_df_source_pretrain.to_csv(os.path.join(OUTPUT_PATH, 'lang_to_xx_mt5_pretrain.csv'), index=False)

    # and target
    mc4_df = pd.read_csv(MC4_PATH)
    mc4_df['lang_code'] = mc4_df['lang_code'].apply(func=lambda x: f"xx-{x}")
    mc4_df = mc4_df.rename(columns={'lang_code': 'label'})
    df_target = (pd.read_csv(os.path.join(BASE_OUTPUT_PATH, 'lang_to_lang', 'xx_to_lang.csv')))
    mt5_df_target_pretrain = df_target.merge(mc4_df, on='label', how='right') # merge on target lang, if it has no pretraining then remove it

    # save to csv
    mt5_df_target_pretrain.to_csv(os.path.join(OUTPUT_PATH, 'xx_to_lang_mt5_pretrain.csv'), index=False)


def only_take_latn_cyrl_arab(x):
    if x in ['Latn', 'Cyrl', 'Arab']:
        return x
    else:
        return 'other'
# ----------------------------
# Script-wise analysis (Latn, Cyrl, Arab)
# ----------------------------
def script_analysis(OUTPUT_PATH, df):
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'script_analysis')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    df_script = df.copy()
    df_script['target_script'] = df_script['target_lang'].apply(func=lambda x: x.split('_')[1]).apply(func=only_take_latn_cyrl_arab)
    df_script['source_script'] = df_script['source_lang'].apply(func=lambda x: x.split('_')[1]).apply(func=only_take_latn_cyrl_arab)
    # Group by source script
    df_script_source = df_script[['source_script', 'chrf_unreduced']].groupby(['source_script'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index()
    df_script_source.rename(columns={'source_script': 'label', 'chrf_unreduced': 'chrF++'}, inplace=True)
    df_script_source = df_script_source[['label', 'chrF++']]
    df_script_source['label'] = df_script_source['label'].apply(func=lambda x: f"{x}-xx")
    df_script_source.to_csv(os.path.join(OUTPUT_PATH, 'script_to_xx.csv'), index=False)

    # Group by target script
    df_script_target = df_script[['target_script', 'chrf_unreduced']].groupby(['target_script'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index()
    df_script_target.rename(columns={'target_script': 'label', 'chrf_unreduced': 'chrF++'}, inplace=True)
    df_script_target = df_script_target[['label', 'chrF++']]
    df_script_target['label'] = df_script_target['label'].apply(func=lambda x: f"xx-{x}")
    df_script_target.to_csv(os.path.join(OUTPUT_PATH, 'xx_to_script.csv'), index=False)

    # Group by both source and target script
    df_script_both = df_script[['source_script', 'target_script', 'chrf_unreduced']].groupby(['source_script', 'target_script'])['chrf_unreduced'].apply(lambda x: chrf_unreduced_str_to_aggregate(x)).reset_index()
    df_script_both['label'] = df_script_both['source_script'] + '-' + df_script_both['target_script']
    df_script_both.rename(columns={'chrf_unreduced': 'chrF++'}, inplace=True)
    df_script_both = df_script_both[['label', 'chrF++']]
    df_script_both.to_csv(os.path.join(OUTPUT_PATH, 'script_to_script.csv'), index=False)


if __name__ == "__main__":
    # CONFIG:
    PARAM_COUNTS = ['1b', '3b']#['600m']#, '1b', '3b']
    EXPERIMENTS = ['scaffold', 'baseline', 'packed']
    for PARAM_COUNT in PARAM_COUNTS:
        for EXPERIMENT in EXPERIMENTS:
            INPUT_DATASET = 'flores200_devtest_mt5-{}-flores200-{}'.format(PARAM_COUNT, EXPERIMENT)
            OUTPUT_PATH = os.path.join(BASE_DIR, PARAM_COUNT, EXPERIMENT)

            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)

            # Read in data
            input_dataset = load_dataset(HF_PATH + INPUT_DATASET, split='devtest')
            df = pd.DataFrame(input_dataset)
            df = df[['source_lang', 'target_lang', 'chrf_unreduced']]

            # Run funcs:
            lang_to_lang(OUTPUT_PATH, df)
            avg_stats(OUTPUT_PATH, df)
            pretraining_mt5_analysis(OUTPUT_PATH)
            script_analysis(OUTPUT_PATH, df)


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
