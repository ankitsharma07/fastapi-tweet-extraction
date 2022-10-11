#!/usr/bin/env python3

import os
import re

import numpy as np
import pandas as pd
import torch
from torch.cuda import is_available
import torch.nn as nn
import transformers
from tqdm import tqdm

import config
import engine
import dataset
from model import BERTBaseUncased
import utils


def run_inference():
    device = torch.device("cpu")
    model = BERTBaseUncased()
    model.to(device)
    model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(config.MODEL_PATH))
    else:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    # Test Data
    df_test = pd.read_csv(config.TEST_FILE)
    df_test.loc[:, "selected_text"] = df_test.text.values

    # Load the TweetDataset
    test_dataset = dataset.TweetDataset(
        tweet = df_test.text.values,
        sentiment = df_test.sentiment.values,
        selected_text = df_test.selected_text.values
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle = False,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers = 1
    )

    # Eval
    all_outputs = []
    fin_output_start = []
    fin_output_end = []
    fin_padding_lens = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_tweet = []
    fin_tweet_token_ids = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            tweet_tokens = d["tweet_tokens"]
            padding_len = d["padding_length"]
            sentiment = d["sentiment"]
            orig_selected = d["original_selected_text"]
            orig_sentiment = d["original_sentiment"]
            orig_tweet = d["original_tweet"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            sentiment = sentiment.to(device, dtype=torch.float)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            fin_output_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())
            fin_output_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())

            fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
            fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())

            fin_tweet_tokens.extend(tweet_tokens)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_selected)
            fin_orig_tweet.extend(orig_tweet)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    fin_output_start = (fin_output_start ) / 2
    fin_output_end = (fin_output_end ) / 2

    fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)

    jaccards = []
    threshold = 0.3
    for j in range(len(fin_tweet_tokens)):
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        sentiment_val = fin_orig_sentiment[j]

        if padding_len > 0:
            mask_start = fin_output_start[j, 3:-1][:-padding_len] >= threshold
            mask_end = fin_output_end[j, 3:-1][:-padding_len] >= threshold
            tweet_token_ids = fin_tweet_token_ids[j, 3:-1][:-padding_len]
        else:
            mask_start = fin_output_start[j, 3:-1] >= threshold
            mask_end = fin_output_end[j, 3:-1] >= threshold
            tweet_token_ids = fin_tweet_token_ids[j, 3:-1]

        mask = [0] * len(mask_start)
        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]
        if len(idx_start) > 0:
            idx_start = idx_start[0]
            if len(idx_end) > 0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
        else:
            idx_start = 0
            idx_end = 0

        for mj in range(idx_start, idx_end + 1):
            mask[mj] = 1

        output_tokens = [x for p, x in enumerate(tweet_token_ids) if mask[p] == 1]

        filtered_output = config.TOKENIZER.decode(output_tokens)
        filtered_output = filtered_output.strip().lower()

        if sentiment_val == "neutral":
            filtered_output = original_tweet

        all_outputs.append(filtered_output.strip())

        print(all_outputs)

if __name__ == '__main__':
    run_inference()
