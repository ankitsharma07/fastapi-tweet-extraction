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


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())

        len_st = len(selected_text)
        idx0 = -1
        idx1 = -1
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind+len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1

        tok_tweet = self.tokenizer.encode(sequence=self.sentiment[item], pair=tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[3:-1]
        # print(tok_tweet_tokens)
        # print(tok_tweet.offsets)
        # ['[CLS]', 'spent', 'the', 'entire', 'morning', 'in', 'a', 'meeting', 'w', '/',
        # 'a', 'vendor', ',', 'and', 'my', 'boss', 'was', 'not', 'happy', 'w', '/', 'them',
        # '.', 'lots', 'of', 'fun', '.', 'i', 'had', 'other', 'plans', 'for', 'my', 'morning', '[SEP]']
        targets = [0] * (len(tok_tweet_tokens) - 4)
        if self.sentiment[item] == "positive" or self.sentiment[item] == "negative":
            sub_minus = 8
        else:
            sub_minus = 7

        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1 - sub_minus:offset2 - sub_minus]) > 0:
                targets[j] = 1

        targets = [0] + [0] + [0] + targets + [0]

        #print(tweet)
        #print(selected_text)
        #print([x for i, x in enumerate(tok_tweet_tokens) if targets[i] == 1])
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1

        #print(targets_start)
        #print(targets_end)

        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * 3 + [1] * (len(tok_tweet_ids) - 3)

        padding_length = self.max_len - len(tok_tweet_ids)
        ids = tok_tweet_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        targets = targets + ([0] * padding_length)
        targets_start = targets_start + ([0] * padding_length)
        targets_end = targets_end + ([0] * padding_length)

        sentiment = [1, 0, 0]
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'tweet_tokens': " ".join(tok_tweet_tokens),
            'targets': torch.tensor(targets, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'padding_len': torch.tensor(padding_length, dtype=torch.long),
            'orig_tweet': self.tweet[item],
            'orig_selected': self.selected_text[item],
            'sentiment': torch.tensor(sentiment, dtype=torch.float),
            'orig_sentiment': self.sentiment[item]
        }

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

    model2 = BERTBaseUncased()
    model2.to(device)
    model2 = nn.DataParallel(model2)
    if torch.cuda.is_available():
        model2.load_state_dict(torch.load(config.MODEL_PATH))
    else:
        model2.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device("cpu")))
    model2.eval()

    # Test Data
    df_test = pd.read_csv(config.TEST_FILE)
    df_test.loc[:, "selected_text"] = df_test.text.values

    # Custom data
    test_tweet = ["Recession hit Veronique Branquinho, she has to quit her company, such a shame!"]
    test_sentiment = ["negative"]
    test_selected_text = test_tweet

    test_tweet_arr = np.asarray(test_tweet)
    test_sentiment_arr = np.asarray(test_sentiment)
    test_selected_text_arr = np.asarray(test_selected_text)


    # Load the TweetDataset
    # test_dataset = dataset.TweetDataset(
    #     tweet = df_test.text.values,
    #     sentiment = df_test.sentiment.values,
    #     selected_text = df_test.selected_text.values
    # )

    test_dataset = TweetDataset(
        tweet = test_tweet_arr,
        sentiment = test_sentiment_arr,
        selected_text = test_selected_text_arr
    )

    print(test_dataset)

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle = False,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers = 1
    )

    # Evaluation
    all_outputs = []
    fin_outputs_start = []
    fin_outputs_end = []
    fin_outputs_start2 = []
    fin_outputs_end2 = []
    fin_tweet_tokens = []
    fin_padding_lens = []
    fin_orig_selected = []
    fin_orig_sentiment = []
    fin_orig_tweet = []
    fin_tweet_token_ids = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            print("Hello", d)

            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            tweet_tokens = d["tweet_tokens"]
            padding_len = d["padding_len"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_sentiment = d["orig_sentiment"]
            orig_tweet = d["orig_tweet"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            sentiment = sentiment.to(device, dtype=torch.float)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            outputs_start2, outputs_end2 = model2(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())
            fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())
            fin_outputs_start2.append(torch.sigmoid(outputs_start2).cpu().detach().numpy())
            fin_outputs_end2.append(torch.sigmoid(outputs_end2).cpu().detach().numpy())

            fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
            fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())

            fin_tweet_tokens.extend(tweet_tokens)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_selected)
            fin_orig_tweet.extend(orig_tweet)

    fin_outputs_start = np.vstack(fin_outputs_start)
    fin_outputs_end = np.vstack(fin_outputs_end)
    fin_outputs_start2 = np.vstack(fin_outputs_start2)
    fin_outputs_end2 = np.vstack(fin_outputs_end2)

    fin_outputs_start = (fin_outputs_start + fin_outputs_start2) / 2
    fin_outputs_end = (fin_outputs_end + fin_outputs_end2) / 2

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
            mask_start = fin_outputs_start[j, 3:-1][:-padding_len] >= threshold
            mask_end = fin_outputs_end[j, 3:-1][:-padding_len] >= threshold
            tweet_token_ids = fin_tweet_token_ids[j, 3:-1][:-padding_len]
        else:
            mask_start = fin_outputs_start[j, 3:-1] >= threshold
            mask_end = fin_outputs_end[j, 3:-1] >= threshold
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
