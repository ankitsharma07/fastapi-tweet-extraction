import numpy as np
import pandas as pd
import torch

import config


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())

        len_selected_text = len(selected_text)
        idx0 = -1
        idx1 = -1

        for index in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[index: index + len_selected_text] == selected_text:
                idx0 = index
                idx1 = index + len_selected_text - 1
                break # only interested in first match

        char_targets = [0] * len(tweet)

        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1
        # wherever there is a match
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # https://github.com/huggingface/tokenizers checkout this repo to understand more about tokenizers
        tokenized_tweet = self.tokenizer.encode(tweet)
        tokenized_tweet_tokens = tokenized_tweet.tokens
        tokenized_tweet_ids = tokenized_tweet.ids
        tokenized_tweet_offsets = tokenized_tweet.offsets[1:-1] # The first and last tokens are always [CLS] and [SEP]

        targets = [0] * (len(tokenized_tweet_tokens) - 2 )

        # [0, 0, 0, 0, 0, 0, 0]
        for j, (offset1, offset2) in enumerate(tokenized_tweet_offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                targets[j] = 1

        # targets = [0, 0, 1, 1, 1, 0, 0]

        targets = [0] + targets + [0] # CLS , SEP why??
        targets_start = [0] *  len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1

        # attention masks
        mask = [1] * len(tokenized_tweet_ids)
        token_type_ids = [0] * len(tokenized_tweet_ids)

        padding_length = self.max_len - len(tokenized_tweet_ids)
        ids = tokenized_tweet_ids + [0] * padding_length
        mask = mask + [0] * padding_length
        token_type_ids = token_type_ids + [0] * padding_length
        targets = targets + [0] * padding_length
        targets_start = targets_start + [0] * padding_length
        targets_end = targets_end + [0] * padding_length

        # using sentiment
        sentiment = [1, 0, 0] # neutral
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "targets_start": torch.tensor(targets_start, dtype=torch.long),
            "targets_end": torch.tensor(targets_end, dtype=torch.long),
            "padding_length": torch.tensor(padding_length, dtype=torch.long),
            "tweet_tokens": " ".join(tokenized_tweet_tokens),
            "original_tweet": self.tweet[item],
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "original_sentiment": self.sentiment[item],
            "original_selected_text": self.selected_text[item]
        }


# testing
# if __name__ == '__main__':
#     df = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
#     dataset = TweetDataset(
#         tweet = df.text.values,
#         sentiment = df.sentiment.values,
#         selected_text = df.selected_text.values
#     )

#     print(dataset[0])
