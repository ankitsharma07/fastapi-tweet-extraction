import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import utils

def loss_fn(out1, out2, target1, target2):
    loss1 = nn.BCEWithLogitsLoss()(out1, target1)
    loss2 = nn.BCEWithLogitsLoss()(out2, target2)

    return loss1 + loss2

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for batch_index, datasets, in enumerate(tk0):
        ids = datasets["ids"]
        token_type_ids = datasets["token_type_ids"]
        mask = datasets["mask"]
        targets_start = datasets["targets_start"]
        targets_end = datasets["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)
        optimizer.zero_grad()
        out1, out2 = model (
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(out1, out2, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)

def eval_fn(data_loader, model, device):
    model.eval()
    fin_output_start = []
    fin_output_end = []
    fin_padding_lens = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_tweet = []

    for batch_index, datasets, in enumerate(data_loader):
        ids = datasets["ids"]
        token_type_ids = datasets["token_type_ids"]
        mask = datasets["mask"]
        tweet_tokens = datasets["tweet_tokens"]
        padding_len = datasets["padding_length"]
        orig_sentiment = datasets["original_sentiment"]
        orig_selected = datasets["original_selected_text"]
        orig_tweet = datasets["original_tweet"]


        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        out1, out2 = model (
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        fin_output_start.append(torch.sigmoid(out1).cpu().detach().numpy())
        fin_output_end.append(torch.sigmoid(out1).cpu().detach().numpy())
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())

        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(orig_sentiment)
        fin_orig_selected.extend(orig_selected)
        fin_orig_tweet.extend(orig_tweet)

    # select
    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    threshold = 0.2
    jaccards = []

    # iterate predictions
    for j in range(len(fin_tweet_tokens)):
        target_string = fin_orig_selected[j]
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_lens[j]
        original_tweet = fin_orig_tweet[j]
        sentiment = fin_orig_sentiment[j]

        if padding_len > 0:
            mask_start = fin_output_start[j, :][:-padding_len] >= threshold
            mask_end = fin_output_end[j, :][:-padding_len] >= threshold
        else:
            mask_start = fin_output_start[j, :] >= threshold
            mask_end = fin_output_end[j, :] >= threshold

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

        output_tokens = [x for p, x in enumerate(tweet_tokens.split()) if mask[p] == 1]
        output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]

        final_output = ""

        for out in output_tokens:
            if out.startswith("##"):
                final_output = final_output + out[2:]
            elif len(out) == 1 and out in string.punctuation:
                final_output = final_output + out
            else:
                final_output = final_output + " " + out
        final_output = final_output.strip()

        if sentiment == "neutral" or len(original_tweet.split()) < 4:
            final_output = original_tweet

        jac = utils.jaccard(target_string.strip(), final_output.strip())
        jaccards.append(jac)

    mean_jaccard = np.mean(jaccards)

    return mean_jaccard
