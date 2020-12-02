from random import randint, random
import os
import re
import pandas as pd
import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from functools import partial
import random
import os
import re


# Create Dataset
class IterableCSVDataset(IterableDataset):
    """
    Custom CSV pytorch dataset for reading
    """

    def __init__(self, data_directory: str, batch_size: int, shuffle=True):
        super(IterableCSVDataset).__init__()

        self._list_paths_to_csv = list(pathlib.Path(data_directory).glob('*.csv'))
        self._current_csv_idx = 0   # keep track of the current file we are reading from
        self._batch_size = batch_size
        self._current_iterator = None
        self._shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                batch = next(self._current_iterator)
                return batch if not self._shuffle else batch.sample(frac=1).reset_index(drop=True)
            except (StopIteration, TypeError):
                print("caught")
                # If the current_iterator has been exhausted, or does not yet exist, then we need to create one.
                self._current_csv_idx += 1

                # check that there are files remaining
                if self._current_csv_idx < len(self._list_paths_to_csv):
                    csv_name = self._list_paths_to_csv[self._current_csv_idx]  # get the name of the next file
                    self._current_iterator = self.build_iterator_from_csv(csv_name)  # pandas.io.parsers.TextFileReader
                else:
                    # there is no more data to explore
                    raise StopIteration

    def build_iterator_from_csv(self, path_to_csv):
        return pd.read_csv(path_to_csv, skiprows=1, chunksize=self._batch_size, iterator=True, delimiter="|")


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, csv_file, transform=None):
        super(CSVDataset, self).__init__()
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 0]
        if self.transform:
            sample = self.transform(text)
            return sample
        return text


class ELECTRADataProcessor(object):
    def __init__(self, tokenizer, max_length, device, text_col='text', lines_delimiter='\n'):
        # turn minimize data_size off because we are using a custom dataset
        # which does not do automatic padding like fastai.

        self.tokenizer = tokenizer
        self._max_length = max_length
        self._target_length = max_length
        self.device = device

        self.text_col = text_col
        self.lines_delimiter = lines_delimiter

    def __call__(self, text):
        """
        Call method allows instances of classes to behave like functions.

        texts is the WHOLE dataset, not just an individual batch.
        :param text:
        :return:
        """

        # decide on the target length
        # todo remove import random and simplify random.random
        self._target_length = randint(5, self._max_length) if random.random() < 0.05 else self._max_length
        processed_sample = self.process_sample(text)
        return np.array(processed_sample)


    def process_sample(self, sample: str):
        """
        Sample is a string from the dataset (i.e. a single example)
        :param sample:
        :return:
        """
        line = sample.strip().replace("\n", " ").replace("()", "")

        # create tokens using the tokenizer provided
        tokens = self.tokenizer.tokenize(line)

        # convert the tokens to ids - returns list of token ids
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # reduce this to the max_length - 2 (accounts for added tokens)
        # snip to target_length
        additional_tokens = len(token_ids) - self._target_length - 2

        if additional_tokens > 0:
            # token_ids must be trimmed
            first_half = randint(0, additional_tokens)
            second_half = additional_tokens - first_half
            token_ids = token_ids[first_half : len(token_ids) - second_half]

        # Create a "sentence" of input ids from the first segment
        input_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]

        # add padding to max_length
        input_ids += [self.tokenizer.pad_token_id] * (self._max_length - len(input_ids))

        return input_ids



"""
Modified from HuggingFace/transformers (https://github.com/huggingface/transformers/blob/0a3d0e02c5af20bfe9091038c4fd11fb79175546/src/transformers/data/data_collator.py#L102). 
It is a little bit faster cuz 
- intead of a[b] a on gpu b on cpu, tensors here are all in the same device
- don't iterate the tensor when create special tokens mask
And
- doesn't require huggingface tokenizer
- cost you only 550 µs for a (128,128) tensor on gpu, so dynamic masking is cheap   
"""


def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1,
                orginal_prob=0.1, ignore_index=-100):
    """
    Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence.
    * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
    """

    device = inputs.device
    labels = inputs.clone()

    # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)

    for sp_id in special_token_indices:
        special_tokens_mask = special_tokens_mask | (inputs == sp_id)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    mlm_mask = torch.bernoulli(probability_matrix).bool()
    labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

    # mask (mlm_probability * (1-replace_prob-orginal_prob))
    mask_prob = 1 - replace_prob - orginal_prob
    mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
    inputs[mask_token_mask] = mask_token_index

    # replace with a random token (mlm_probability * replace_prob)
    if int(replace_prob) != 0:
        rep_prob = replace_prob / (replace_prob + orginal_prob)
        replace_token_mask = torch.bernoulli(
            torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
        inputs[replace_token_mask] = random_words[replace_token_mask]

    # do nothing (mlm_probability * orginal_prob)
    pass

    return inputs, labels, mlm_mask


class MaskedLM:
    def __init__(self, mask_tok_id, special_tok_ids, vocab_size, ignore_index=-100, **kwargs):
        self.ignore_index = ignore_index

        # assumes for_electra is true
        self.mask_tokens = partial(mask_tokens, mask_token_index=mask_tok_id, special_token_indices=special_tok_ids,
                                   vocab_size=vocab_size, ignore_index=-100, **kwargs)

    def mask_batch(self, inputs) -> tuple:
        """
        Compute the masked inputs - in ELECTRA, MLM is used, therefore the raw batches should
        not be passed to the model.
        :return: None

        ---- Attributes of Learner: ----
        xb: last input drawn from self.dl (current DataLoader used for iteration), potentially modified by callbacks
        yb: last target drawn from self.dl (potentially modified by callbacks).
        --------------------------------
        """
        input_ids = inputs[0]
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids)

        # return self.learn.xb, self.learn.yb
        return (masked_inputs, is_mlm_applied, labels), (labels,)



