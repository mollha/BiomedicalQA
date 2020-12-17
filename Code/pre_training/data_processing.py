from random import randint, random
import pandas as pd
import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from functools import partial
import random
import sys


# ------------ CUSTOM PYTORCH DATASET IMPLEMENTATIONS ------------
class MappedCSVDataset(Dataset):
    """
    Mapped Pytorch dataset which loads in a sub-dataset from separate pre-training data files.
    """

    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file, delimiter='|', error_bad_lines=False, skiprows=1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.dataframe.iloc[idx, 0]


class IterableCSVDataset(IterableDataset):
    """
    Custom CSV pytorch dataset for iterating through all data files.
    This class creates a MappedCSVDataset for each unique file.
    """

    def __init__(self, data_directory: str, batch_size: int, device, transform=None, drop_incomplete_batches=True):
        super(IterableCSVDataset).__init__()

        self._batch_size = batch_size
        self._transform = transform
        self._device = device
        self._dataset_size = None

        self._drop_incomplete_batches = drop_incomplete_batches
        self._list_paths_to_csv = list(pathlib.Path(data_directory).glob('*.csv'))
        self._list_paths_to_csv.sort()

        if len(self._list_paths_to_csv) == 0:
            raise FileNotFoundError("CSV files not found in directory {}. Pre-training cancelled."
                                    .format(data_directory))

    def __iter__(self):
        # When calling for a new iterable, reset csv_idx and current_iterator
        self._current_csv_idx = 0
        self._current_iterator = None
        self._intermediate_dataset_size = 0
        return self

    def __next__(self):
        while True:
            try:
                batch = next(self._current_iterator)
                num_samples_in_batch = len(batch)

                if self._drop_incomplete_batches and num_samples_in_batch < self._batch_size:
                    # this batch is an incomplete batch, so drop it
                    continue

                self._intermediate_dataset_size += num_samples_in_batch
                if self._transform:
                    batch = self._transform.process_batch(batch)
                return batch

            except (StopIteration, TypeError) as e:

                # If the current_iterator has been exhausted, or does not yet exist, then we need to create one.
                if type(e) != TypeError:
                    self._current_csv_idx += 1

                # check that there are files remaining
                if self._current_csv_idx < len(self._list_paths_to_csv):
                    csv_name = self._list_paths_to_csv[self._current_csv_idx]  # get the name of the next file
                    self._current_iterator = self.build_iterator_from_csv(csv_name)  # pandas.io.parsers.TextFileReader
                else:
                    # there is no more data to explore
                    if self._dataset_size is None:
                        self._dataset_size = self._intermediate_dataset_size
                        sys.stderr.write("Dataset size: {}".format(self._dataset_size))
                    return None

    def build_iterator_from_csv(self, path_to_csv):
        sys.stderr.write("\nReading CSV {}".format(str(path_to_csv)[-11:]))

        csv_dataset = MappedCSVDataset(path_to_csv)
        data_loader = DataLoader(csv_dataset, batch_size=self._batch_size, shuffle=True)
        return iter(data_loader)

    def resume_from_step(self, training_step):
        for i in range(training_step):
            next(self)
        sys.stderr.write("\nResuming training from csv {} ({})\n".format(self._current_csv_idx, str(
            self._list_paths_to_csv[self._current_csv_idx])[-11:]))


# ------------ DEFINE DATA TRANSFORMATION METHOD FROM RAW TEXT TO TOKENS ------------
class ELECTRADataProcessor(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self._max_length = max_length
        self._target_length = max_length

    def __call__(self, text):
        """
        Call method allows instances of classes to behave like functions.
        :param text: A string representing a sample of text
        :return: an array containing the transformed sample
        """
        text = str(text)  # ensure that text is a string
        # decide if the target length should be shorter than the max length for this particular call
        self._target_length = randint(5, self._max_length) if random.random() < 0.05 else self._max_length
        processed_sample = self.process_sample(text)
        return np.array(processed_sample, dtype=np.int32)

    def process_sample(self, sample: str) -> list:
        """
        Transform a single sample into a list of token ids
        :param sample: a string from the dataset (i.e. a single example)
        :return: list of token ids
        """
        line = sample.strip().replace("\n", " ").replace("()", "")
        tokens = self.tokenizer.tokenize(line)  # create tokens using the tokenizer provided
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # convert the tokens to ids - returns list of ids

        # reduce this to the max_length - 2 (accounts for added tokens) - snip to target_length
        additional_tokens = len(token_ids) - self._target_length + 2

        if additional_tokens > 0:  # token_ids must be trimmed if they are longer than the target length
            first_half = randint(0, additional_tokens)  # how many tokens will be cut from the start
            second_half = additional_tokens - first_half  # how many tokens will be cut from the end
            token_ids = token_ids[first_half: len(token_ids) - second_half]  # cut additional tokens

        # Create a "sentence" of input ids from the first segment
        input_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
        input_ids += [self.tokenizer.pad_token_id] * (self._max_length - len(input_ids))  # add padding up to max_length
        return input_ids

    def process_batch(self, texts):
        """
        Given a batch of texts, process each text individually then recombine into a tensor.
        :param texts: a batch of samples
        :return: a LongTensor containing the transformed batch
        """
        sample_list = []
        for text in texts:
            sample_list.append(self(text))  # process the sample then combine
        return torch.LongTensor(np.stack(sample_list))  # construct a new tensor from the list of processed samples


# ------------ DEFINE DATA TRANSFORMATION METHOD FROM RAW TEXT TO TOKENS ------------
# Adapted from Richard Wang's implementation on Github.
# (https://github.com/richarddwang/electra_pytorch/blob/80d1790b6675720832c5db5f22b7e036f68208b8/pretrain.py#L170).
class MaskedLM:
    def __init__(self, mask_tok_id, special_tok_ids, vocab_size, ignore_index=-100, **kwargs):
        self.ignore_index = ignore_index
        self.mask_input_tokens = partial(self.mask_tokens, mask_token_index=mask_tok_id,
                                         special_token_indices=special_tok_ids,
                                         vocab_size=vocab_size, ignore_index=-100, **kwargs)

    def mask_batch(self, input_ids) -> tuple:
        """
        Compute the masked inputs - in ELECTRA, MLM is used, therefore the raw batches should
        not be passed to the model.
        :return: None
        """
        masked_inputs, labels, is_mlm_applied = self.mask_input_tokens(input_ids)
        return (masked_inputs, is_mlm_applied, labels), (labels,)

    @staticmethod
    def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1,
                    original_prob=0.1, ignore_index=-100):
        """
        Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK,
        replace_prob% random, original_prob% original within mlm_probability% of tokens in the sentence.
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

        # mask (mlm_probability * (1-replace_prob-original_prob))
        mask_prob = 1 - replace_prob - original_prob
        mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
        inputs[mask_token_mask] = mask_token_index

        # replace with a random token (mlm_probability * replace_prob)
        if int(replace_prob) != 0:
            rep_prob = replace_prob / (replace_prob + original_prob)
            replace_token_mask = torch.bernoulli(
                torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
            random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
            inputs[replace_token_mask] = random_words[replace_token_mask]

        # do nothing (mlm_probability * original_prob)
        pass

        return inputs, labels, mlm_mask
