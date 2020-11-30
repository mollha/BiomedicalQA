from random import randint, random
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial
import random
import os
import re



""" Completely overhaul the way data is read in
 Essentially, we need to read in text and produce training examples.
 
 
 
 """




class ELECTRADataProcessor(object):
    """Given a stream of input text, creates pre-training examples."""

    def __init__(self, hf_dset, tokenizer, max_length, text_col='text', lines_delimiter='\n',
                 minimize_data_size=True, apply_cleaning=True):
        # turn minimize data_size off because we are using a custom dataset
        # which does not do automatic padding like fastai.

        self.tokenizer = tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length

        self.hf_dset = hf_dset
        self.text_col = text_col
        self.lines_delimiter = lines_delimiter
        self.minimize_data_size = minimize_data_size
        self.apply_cleaning = apply_cleaning

    def map(self, **kwargs):
        "Some settings of datasets.Dataset.map for ELECTRA data processing"

        num_proc = kwargs.pop('num_proc', os.cpu_count())
        return self.hf_dset.my_map(
            function=self,
            batched=True,
            remove_columns=self.hf_dset.column_names,  # this is must b/c we will return different number of rows
            disable_nullable=True,
            input_columns=[self.text_col],
            writer_batch_size=10 ** 4,
            num_proc=num_proc,
            **kwargs
        )

    def __call__(self, texts):
        """
        Call method allows instances of classes to behave like functions.

        I am starting to think that texts is the WHOLE dataset, not just an individual batch.
        :param texts:
        :return:
        """

        if self.minimize_data_size:
            new_example = {'input_ids': [], 'sentA_length': []}
        else:
            new_example = {'input_ids': [], 'input_mask': [], 'segment_ids': []}

        for text in texts:  # for every doc
            for line in re.split(self.lines_delimiter, text):  # for every paragraph

                if re.fullmatch(r'\s*', line):
                    continue  # empty string or string with all space characters

                # filter out lines that are shorter than 80 characters
                if self.apply_cleaning and len(line) < 80:
                    continue

                example = self.add_line(line)
                if example:
                    for k, v in example.items():
                        new_example[k].append(v)

            # self.add_line() to current_length.
            # We want to check that there is at least one token to build an example from.
            if self._current_length != 0:
                example = self._create_example()
                for k, v in example.items():
                    new_example[k].append(v)

        return new_example

    def add_line(self, line):
        """
        Given a single line, clean it, convert it to token ids.
        Add the token ids to the list of current sentences.

        Call create example when we are exceeding target length.
        """

        """Adds a line of text to the current example being built."""
        # clean the line by removing leading and trailing spaces
        # and newlines.
        line = line.strip().replace("\n", " ").replace("()", "")

        # create tokens using the tokenizer provided
        tokens = self.tokenizer.tokenize(line)

        # convert the tokens to ids
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        self._current_sentences.append(token_ids)
        self._current_length += len(token_ids)

        if self._current_length >= self._target_length:
            return self._create_example()
        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # this is called when the target length is reached

        # 10% chance to only have one segment as in classification tasks
        print('Heya', self.minimize_data_size)

        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self._target_length - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in self._current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (len(first_segment) == 0 or
                    len(first_segment) + len(sentence) < first_segment_target_length or
                    (len(second_segment) == 0 and
                     len(first_segment) < first_segment_target_length and
                     random.random() < 0.5)):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(0, self._max_length - len(first_segment) - 3)]

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0

        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return self._make_example(first_segment, second_segment)

    def _make_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a train Example."""
        # Create a "sentence" of input ids from the first segment
        input_ids = [self.tokenizer.cls_token_id] + first_segment + [self.tokenizer.sep_token_id]


        # get the length of the input ids
        sentA_length = len(input_ids)

        # produce segment ids, all zeros, matching the length of the input ids
        segment_ids = [0] * sentA_length

        # if a second segment exists, then extend input_ids to include the ids of
        # the second segment and another separator token.
        if second_segment:
            input_ids += second_segment + [self.tokenizer.sep_token_id]

            # extend segment ids to length of second segment + 1,
            segment_ids += [1] * (len(second_segment) + 1)

        if self.minimize_data_size:
            return {
                'input_ids': input_ids,
                'sentA_length': sentA_length,
            }
        else:
            # pad the input data
            input_mask = [1] * len(input_ids)
            input_ids += [0] * (self._max_length - len(input_ids))
            input_mask += [0] * (self._max_length - len(input_mask))
            segment_ids += [0] * (self._max_length - len(segment_ids))
            return {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            }


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

        input_ids, sent_lengths = inputs
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids)

        # return self.learn.xb, self.learn.yb
        return (masked_inputs, sent_lengths, is_mlm_applied, labels), (labels,)



