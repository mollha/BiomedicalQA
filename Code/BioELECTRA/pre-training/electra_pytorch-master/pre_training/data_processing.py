from random import randint, random
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial


# Create dummy csv data
nb_samples = 111
a = np.arange(nb_samples)
df = pd.DataFrame(a, columns=['data'])
df.to_csv('data.csv', index=False)

path_to_csv = 'data.csv'
chunksize=10

my_csv = pd.read_csv(path_to_csv, header=[0], chunksize=chunksize, iterator=True)
iterator = next(my_csv)
# print(my_csv.read(2))
# print(my_csv.read(2))


# Create Dataset
class CSVDataset(torch.utils.data.IterableDataset):
    def __init__(self, path_to_csv, chunk_size, header=False):
        super(CSVDataset).__init__()

        self.path_to_csv = path_to_csv
        self.chunk_size = chunk_size
        self.header = header
        self.iterator = self.get_iterator()

        # self.len = self.count_csv_rows()

    def __iter__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            return pd.DataFrame()

    # def __len__(self):
    #     return self.len

    def get_iterator(self):
        return pd.read_csv(self.path_to_csv, header=[0] if self.header else None, chunksize=self.chunk_size, iterator=True)

    # def count_csv_rows(self) -> int:
    #     """
    #     Given a path to a CSV file, count the number of data samples using lazy-loading.
    #     Each chunk is iterated over and the total sizes are summed to calculate overall length.
    #
    #     :param header: flag indicating whether the csv file contains a header row
    #     :return: number of rows in csv
    #     """
    #
    #     tfr = self.get_iterator()
    #     row_count = 0
    #
    #     while True:
    #         try:
    #             chunk = next(tfr)
    #             row_count += len(chunk)
    #         except StopIteration:
    #             return row_count - 1 if self.header else row_count


dataset = CSVDataset('data.csv', chunk_size=10, header=True)
print(len(dataset))

# loader = DataLoader(dataset, batch_size=10, num_workers=1, shuffle=False)
#
# for batch_idx, data in enumerate(loader):
#     print('batch: {}\tdata: {}'.format(batch_idx, data))


# class PretrainDataset(torch.utils.data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, list_IDs, labels):
#         'Initialization'
#         self.labels = labels
#         self.list_IDs = list_IDs
#
#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)
#
#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]
#
#         # Load data and get label
#         X = torch.load('data/' + ID + '.pt')
#         y = self.labels[ID]
#
#         return X, y



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


class ELECTRADataProcessor(object):
    """Given a stream of input text, creates pre-training examples."""

    def __init__(self, hf_dset, tokenizer, max_length, text_col='text', lines_delimiter='\n',
                 minimize_data_size=True, apply_cleaning=True):
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
        # This is a powerful method inspired by the tf.data.Dataset map method
        # applies a processing function to each example in a dataset
        # can be done independently or in a batch.

        """
        batched = True means that batches of examples are provided to function
        input_columns are the columns to be passed into function as positional arguments
        If None, a dict mapping to all formatted columns is passed as one argument.
        
        remove_columns =  Remove a selection of columns while doing the mapping.
        function = the function to be applied to batches
        """

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
        :param texts:
        :return:
        """
        if self.minimize_data_size:
            new_example = {'input_ids': [], 'sentA_length': []}
        else:
            new_example = {'input_ids': [], 'input_mask': [], 'segment_ids': []}

        for text in texts:  # for every doc in batch

            for line in re.split(self.lines_delimiter, text):  # for every paragraph in doc

                if re.fullmatch(r'\s*', line):  # empty string or string with all space characters
                    continue

                # filter out lines that are shorter than 80 characters
                if self.apply_cleaning and len(line) < 80:
                    continue

                example = self.add_line(line)
                if example:
                    for k, v in example.items():
                        new_example[k].append(v)

            if self._current_length != 0:
                example = self._create_example()
                for k, v in example.items():
                    new_example[k].append(v)

        return new_example

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        # clean the line by removing leading and trailing spaces and newlines.
        line = line.strip().replace("\n", " ").replace("()", "")

        tokens = self.tokenizer.tokenize(line)  # create tokens using the tokenizer provided
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # convert the tokens to ids

        self._current_sentences.append(token_ids)
        self._current_length += len(token_ids)

        if self._current_length >= self._target_length:
            return self._create_example()

        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # 10% chance to only have one segment as in classification tasks
        # -3 due to not yet having [CLS]/[SEP] tokens in the input text
        segment_1_target_length = 100000 if random() < 0.1 else (self._target_length - 3) // 2
        segment_1, segment_2 = [], []

        for sentence in self._current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length

            sentence_under_length = (len(segment_1) + len(sentence) < segment_1_target_length)
            first_segment_under_length = (len(segment_1) < segment_1_target_length)

            if (len(segment_1) == 0 or sentence_under_length or
                    (len(segment_2) == 0 and first_segment_under_length and random() < 0.5)):
                segment_1 += sentence
            else:
                segment_2 += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        # segment examples if they exceed the maximum length
        segment_1 = segment_1[:self._max_length - 2]
        segment_2 = segment_2[:max(0, self._max_length - len(segment_1) - 3)]

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0

        # small chance for random-length instead of max_length-length example
        self._target_length = randint(5, self._max_length) if random() < 0.05 else self._max_length

        return self._make_example(segment_1, segment_2)

    def _make_example(self, segment_1, segment_2):
        """Converts two "segments" of text into a tf.train.Example."""
        # The training example is a dictionary of "input_ids" "sentA_length" not sure which one this is

        # sep_token: special token separating two different sentences in the same input
        # cls_token: special token representing the class of the input. It is a sentence-level
        # representation for classification

        # this is used by BERT and ELECTRA

        input_ids = [self.tokenizer.cls_token_id] + segment_1 + [self.tokenizer.sep_token_id]
        sentA_length = len(input_ids)

        # length of input ids is the sentA_length

        segment_ids = [0] * sentA_length

        if segment_2:
            input_ids += segment_2 + [self.tokenizer.sep_token_id]
            segment_ids += [1] * (len(segment_2) + 1)

        if self.minimize_data_size:
            return {
                'input_ids': input_ids,
                'sentA_length': sentA_length,
            }
        else:
            input_mask = [1] * len(input_ids)
            input_ids += [0] * (self._max_length - len(input_ids))
            input_mask += [0] * (self._max_length - len(input_mask))
            segment_ids += [0] * (self._max_length - len(segment_ids))
            return {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            }
