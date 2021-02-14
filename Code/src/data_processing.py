from random import randint, random
import pandas as pd
import pathlib
import torch
import time
import unidecode
from tqdm import trange
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from functools import partial
import random

# ----------------------- SPECIFY DATASET PATHS -----------------------
# the folder structure of bioasq is different to squad, as we need to download matching articles
datasets = {
    "bioasq": {"train": ["raw_data/training9b.json"],
               "test": ["raw_data/8B1_golden.json", "raw_data/8B2_golden.json", "raw_data/8B3_golden.json",
                        "raw_data/8B4_golden.json", "raw_data/8B5_golden.json"]
               },
    "squad": {
        "train": ["train-v2.0.json"],
        "test": ["dev-v2.0.json"],
    }
}


# ------------- DEFINE TRAINING FEATURE CLASSES ------------
class BinaryFeature:
    def __init__(self, question_id, input_ids, attention_mask, token_type_ids, answer_text):
        self._question_id = question_id
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self._token_type_ids = token_type_ids
        self._answer_text = answer_text.lower()

        if self._answer_text == "yes":
            self._label = 1
        elif self._answer_text == "no":
            self._label = 0
        else:
            raise Exception('Answer text "{}" is not yes or no.'.format(self._answer_text))

    def get_features(self):
        return (
            self._question_id,
            self._input_ids,
            self._attention_mask,
            self._token_type_ids,
            self._answer_text,
            self._label
        )


class FactoidFeature:
    def __init__(self, question_id, input_ids, attention_mask, token_type_ids, answer_start, answer_end,
                 answer_text):
        self._question_id = question_id
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self._token_type_ids = token_type_ids
        self._answer_start = answer_start
        self._answer_end = answer_end
        self._answer_text = answer_text

    def get_features(self):
        return (
            self._question_id,
            self._input_ids,
            self._attention_mask,
            self._token_type_ids,
            self._answer_text,
            self._answer_start,
            self._answer_end,
        )


def sub_tokenize_answer_tokens(tokenizer, pre_token, sub_tokens, pre_token_absolute_start, match_position):
    """
    Given that we know which token is found at a given position in the original context, we need to break this
    token into sub-tokens if the goal position is not somewhere in the middle of the token.

    :param match_position: the position we want to find the matching token for
    :param pre_token: the pre-token that was created by splitting the original context on whitespace
    :param tokenizer: the tokenizer used to tokenize text
    :param sub_tokens: the sub_tokens created when pre_token is tokenized
    :return:
    """
    # print('pre_token', pre_token)
    # print('subtokens', sub_tokens)

    # Pass over the sub_tokens and condense multiple [UNK] tokens in a row into a single [UNK]
    sub_tokens_condensed = []
    for idx in range(len(sub_tokens)):
        sub_token = sub_tokens[idx]

        # If sub-token is an unk token, we need to put it in the condensed list if the previous token was not also unk
        if sub_token == tokenizer.unk_token and (
                len(sub_tokens_condensed) == 0 or sub_tokens_condensed[-1] != tokenizer.unk_token):
            sub_tokens_condensed.append(sub_token)
        elif sub_token != tokenizer.unk_token:
            sub_tokens_condensed.append(sub_token)

    # Pass over the condensed sub_tokens and find their start and end positions
    sub_position_relative_mapping = []  # a list containing [token, [start_pos, end_pos]]
    pre_token_relative_position = 0
    for sub_token in sub_tokens_condensed:
        if sub_token == tokenizer.unk_token:  # we have found an unknown token
            sub_position_relative_mapping.append([sub_token, None])
        else:
            text_sub_token = sub_token.lstrip('#')  # remove leading hash-tags to convert token to string
            try:
                position_of_sub_token = pre_token.index(text_sub_token, pre_token_relative_position)
            except ValueError:  # Substring text_sub_token not found in pre_token
                continue

            sub_position_relative_mapping.append(
                [sub_token, [pre_token_absolute_start + position_of_sub_token, pre_token_absolute_start + position_of_sub_token + len(text_sub_token)]])
            pre_token_relative_position = position_of_sub_token + len(text_sub_token)  # skip past end of current token

    # Fill in the blanks for the unknown tokens that we had to skip on the first pass
    for idx in range(len(sub_position_relative_mapping)):
        # current_pair is a tuple of sub_token and start_end_pair (start and end positions)
        current_pair = sub_position_relative_mapping[idx]
        last_pair = None if idx == 0 else sub_position_relative_mapping[idx - 1]

        if current_pair[1] is None:  # need to populate the start position of the current pair
            if last_pair is None:  # current_pair represents the first token
                current_pair[1] = [pre_token_absolute_start, None]
            else:
                # the position where the previous token ended is 1 behind the start position of the current token
                current_pair[1] = [last_pair[1] + 1,
                                   None if idx < len(sub_position_relative_mapping) - 1 else pre_token_absolute_start + len(pre_token)]

        # need to populate the end position of the last pair
        if last_pair is not None and last_pair[0] == tokenizer.unk_token:
            # the position where the current token started is 1 ahead the end position of the last token
            last_pair[1][1] = current_pair[1][0] - 1

    # At this stage, we should have a mapping of every sub_token to its start and end positions
    for idx, (sub_token, (st_start_pos, st_end_pos)) in enumerate(sub_position_relative_mapping):
        if st_start_pos == match_position or st_end_pos == match_position:  # start position begins at the start of the current token, not in the middle
            # we can keep this token intact
            return sub_position_relative_mapping
        elif st_start_pos < match_position < st_end_pos:  # start position in the middle of the current token
            # we need to repeat this process to break down the current token further
            new_pre_token = sub_token.lstrip('#')  # remove leading hash-tags to convert token to string
            num_sub_tokens = tokenizer.tokenize(new_pre_token)
            # print('Num sub tokens', num_sub_tokens)

            if len(num_sub_tokens) == 1 and num_sub_tokens[0] == new_pre_token:
                lone_subtoken = num_sub_tokens.pop()

                # print('Num sub tokens again', num_sub_tokens)
                # treat subtokens as a list of characters
                num_sub_tokens = ["##{}".format(t) if i != 0 else t for i, t in enumerate(lone_subtoken)]
                # print('Num sub tokens again again', num_sub_tokens)


            # recursive call to further break down sub-token
            sub_mapping = sub_tokenize_answer_tokens(tokenizer, new_pre_token, num_sub_tokens, st_start_pos,
                                                     match_position)

            return sub_position_relative_mapping[0:idx] + sub_mapping + sub_position_relative_mapping[idx + 1:]


def convert_examples_to_features(examples, tokenizer, max_length):
    # What happens with evaluation if we don't know the answer?
    # we could do something interesting with these sorts of predictions, where we sub-tokenize around the predicted answer
    # and ask for the predictions again - let's look at this for making predictions
    # Assumes that the start and end positions actually point to the answer

    metrics = {
        "non_match_gt": 0,
        "non_match_pt": 0,
        "empty_mapping": 0,
    }

    feature_list = []
    for example_number in trange(len(examples), desc="Examples \u2b62 Features"):
        example = examples[example_number]
        question, short_context = example._question, example._short_context  # get context and question from example

        if example._question_type == "yesno":  # if we're looking at a yes/no question
            # concatenate context with question - [CLS] SHORT_CONTEXT [SEP] QUESTION [SEP]
            tokenized_input = tokenizer(question, short_context, padding="max_length", truncation="only_second",
                                        max_length=max_length)  # only truncate the second sequence

            # prepare the input_ids, attention_mask and token_type_ids as tensors
            input_ids = tokenized_input["input_ids"]
            attention_mask = tokenized_input["attention_mask"]
            token_type_ids = tokenized_input["token_type_ids"]
            feature = BinaryFeature(example._question_id, input_ids, attention_mask,
                                    token_type_ids, example._answer)
            feature_list.append(feature)
            # todo handle impossible yes no questions / ones without the answer.
            continue

        # If question type is factoid or list (i.e. not a yes/no question). Given start and end positions in the text,
        # we now need to find the tokenized start and end positions of the answer(s)
        text_start_pos = example._answer_start
        text_end_pos = example._answer_end
        answer = example._answer

        # if text_start_pos and text_end_pos are -1, then we have an impossible question.
        # if text_start_pos and text_end_pos are None, then we have a test question with no answer.
        if text_start_pos is None and text_end_pos is None:
            tokenized_input = tokenizer(question, short_context, padding="max_length", truncation="only_second",
                                        max_length=max_length)  # only truncate the second sequence

            # If it is not included, for impossible instances the target prediction
            # for both start and end (tokenized) position is 0, i.e. the [CLS] token
            # This is -1 for examples and 0 for features, as tokenized pos in features & char pos in examples
            feature = FactoidFeature(example._question_id, tokenized_input["input_ids"],
                                     tokenized_input["attention_mask"], tokenized_input["token_type_ids"],
                                     None, None, None)
            feature_list.append(feature)
            continue

        # Lets create a mapping of characters in the original context to the position in the tokenized context.
        # This will help us to find the tokens that correspond to the answer.
        pre_tokenized_context = short_context.split()  # split on whitespace (lower-case and remove accents done prior)

        # It is important for us to define the context position mapping before we tokenize.
        # After tokenization, we will have some [UNK] tokens which cannot be directly mapped to our original context.
        char_pos_in_original_context = 0
        context_position_mapping = []
        for pre_token in pre_tokenized_context:  # Iterate over each pre-token and count the characters it spans
            # find the range of character positions (from original context) that are contained within this pre-token
            char_range = (char_pos_in_original_context, char_pos_in_original_context + len(pre_token))
            context_position_mapping.append((pre_token, char_range))  # put this mapping in our list of mappings
            char_pos_in_original_context += len(pre_token) + 1  # add one to skip over the next whitespace

        # Verify that we haven't lost any tokens and have counted through correctly.
        # This will happen if we have double whitespaces - but we assume we don't have these.
        num_pre_token_chars = sum([len(pt) for pt, rge in context_position_mapping]) + len(pre_tokenized_context) - 1
        num_original_chars = len(short_context)
        if num_pre_token_chars != num_original_chars:
            metrics["non_match_pt"] += 1
            # Number of characters in the pre-tokenized context does not match num in original context
            continue

        # Now we need to scan through our mapping and tokenize each of our pre-tokens
        # We need to find the tokenized versions of the original start and end positions
        input_ids = []
        start_token_position, end_token_position = None, None
        for pre_token, char_range in context_position_mapping:
            # char_range is a tuple of the form (start_position, end_position) where end_position is non-inclusive
            pre_token_start_pos, pre_token_end_pos = char_range
            token_sub_tokens = tokenizer.tokenize(pre_token)  # tokenize the pre-tokenized word
            mapping = None

            # If we have found the token containing the start position, then we need to find the corresponding (sub)tok
            if pre_token_start_pos <= text_start_pos < pre_token_end_pos:  # found the token corresponding to start
                # We receive a mapping of the sub_tokens that we passed to this function
                mapping = sub_tokenize_answer_tokens(tokenizer, pre_token, token_sub_tokens,
                                                     pre_token_start_pos, text_start_pos)
                if mapping is None:
                    metrics["empty_mapping"] += 1
                    continue
                token_sub_tokens = [t[0] for t in mapping]

            if pre_token_start_pos <= text_end_pos <= pre_token_end_pos:  # found the token corresponding to end
                # We receive a mapping of the sub_tokens that we passed to this function
                mapping = sub_tokenize_answer_tokens(tokenizer, pre_token, token_sub_tokens,
                                                     pre_token_start_pos, text_end_pos)

                if mapping is None:
                    metrics["empty_mapping"] += 1
                    continue

                token_sub_tokens = [t[0] for t in mapping]

            if mapping is not None:
                # see if the start position or end position are in here
                for idx, (sub_token, (s, e)) in enumerate(mapping):
                    if start_token_position is None and s == text_start_pos:
                        # we need to find the absolute tokenized position, this is the combination of
                        # all tokens already added to input ids and the sub-tokens that came before.
                        start_token_position = idx + len(input_ids)  # add tokens that came prior
                    if end_token_position is None and e == text_end_pos:
                        end_token_position = idx + len(input_ids) + 1  # add tokens that came prior (and 1 for non-inc)

            token_input_ids = tokenizer.convert_tokens_to_ids(token_sub_tokens)
            input_ids.extend(token_input_ids)  # add the sub-token ids to our input_ids list

        # Combine the characters from our tokens and compare with the answer -> did we build is correctly?
        test_tokenization = tokenizer.convert_ids_to_tokens(input_ids[start_token_position:end_token_position])
        clean_test_tokenization = tokenizer.convert_tokens_to_string(test_tokenization).replace(" ", "").lstrip('#')
        if answer.replace(" ", "") != clean_test_tokenization:  # Ground truth answer does not match joined token answer
            metrics["non_match_gt"] += 1
            continue

        # The attention mask indicates which tokens should be attended to (1) for yes and (0) for no.
        # For instance, padding tokens should not be attended to, but we don't add these until later
        # Hence, attention mask at this stage should just be a list of 1s
        attention_mask = len(input_ids) * [1]

        # Token type ids are used when distinguishing between the context and the question
        # Since we're tokenizing the context, we need to set all values to zero
        token_type_ids = len(input_ids) * [1]

        # At this point, we have the tokenized start and end position of the answer, input ids, attention mask,
        # token_type_ids question etc. We need to combine these components into actual features.
        tokenized_question = tokenizer.tokenize(question)
        question_input_ids = tokenizer.convert_tokens_to_ids(tokenized_question)
        question_attention_mask = len(question_input_ids) * [1]
        question_token_type_ids = len(question_input_ids) * [0]

        num_question_tokens = len(tokenized_question)  # tokenize the question
        num_additional_tokens = 3  # refers to the [CLS] tokens and [SEP] tokens used when combining question & context
        num_context_tokens = max_length - num_question_tokens - num_additional_tokens

        # We need to take "doc strides" of the context paragraph of lengths up to num_context_tokens
        # These strides must centre around the tokenized start and end positions
        # Hence, the stride can begin as late as start_token_position and end as early as end_token_position
        # We also perform padding here and create our features.

        # Note: the question token type ids are 0s, the context token type ids are 1s
        num_answer_tokens = end_token_position - start_token_position
        # Shift the doc stride in intervals of 20 so that we don't create way too many features
        for left_stride in range(0, num_context_tokens - num_answer_tokens, 20):
            right_stride = num_context_tokens - num_answer_tokens - left_stride
            left_clip = max(0, start_token_position - left_stride)
            right_clip = min(end_token_position + right_stride, len(input_ids))

            clipped_input_ids = input_ids[left_clip: right_clip]
            clipped_attention_mask = attention_mask[left_clip: right_clip]
            clipped_token_type_ids = token_type_ids[left_clip: right_clip]

            # concatenate the question, special tokens and context to make features
            # [CLS] and first [SEP] are considered part of the context for token_type_ids.
            # we attend over special tokens like [CLS] and [SEP] - just NOT PADDING

            # concatenate context with question - [CLS] QUESTION [SEP] SHORT_CONTEXT [SEP]
            all_input_ids = [tokenizer.cls_token_id] + question_input_ids + [tokenizer.sep_token_id] + clipped_input_ids + [tokenizer.sep_token_id]
            all_attention_mask = [1] + question_attention_mask + [1] + clipped_attention_mask + [1]
            all_token_type_ids = [0] + question_token_type_ids + [0] + clipped_token_type_ids + [1]

            # pad the end with zeros if we have shorter length
            all_input_ids.extend([tokenizer.pad_token_id] * (max_length - len(all_input_ids)))  # add the padding token
            all_attention_mask.extend([0] * (max_length - len(all_attention_mask)))  # do not attend to padded tokens
            all_token_type_ids.extend([0] * (max_length - len(all_token_type_ids)))  # part of the context

            # Now we're ready to create a feature
            feature = FactoidFeature(example._question_id, all_input_ids,
                                     all_attention_mask, all_token_type_ids, start_token_position,
                                     end_token_position, example._answer)
            feature_list.append(feature)

    print('\n------- COLLATING FEATURE METRICS -------')
    total_examples = len(examples)
    total_features = len(feature_list)
    print("Created {} features from {} examples".format(total_features, total_examples))

    total_examples_skipped = sum([metrics[key] for key in metrics.keys()])
    percentage_non_match_gt = 0 if total_examples_skipped == 0 else round(
        100 * metrics["non_match_gt"] / total_examples_skipped, 2)
    percentage_non_match_pt = 0 if total_examples_skipped == 0 else round(
        100 * metrics["non_match_pt"] / total_examples_skipped, 2)
    percentage_empty_mapping = 0 if total_examples_skipped == 0 else round(
        100 * metrics["empty_mapping"] / total_examples_skipped, 2)

    print('{} examples were skipped in total due to the following errors:'.format(total_examples_skipped))
    print("- {} errors ({}%): Ground truth answer does not match joined token answer"
          .format(metrics["non_match_gt"], percentage_non_match_gt))
    print("- {} errors ({}%): Length of pre-tokenized context does not match length of original context"
          .format(metrics["non_match_pt"], percentage_non_match_pt))
    print("- {} errors ({}%): Map returned by sub-tokenize was empty"
          .format(metrics["empty_mapping"], percentage_empty_mapping))
    return feature_list


# ------------ FINE-TUNING PYTORCH DATASETS ------------
class BatchFeatures:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        device = "cuda" if torch.cuda.is_available() else "cpu"  # device

        self.question_ids = list(transposed_data[0])
        self.input_ids = torch.tensor(transposed_data[1], device=device)
        self.attention_mask = torch.tensor(transposed_data[2], device=device)
        self.token_type_ids = torch.tensor(transposed_data[3], device=device)

        # if we have test features, we might not have answer_text, answer_start, answer_end or is_impossible
        # for squad and most of bioasq test datasets, we have most of these fields. (except answer_start and answer_end)
        # i.e. in the bioasq (non-golden) test dataset, answer_text, answer_start and answer_end is None
        self.answer_text = list(transposed_data[4])

        if len(transposed_data) == 7:
            self.answer_start = torch.tensor(transposed_data[5], device=device)
            self.answer_end = torch.tensor(transposed_data[6], device=device)
        else:
            self.labels = torch.tensor(transposed_data[5], device=device)

def collate_wrapper(batch):
    return BatchFeatures(batch)

class QADataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        selected_example = self.examples[idx]
        return selected_example.get_features()

    def __len__(self):
        return len(self.examples)


# ------------ CUSTOM PYTORCH DATASET IMPLEMENTATIONS ------------
class MappedCSVDataset(Dataset):
    """
    Mapped Pytorch dataset which loads in a sub-dataset from separate pre-training data files.
    """

    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file, delimiter='|', error_bad_lines=False, skiprows=1, warn_bad_lines=False)

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
        self._resume = False

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

                if self._resume:  # skip all the extra processing
                    return

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
                        print("Dataset size: {}".format(self._dataset_size))
                    return None

    def build_iterator_from_csv(self, path_to_csv):
        print("\nReading CSV {}".format(str(path_to_csv)[-11:]))

        csv_dataset = MappedCSVDataset(path_to_csv)
        data_loader = DataLoader(csv_dataset, batch_size=self._batch_size, shuffle=True)
        return iter(data_loader)

    def resume_from_step(self, training_step):
        sys.stderr.write("\nTraining step - {}".format(training_step))
        self._resume = True

        start_time = time.time()
        for _ in range(training_step):
            next(self)

        self._resume = False
        sys.stderr.write("\nTook {} seconds to resume from training step {}".format(round(time.time() - start_time, 2),
                                                                                    training_step))

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
