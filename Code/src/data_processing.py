from random import randint, random
import pandas as pd
import pathlib
import torch
import unidecode
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from functools import partial
import random


class SQuADFeature:
    def __init__(self, question_id, input_ids, attention_mask, token_type_ids, answer_start, answer_end,
                 is_impossible):
        self._question_id = question_id
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self._token_type_ids = token_type_ids
        self._answer_start = answer_start
        self._answer_end = answer_end
        self._is_impossible = is_impossible


def convert_samples_to_features(samples, tokenizer, max_length):

    def verify_tokens_match_answer(tokens, answer):
        # if we have an unknown token, we can't really check

        # remove spaces from answer
        characters_in_tokens = "".join([char for token in tokens for char in token if char != "#"])
        characters_in_answer = answer.replace(" ", "").lower()
        unaccented_characters_in_answer = unidecode.unidecode(characters_in_answer)
        unaccented_characters_in_tokens = unidecode.unidecode(characters_in_tokens)

        return unaccented_characters_in_answer == unaccented_characters_in_tokens

    def correct_for_unaccounted(tokens, answer, start, end):
        # sometimes answers are off by a character or two – fix this
        for step_left in range(-3, 3):

            for step_right in range(-3, 3):

                if verify_tokens_match_answer(tokens[start + step_left:end + step_right], answer):
                    return start + step_left, end + step_right  # When the gold label is off by n characters

        raise Exception("Even after correction, the answer will not align.")

    def tokenized_start_end_pos(tokenizer, text: str, start: int, end: int) -> tuple:
        print("start", start)
        print("end", end)
        """
        The tokenizer removes some characters from text when creating tokens e.g.  ̃
        We need to correct for it being out by 1 or 2 characters
        :param tokenizer:
        :param text:
        :param start:
        :param end:
        :return:
        """

        def handle_unk(raw_word: str, tokenized_word: list, unk_idx: int, char_in_raw: int) -> tuple:
            # given an unk token, we need to find what character's it represents
            last_token_idx = len(tokenized_word) - 1
            tokens_handled = 1

            print("unk_idx", unk_idx)
            print("all tokens in word", tokenized_word)

            for tok_idx, tok_word in enumerate(tokenized_word):
                if tok_idx < unk_idx:
                    continue

                print("tok_word", tok_word)
                print("tok_idx", tok_idx)
                print("char_in_raw", char_in_raw)

                if tok_word == tokenizer.unk_token:
                    # found right unk token (there could be multiple)
                    # can't take it's length at face val
                    if tok_idx == last_token_idx:
                        print('skipping this many ->', len(raw_word[char_in_raw:]))
                        return len(raw_word[char_in_raw:]), tokens_handled

                    if tokenized_word[tok_idx + 1] == tokenizer.unk_token:
                        # defer for this round by moving to next token, but increment tokens_handled
                        tokens_handled += 1
                        continue

                    # need to find the start position of the first char of next token
                    for next_tok_char in tokenized_word[tok_idx + 1]:
                        if next_tok_char != "#":
                            # this is the next char to look for
                            print("looking for {} in {}".format(next_tok_char, raw_word[char_in_raw:]))
                            start_idx_next = raw_word[char_in_raw:].index(next_tok_char)

                            print('skipping', start_idx_next)
                            return start_idx_next, tokens_handled

                else:
                    num_non_hash_chars = len(tok_word) - tok_word.count('#')
                    char_in_raw += num_non_hash_chars

        # iterate through each of the words (i.e. split by whitespace)
        words_by_whitespace = text.split(" ")

        # keep a count of the num of characters looked at
        char_count = 0
        num_tokens = 0
        new_s, new_e = None, None

        for word in words_by_whitespace:
            # tokenize each word individually
            tokens_in_words = tokenizer.tokenize(word)
            word_char_count = 0
            skip_tokens = 0

            for token_idx, token_in_word in enumerate(tokens_in_words):
                if skip_tokens > 0:
                    skip_tokens -= 1
                    continue

                if token_in_word == tokenizer.unk_token:
                    num_chars, tokens_handled = handle_unk(word, tokens_in_words, token_idx, word_char_count)
                    skip_tokens += tokens_handled - 1
                    new_char_count = char_count + num_chars
                    word_char_count += num_chars
                else:
                    # get the num of non-hash characters
                    num_non_hashes = len(token_in_word) - token_in_word.count("#")
                    new_char_count = char_count + num_non_hashes
                    word_char_count += num_non_hashes

                # print("char count {}, n_char count {}, char {}, tok {}".format(char_count, new_char_count, text[char_count], token_in_word))

                if char_count <= start <= new_char_count and new_s is None:
                    new_s = num_tokens
                    print('SETTING START AS {}'.format(num_tokens))

                # if end >= new_char_count:
                # todo handle mid token ends
                if char_count <= end <= new_char_count and new_e is None:
                    new_e = num_tokens
                    print('SETTING END AS {}'.format(num_tokens))

                    # # break token down further until we find the ACTUAL end
                    # sub_tokens = tokenizer.tokenize(token_in_word[token_in_word.count("#"):])
                    # cumulative_char_count = char_count
                    #
                    # for sub_char in sub_tokens:
                    #     nnh = len(token_in_word) - token_in_word.count("#")
                    #     cumulative_char_count += nnh
                    #
                    #     if cumulative_char_count == end:
                    #         pass
                    #     elif cumulative_char_count < end:
                    #         continue
                    #
                    #     pass

                if new_e is not None and new_s is not None:
                    break


                char_count = new_char_count
                num_tokens += 1

            char_count += 1  # account for whitespace between words
        return new_s, new_e + 1  # +1 means we don't cut off too early

    feature_list = []

    for squad_example in samples:
        # squad_example.print_info()

        short_context = squad_example._short_context
        full_context = squad_example._full_context
        question = squad_example._question
        start_pos = squad_example._answer_start
        end_pos = squad_example._answer_end

        # concatenate context with question - [CLS] SHORT_CONTEXT [SEP] QUESTION [SEP]
        # tokenizer(question, short_context)
        tokenized_input = tokenizer(question, short_context)
        tokenized_context = tokenizer.tokenize(short_context)
        input_ids, attention_mask, token_type_ids = tokenized_input["input_ids"], tokenized_input["attention_mask"], tokenized_input["token_type_ids"]

        # start and end now refers to tokens, not characters
        # however, they only include characters in short context.
        new_start, new_end = tokenized_start_end_pos(tokenizer, short_context, start_pos, end_pos)
        print(full_context)

        print("Official Answer: {}, Predicted Answer: {}".format(squad_example._answer, tokenized_context[new_start: new_end]))
        print("Clipped Answer: {}".format(short_context[start_pos: end_pos]))

        if tokenizer.unk_token not in tokenized_context:
            new_start, new_end = correct_for_unaccounted(tokenized_context, squad_example._answer, new_start, new_end)

            m = verify_tokens_match_answer(tokenized_context[new_start: new_end], squad_example._answer)
            if not m:
                raise Exception("Characters in tokens '{}' do not match characters in answer '{}'.".format(tokenized_context[new_start: new_end], squad_example._answer))

            # add on 1 for the newly-appended [CLS] token
        new_start = new_start + 1
        new_end = new_end + 1

        # answer_start, answer_end, is_impossible
        feature = SQuADFeature(squad_example._question_id, input_ids, attention_mask, token_type_ids, new_start,
                               new_end, squad_example._is_impossible)

        feature_list.append(feature)

    return feature_list


# ------------ FINE-TUNING PYTORCH DATASETS ------------
class SquadDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


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
                        print("Dataset size: {}".format(self._dataset_size))
                    return None

    def build_iterator_from_csv(self, path_to_csv):
        print("\nReading CSV {}".format(str(path_to_csv)[-11:]))

        csv_dataset = MappedCSVDataset(path_to_csv)
        data_loader = DataLoader(csv_dataset, batch_size=self._batch_size, shuffle=True)
        return iter(data_loader)

    def resume_from_step(self, training_step):
        for i in range(training_step):
            next(self)
        print("\nResuming training from csv {} ({})\n".format(self._current_csv_idx, str(
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
