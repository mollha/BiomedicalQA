import json
from pathlib import Path
import re

import unidecode
from transformers import DistilBertTokenizerFast
from spacy.lang.en import English
import spacy
from nltk.tokenize import sent_tokenize
from transformers.data.processors.squad import SquadExample

import string
from tqdm import tqdm

from spacy.matcher import PhraseMatcher, Matcher
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')


def update_dataset_metrics(total_metrics, additional_metrics):
    for key in additional_metrics:
        if key in total_metrics:
            total_metrics[key] += additional_metrics[key]
        else:
            total_metrics[key] = additional_metrics[key]
    return total_metrics


class BinaryExample:
    def __init__(self, question_id, question, short_context, answer):
        self._question_id = question_id
        self._question = question
        self._short_context = short_context
        self._answer = answer

    def print_info(self):
        print("\n----- Yes/No (binary) Example -----\n")
        print("Question ID:", self._question_id)
        print("Question:", self._question)
        print("Short Context:", self._short_context)
        print("Answer:", self._answer)


class FactoidExample:
    def __init__(self, question_id, question_type, question, short_context, full_context, answer, answer_start, answer_end,
                 is_impossible):
        self._question_id = question_id
        self._question_type = question_type
        self._question = question
        self._short_context = short_context  # the sentence(s) containing the answer
        self._full_context = full_context  # the remaining context containing all sentences provided
        self._answer = answer
        self._answer_start = answer_start
        self._answer_end = answer_end
        self._is_impossible = is_impossible

    def print_info(self):
        print("\n----- Factoid (squad) Example -----\n")
        print("Question ID:", self._question_id)
        print("Question:", self._question)
        print("Short Context:", self._short_context)
        print("Full Context:", self._full_context)
        print("Answer:", self._answer)
        print("Answer Start:", self._answer_start)
        print("Answer End:", self._answer_end)
        print("Is Impossible:", self._is_impossible)


def pre_tokenize(short_context, start_position, end_position):
    # In the BioELECTRA tokenizer that we trained specifically on PubMed vocabulary, the pre-tokenisation steps
    # include removing whitespace, stripping accents and converting to lowercase
    short_context = short_context.lower()
    accent_stripped_context = unidecode.unidecode(short_context)
    start_pos, end_pos = None, None

    # We need to check that removing accents has not affected our start and end answer positions.
    # we need to ensure that the start and end positions will remain valid.
    char_position_in_as_context = 0
    char_position_in_sc_context = 0

    while char_position_in_sc_context <= len(short_context):
        if char_position_in_sc_context == start_position:
            start_pos = char_position_in_as_context  # adjust start position to start position - accents

        if char_position_in_sc_context == end_position:
            end_pos = char_position_in_as_context
            break

        sc_char = short_context[char_position_in_sc_context]
        char_position_in_sc_context += len(sc_char)   # this is > 1 for weird characters
        char_position_in_as_context += len(unidecode.unidecode(sc_char)) - len(sc_char) + 1

    if start_pos is None or end_pos is None:
        raise Exception("Either start position '{}' or end position '{}' is None. This is not allowed."
                        .format(start_pos, end_pos))

    return accent_stripped_context, start_pos, end_pos


# ------------ READ DATASETS INTO THEIR CORRECT FORMAT ------------
def read_squad(paths_to_files: list, testing=False):
    """
    Read the squad data into three categories of contexts, questions and answers

    This function is adapted from the Huggingface squad tutorial at:
    https://huggingface.co/transformers/custom_datasets.html#qa-squad
    :param path_to_file: path to file containing squad data
    :return: a list containing a dictionary for each context, question and answer triple.
    """

    def add_end_idx(answer, context):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        #
        # print('clip con', context[start_idx:end_idx])
        # print('gold_text', gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters

    # Note: the only reason paths_to_files is a list is for consistency as this is needed for reading bioasq.
    path = Path(paths_to_files.pop())
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    dataset = []
    metrics = {
        "impossible_examples": 0,
        "impossible_questions": 0,
        "num_questions": 0,
        "num_examples": 0,
        "num_skipped_examples": 0,
    }

    # todo verify that the start and end positions actually point to the answer
    # todo also make sure that short_context contains the answer if question !impossible
    # todo check in the reading dataset part that there is no leading and trailing whitespace

    for group in tqdm(squad_dict['data'], desc="SQuAD Data \u2b62 Examples"):
        for passage in group['paragraphs']:
            full_context = passage['context'].rstrip()  # remove trailing whitespace

            # remove leading whitespace if it exists and adjust start and end positions
            num_leading_whitespaces = len(full_context) - len(full_context.lstrip())
            full_context = full_context.lstrip()

            for qa in passage['qas']:
                question = qa['question']
                question_id = qa['id']
                is_impossible = qa['is_impossible']
                metrics["impossible_questions"] += 1 if is_impossible else 0
                metrics["num_questions"] += 1

                if is_impossible:
                    # still need to pre-tokenized the text
                    pre_processed_context = unidecode.unidecode(full_context.lower())

                    # impossible questions have an empty list for "answers"
                    # the answer is None, and its start and end positions are negative 1
                    dataset.append(FactoidExample(question_id, "factoid", question, pre_processed_context, pre_processed_context,
                                                  None, -1, -1, is_impossible))
                    continue

                for answer in qa['answers']:
                    answer['answer_start'] = answer['answer_start'] - num_leading_whitespaces  # adjust by leading ws
                    add_end_idx(answer, full_context)  # set the end index - this will also be adjusted by leading ws

                    # When the answer contains special characters, if we pre-tokenize then we will lose some characters
                    answer['text'] = unidecode.unidecode(answer['text'].lower())  # normalize the answer

                    # pre-tokenize and correct answer start and end if necessary
                    pre_processed_context, answer_start, answer_end = pre_tokenize('{}'.format(full_context), answer['answer_start'],
                                                                                   answer['answer_end'])

                    # Correct differences relating to special characters in the answer
                    # Length of the answer in original context
                    length_original = len('{}'.format(full_context[answer['answer_start']:answer['answer_end']]))

                    if length_original - (answer_end - answer_start) > 0:
                        answer_end += length_original - (answer_end - answer_start)

                    if answer['text'] != pre_processed_context[answer_start:answer_end]:
                        # print('Answer "{}" from example does not match the spliced context "{}"'
                        #       .format(answer['text'], pre_processed_context[answer_start:answer_end]))
                        metrics["num_skipped_examples"] += 1
                        continue

                    metrics["num_examples"] += 1
                    metrics["impossible_examples"] += 1 if is_impossible else 0

                    # todo should we have a short context, or not?
                    dataset.append(FactoidExample(question_id, "factoid", question, pre_processed_context, pre_processed_context, answer['text'],
                                                  answer_start, answer_end, is_impossible))
        break  # todo remove later

    # ------ DISPLAY METRICS -------
    total_questions = metrics["num_questions"]
    total_examples = metrics["num_examples"]

    print('\n------- COLLATING {}-SET METRICS -------'.format("TEST" if testing else "TRAIN"))
    print("There are {} questions and {} examples".format(total_questions, total_examples))

    percentage_impossible_examples = 0 if total_examples == 0 else round(
        100 * metrics["impossible_examples"] / total_examples, 2)

    percentage_impossible_questions = 0 if total_questions == 0 else round(
        100 * metrics["impossible_questions"] / total_questions, 2)

    print("\n- Impossible Instances: {} questions ({}%), {} examples ({}%)"
          .format(metrics["impossible_questions"], percentage_impossible_questions, metrics["impossible_examples"],
                  percentage_impossible_examples))

    print("- Non-Impossible Instances: {} questions ({}%), {} examples ({}%)"
          .format(total_questions - metrics["impossible_questions"], 100.0 - percentage_impossible_questions,
                  total_examples - metrics["impossible_examples"], 100.0 - percentage_impossible_examples))

    print('-', metrics["num_skipped_examples"], 'examples were skipped.\n')
    return {"factoid": dataset}


def read_bioasq(paths_to_files: list, testing=False):
    """
    Read the bioasq data into three categories of contexts, questions and answers.
    BioASQ contexts are different than SQuAD, as they are a combination of articles and snippets.
    :param path_to_file: path to file containing squad data
    :return: a list containing a dictionary for each context, question and answer triple.
    """

    # todo why don't we just train list and factoid questions together - it makes more sense???
    # we will have a better model this way since they both predict in the same way.

    dataset = {
        "list": [],
        "factoid": [],
        "yesno": []
    }

    # todo - read and combine multiple dictionaries here to allow reading multiple dataset files.
    bioasq_dict = {"questions": []}
    for path_to_file in paths_to_files:  # iterate over the different bioasq
        with open(path_to_file, 'rb') as f:
            bioasq_sub_dict = json.load(f)
            bioasq_dict["questions"].extend(bioasq_sub_dict["questions"])

    def match_answer_to_passage(answer: str, passage: str) -> list:
        #
        # print("\nanswer", answer)
        # print("passage", passage)
        # remove punctuation

        answer = answer.lower()
        passage = passage.lower()

        def remove_punctuation(text: str) -> tuple:
            position_list = []
            escape_words = ["&apos;"]

            spacy_doc = nlp(text)
            spacy_tokens = [token.text for token in spacy_doc if token.tag_ == 'POS']

            escape_words.extend(spacy_tokens)  # also don't care about POS
            escape_range = [range(m.start(), m.end()) for word in escape_words for m in re.finditer(word, text)]

            for char_position, char in enumerate(text):
                # this character needs to be removed if in punctuation, but we should preserve its position
                in_escape_range = False
                for ra in escape_range:
                    if char_position in ra:
                        in_escape_range = True
                        break

                if char not in string.punctuation and not in_escape_range:
                    position_list.append(char_position)

            return "".join([c for idx, c in enumerate(text) if idx in position_list]), position_list

        p_answer, answer_positions = remove_punctuation(answer)
        p_passage, passage_positions = remove_punctuation(passage)

        if p_answer in p_passage:  # if the answer appears in the passage at least once
            match_cleaned_starts = [m.start() for m in re.finditer(p_answer, p_passage)]
            match_raw_starts = [passage_positions[m] for m in match_cleaned_starts]
            match_raw_ends = [passage_positions[m + len(p_answer) - 1] for m in match_cleaned_starts]
            # todo check if we should return end instead of end + 1
            return [[start, end + 1] for start, end in zip(match_raw_starts, match_raw_ends)]

        return []  # answer does not appear in passage

    def process_factoid_or_list_question(data):
        question_id = data["id"]
        question = data["body"]
        q_type = data["type"]
        answer_list = data["exact_answer"]
        snippets = data["snippets"]
        examples_from_question = []

        flattened_answer_list = []
        for answer in answer_list:  # flatten list of lists
            if type(answer) == list:
                flattened_answer_list.extend([sub_answer for sub_answer in answer])
            else:
                flattened_answer_list.append(answer)

        metrics = {  # this dictionary is combined with metrics from previous questions
            "impossible_examples": 0,
            "num_questions": 1,
            "num_examples": 0,
            "num_skipped_examples": 0
        }

        for snippet in snippets:
            context = snippet['text']
            # As we match to the context paragraph ourselves, we can remove whitespace without affecting start/end pos'
            context = context.strip()

            for answer in flattened_answer_list:  # for each of our candidate answers
                answer = unidecode.unidecode(answer.lower())  # normalize the answer
                matches = match_answer_to_passage("".join(answer), context)

                if len(matches) == 0:  # there are no matches in the passage and the question is impossible
                    # still need to pre-tokenized the text
                    pre_processed_context = unidecode.unidecode(context.lower())
                    metrics["impossible_examples"] += 1
                    metrics["num_examples"] += 1
                    # the answer is None, and its start and end positions are negative 1
                    examples_from_question.append(FactoidExample(question_id, q_type, question, pre_processed_context, pre_processed_context,
                                                  None, -1, -1, True))
                    # examples_from_question.append(FactoidExample(question_id, q_type, question, context,
                    #                                              context if article is None else article[section], "",
                    #                                              start_pos, end_pos, is_impossible))
                    continue

                is_impossible = False
                # we have at least one match, iterate through each match
                # and create an example for each get the matching text
                for start_pos, end_pos in matches:
                    # pre-tokenize and correct answer start and end if necessary
                    pre_processed_context, answer_start, answer_end = pre_tokenize(context, start_pos, end_pos)

                    metrics["num_examples"] += 1

                    # We need to verify that the start and end positions actually point to the answer before we
                    # expect our feature creation code to find the correctly tokenized positions.
                    matching_text = pre_processed_context[answer_start:answer_end]
                    if answer != matching_text:
                        # print('Answer "{}" from example does not match the spliced context "{}"'
                        #       .format(answer, matching_text))
                        metrics["num_skipped_examples"] += 1
                        continue

                    # In the case where we match answers to the context, we know that context contains the answer
                    # if the question is not impossible. We don't need to check this like in SQuAD.

                    # Create an example for every match
                    examples_from_question.append(
                        FactoidExample(question_id, q_type, question, pre_processed_context, pre_processed_context,
                                       matching_text, start_pos, end_pos, is_impossible)
                    )
        return examples_from_question, metrics

    def process_yesno_question(data):
        question_id = data["id"]
        question = data["body"]
        snippets = data["snippets"]
        answer = data["exact_answer"].lower()

        metrics = {
            "num_examples": len(snippets),
            "num_questions": 1,
            "num_yes_questions": 1 if answer == "yes" else 0,
            "num_yes_examples": len(snippets) if answer == "yes" else 0,
            "num_no_questions": 1 if answer == "no" else 0,
            "num_no_examples": len(snippets) if answer == "no" else 0,
        }

        examples_from_question = []
        for snippet in snippets:
            snippet_text = snippet['text']
            example = BinaryExample(question_id=question_id,
                                    question=question,
                                    short_context=snippet_text,
                                    answer=answer)
            examples_from_question.append(example)
        return examples_from_question, metrics

    fc_map = {
        "factoid": process_factoid_or_list_question,
        "yesno": process_yesno_question,
        "list": process_factoid_or_list_question,
    }

    combined_metrics = {q: {} for q in dataset.keys()}

    for data_point in tqdm(bioasq_dict['questions'], desc="BioASQ Data \u2b62 Examples"):
        question_type = data_point["type"]
        # todo remove all except summary from here - we only exclude the ones we can't handle for now.
        # we don't care about summary questions
        if question_type in ['summary', 'list']:
            continue

        try:
            fc = fc_map[question_type]
        except KeyError:
            if question_type == "summary":  # summary questions not required
                continue
            raise KeyError("Question type {} is not in fc_map".format_map(question_type))

        example_list, question_metrics = fc(data_point)  # apply the right function for the question type
        combined_metrics[question_type] = update_dataset_metrics(combined_metrics[question_type], question_metrics)
        dataset[question_type].extend(example_list)  # collate examples

    # ------ DISPLAY METRICS -------
    total_questions = sum([combined_metrics[qt]["num_questions"] for qt in combined_metrics.keys() if len(combined_metrics[qt]) > 0])
    total_examples = sum([combined_metrics[qt]["num_examples"] for qt in combined_metrics.keys() if len(combined_metrics[qt]) > 0])
    print('\n------- COLLATING {}-SET METRICS -------'.format("TEST" if testing else "TRAIN"))
    print("Across all question types, there are {} questions and {} examples".format(total_questions, total_examples))

    for qt in combined_metrics.keys():  # iterate over each question type
        print('\n------- {} METRICS -------'.format(qt.upper()))
        qt_metrics = combined_metrics[qt]  # get the metrics for that question type
        print(qt_metrics)
        if len(qt_metrics) == 0:
            print("No metrics available for question type '{}'".format(qt))
            continue

        # compute this for all metric types
        num_examples = qt_metrics["num_examples"]
        num_questions = qt_metrics["num_questions"]
        percentage_num_examples = 0 if total_examples == 0 else round(100 * num_examples / total_examples, 2)
        percentage_num_questions = 0 if total_questions == 0 else round(100 * num_questions / total_questions, 2)

        print("Created {} examples from {} questions".format(num_examples, num_questions))
        print("{}-type questions account for {}% of total questions and {}% of total examples"
              .format(qt, percentage_num_questions, percentage_num_examples))

        if qt == "yesno":
            percentage_yes_questions = 0 if num_questions == 0 else round(100 * qt_metrics["num_yes_questions"] / num_questions, 2)
            percentage_yes_examples = 0 if num_examples == 0 else round(100 * qt_metrics["num_yes_examples"] / num_examples, 2)
            percentage_no_questions = 0 if num_questions == 0 else round(100 * qt_metrics["num_no_questions"] / num_questions, 2)
            percentage_no_examples = 0 if num_examples == 0 else round(100 * qt_metrics["num_no_examples"] / num_examples, 2)

            print("\n- Positive Instances: {} questions ({}%), {} examples ({}%)"
                  .format(qt_metrics["num_yes_questions"], percentage_yes_questions, qt_metrics["num_yes_examples"],
                          percentage_yes_examples))
            print("- Negative Instances: {} questions ({}%), {} examples ({}%)"
                  .format(qt_metrics["num_no_questions"], percentage_no_questions, qt_metrics["num_no_examples"],
                          percentage_no_examples))

        elif qt == "factoid":
            percentage_impossible_examples = 0 if num_examples == 0 else round(100 * qt_metrics["impossible_examples"] / num_examples, 2)

            print("\n- Impossible Examples: {} examples ({}%)"
                  .format(qt_metrics["impossible_examples"], percentage_impossible_examples))
            print("- Non-Impossible Examples: {} examples ({}%)"
                  .format(num_examples - qt_metrics["impossible_examples"], 100 - percentage_impossible_examples))

            print('-', qt_metrics["num_skipped_examples"], 'examples were skipped.\n')
    return dataset


dataset_to_fc = {
    "squad": read_squad,
    "bioasq": read_bioasq
}
