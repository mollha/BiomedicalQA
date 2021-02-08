import json
from pathlib import Path
import re
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


class BinaryExample:
    def __init__(self, question_id, question, short_context, answer):
        self._question_type = "yesno"
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
    def __init__(self, question_id, question, short_context, full_context, answer, answer_start, answer_end,
                 is_impossible):
        self._question_type = "factoid"
        self._question_id = question_id
        self._question = question
        self._short_context = short_context
        self._full_context = full_context
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


# ------------ READ DATASETS INTO THEIR CORRECT FORMAT ------------
def read_squad(path_to_file: Path, testing=False):
    """
    Read the squad data into three categories of contexts, questions and answers

    This function is adapted from the Huggingface squad tutorial at:
    https://huggingface.co/transformers/custom_datasets.html#qa-squad
    :param path_to_file: path to file containing squad data
    :return: a list containing a dictionary for each context, question and answer triple.
    """

    def find_sentence(token_pos: int, sentence_lengths: list) -> int:
        """
        This function assumes that the whole token is in a single sentence.
        :param token_pos: int
        :param sentence_lengths: list of sentences lengths (e.g. [166, 207, 195, 171])
        :return: int
        """
        if token_pos > sum(sentence_lengths) - 1:
            raise Exception('Token position provided cannot exist in any of these sentences (too large)')

        for idx, length in enumerate(sentence_lengths):
            if token_pos < length:
                return idx
            token_pos -= length

    def add_end_idx(answer, context):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters

    path = Path(path_to_file)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    dataset = []

    for group in tqdm(squad_dict['data']):
        for passage in group['paragraphs']:
            full_context = passage['context'].rstrip()  # remove trailing whitespace

            context_sentences = [sent for sent in sent_tokenize(full_context)]

            # combine sentences when they are suspiciously short
            # the nltk sentence tokenizer is not very advanced - it interprets almost
            # every full-stop as the end of a sentence which sometimes cuts an answer in half
            #
            # combined_context_sentences = []
            # for context_sentences_idx in range(len(context_sentences)):
            #     if context_sentences_idx > 0:
            #         if len(context_sentences[context_sentences_idx]) < 10:
            #             # combine with previous sentence
            #             context_sentences[context_sentences_idx]

            # context_sentences = [sent.string.strip() for sent in nlp(full_context).sents]
            # start_sentences = [(full_context.find(sent), len(sent)) for sent in context_sentences]

            last_sentence = (0, 0)
            cumulative_length = 0
            for sent_idx in range(len(context_sentences)):
                sent = context_sentences[sent_idx]
                sentence_length = len(sent)

                # skip first sentence
                if sent_idx == 0:
                    cumulative_length += sentence_length
                    last_sentence = (0, sentence_length)
                    continue

                start_position_of_sent_in_context = full_context.find(sent, cumulative_length)
                num_whitespaces_to_add = start_position_of_sent_in_context - (last_sentence[0] + last_sentence[1])
                context_sentences[sent_idx - 1] = context_sentences[sent_idx - 1] + (" " * num_whitespaces_to_add)

                cumulative_length += sentence_length + num_whitespaces_to_add
                last_sentence = (start_position_of_sent_in_context, sentence_length)

            context_sent_lengths = [len(sent) for sent in context_sentences]

            if sum(context_sent_lengths) != len(full_context):
                raise Exception('length no match')

            for qa in passage['qas']:
                question = qa['question']
                question_id = qa['id']
                is_impossible = qa['is_impossible']

                for answer in qa['answers']:
                    add_end_idx(answer, full_context)
                    answer_text = answer['text']

                    sentence_number = find_sentence(answer['answer_start'], context_sent_lengths)
                    normalised_answer_start = answer['answer_start'] - sum(context_sent_lengths[:sentence_number])
                    normalised_answer_end = normalised_answer_start + answer['answer_end'] - answer['answer_start']

                    short_context = context_sentences[sentence_number]
                    dataset.append(FactoidExample(question_id, question, short_context, full_context, answer_text,
                                                normalised_answer_start, normalised_answer_end, is_impossible))

            break  # todo remove
    return dataset


def update_dataset_metrics(total_metrics, additional_metrics):
    for key in additional_metrics:
        if key in total_metrics:
            total_metrics[key] += additional_metrics[key]
        else:
            total_metrics[key] = additional_metrics[key]
    return total_metrics


def read_bioasq(path_to_file: Path, testing=False):
    """
    Read the bioasq data into three categories of contexts, questions and answers.
    BioASQ contexts are different than SQuAD, as they are a combination of articles and snippets.

    Need to create question-passage pairs
    {
        "id": ,
        "question": ,
        "context": ,
        "answer": ,
    }

    :param path_to_file: path to file containing squad data
    :return: a list containing a dictionary for each context, question and answer triple.
    """

    dataset = {
        "list": [],
        "factoid": [],
        "yesno": []
    }

    with open(path_to_file, 'rb') as f:
        bioasq_dict = json.load(f)

    articles_file_name = "pubmed_{}".format(str(path_to_file).split('/').pop())
    articles_file_path = Path(path_to_file / '../../processed_data/{}'.format(articles_file_name)).resolve()

    with open(articles_file_path, 'rb') as f:
        articles_dict = json.load(f)

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

    def process_factoid_question(data):
        question_id = data["id"]
        question = data["body"]
        answer = data["exact_answer"]
        snippets = data["snippets"]
        examples_from_question = []

        metrics = {  # this dictionary is combined with metrics from previous questions
            "impossible_examples": 0,
            "num_questions": 1,
            "num_examples": 0,
        }

        for snippet in snippets:
            context = snippet['text']
            article_id = snippet["document"].split('/').pop()
            section = snippet["beginSection"] if snippet["beginSection"] != "sections.0" else "abstract"

            if snippet["beginSection"] != snippet["endSection"]:
                raise Exception('{} is not {}'.format(snippet["beginSection"], snippet["endSection"]))

            try:
                article = articles_dict[article_id]
            except KeyError:
                article = None
                # the article did not exist

            # print(context)
            # print(answer)

            if testing:
                # create an example per match
                metrics["num_examples"] += 1
                examples_from_question.append(
                    FactoidExample(question_id, question, context, context if article is None else article[section],
                                   answer, None, None, None)
                )
            else:
                matches = match_answer_to_passage("".join(answer), context)

                if len(matches) == 0:  # there are no matches in the passage at all.
                    is_impossible = True
                    start_pos, end_pos = -1, -1
                    metrics["impossible_examples"] += 1
                    metrics["num_examples"] += 1
                    examples_from_question.append(FactoidExample(question_id, question, context,
                                                                 context if article is None else article[section], "",
                                                                 start_pos, end_pos, is_impossible))
                else:
                    is_impossible = False

                    # we have at least one match, iterate through each match
                    # and create an example for each get the matching text
                    for start_pos, end_pos in matches:
                        matching_text = context[start_pos:end_pos]
                        metrics["num_examples"] += 1

                        # create an example per match
                        examples_from_question.append(
                            FactoidExample(question_id, question, context, context if article is None else article[section],
                                           matching_text, start_pos, end_pos, is_impossible)
                        )

        return examples_from_question, metrics

    def process_list_question(data):
        print(data)
        question_id = data["id"]
        question = data["body"]
        snippets = data["snippets"]
        answer = data["exact_answer"]

        pass

    def process_yesno_question(data):
        question_id = data["id"]
        question = data["body"]
        snippets = data["snippets"]
        answer = data["exact_answer"].lower()

        metrics = {
            "num_examples": 0,
            "num_questions": 0,
            "num_yes_questions": 0,
            "num_yes_examples": 0,
            "num_no_questions": 0,
            "num_no_examples": 0,
        }

        if answer == "yes":
            metrics["num_yes_questions"] += 1
            metrics["num_yes_examples"] += len(snippets)
        elif answer == "no":
            metrics["num_no_questions"] += 1
            metrics["num_no_examples"] += len(snippets)

        examples_from_question = []
        for snippet in snippets:
            snippet_text = snippet['text']
            metrics["num_examples"] += 1
            example = BinaryExample(question_id=question_id,
                                    question=question,
                                    short_context=snippet_text,
                                    answer=answer)
            examples_from_question.append(example)
        return examples_from_question, metrics

    fc_map = {
        "factoid": process_factoid_question,
        "yesno": process_yesno_question,
        "list": process_list_question,
    }

    combined_metrics = {q: {} for q in dataset.keys()}

    for data_point in bioasq_dict['questions']:
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
    for qt in combined_metrics.keys():  # iterate over each question type
        print('\n------- {} METRICS -------'.format(qt.upper()))
        qt_metrics = combined_metrics[qt]  # get the metrics for that question type
        if len(qt_metrics) == 0:
            print("No metrics available for question type '{}'".format(qt))

        # compute this for all metric types
        num_examples = qt_metrics["num_examples"]
        num_questions = qt_metrics["num_questions"]
        print("Created {} examples from {} questions".format(num_examples, num_questions))

        if qt == "yesno":
            percentage_yes_questions = 0 if num_questions == 0 else round(100 * qt_metrics["num_yes_questions"] / num_questions, 2)
            percentage_yes_examples = 0 if num_examples == 0 else round(100 * qt_metrics["num_yes_examples"] / num_examples, 2)
            percentage_no_questions = 0 if num_questions == 0 else round(100 * qt_metrics["num_no_questions"] / num_questions, 2)
            percentage_no_examples = 0 if num_examples == 0 else round(100 * qt_metrics["num_no_examples"] / num_examples, 2)

            print("- Positive Instances: {} questions ({}%), {} examples ({}%)"
                  .format(qt_metrics["num_yes_questions"], percentage_yes_questions, qt_metrics["num_yes_examples"],
                          percentage_yes_examples))
            print("- Negative Instances: {} questions ({}%), {} examples ({}%)"
                  .format(qt_metrics["num_no_questions"], percentage_no_questions, qt_metrics["num_no_examples"],
                          percentage_no_examples))

        elif qt == "factoid":
            percentage_impossible_examples = 0 if num_examples == 0 else round(100 * qt_metrics["impossible_examples"] / num_examples, 2)

            print("- Impossible Examples: {} examples ({}%)"
                  .format(qt_metrics["impossible_examples"], percentage_impossible_examples))
            print("- Non-Impossible Examples: {} examples ({}%)"
                  .format(num_examples - qt_metrics["impossible_examples"], 100 - percentage_impossible_examples))



    return dataset, combined_metrics


dataset_to_fc = {
    "squad": read_squad,
    "bioasq": read_bioasq
}
