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


class SQuADExample:
    def __init__(self, question_id, question, short_context, full_context, answer, answer_start, answer_end,
                 is_impossible):
        self._question_id = question_id
        self._question = question
        self._short_context = short_context
        self._full_context = full_context
        self._answer = answer
        self._answer_start = answer_start
        self._answer_end = answer_end
        self._is_impossible = is_impossible

    def write(self):
        pass

    def print_info(self):
        print("\nSQuAD Example\n--------------")
        print("Question ID:", self._question_id)
        print("Question:", self._question)
        print("Short Context:", self._short_context)
        print("Full Context:", self._full_context)
        print("Answer:", self._answer)
        print("Answer Start:", self._answer_start)
        print("Answer End:", self._answer_end)
        print("Is Impossible:", self._is_impossible)


# ------------ READ DATASETS INTO THEIR CORRECT FORMAT ------------
def read_squad(path_to_file: Path):
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
            full_context = passage['context'].rstrip()      # remove trailing whitespace

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
                #
                # print("start_position_of_sent_in_context {}".format(start_position_of_sent_in_context))
                # print("cumulativee length {}".format(cumulative_length))
                # print("last sentence", last_sentence)
                num_whitespaces_to_add = start_position_of_sent_in_context - (last_sentence[0] + last_sentence[1])

                # print("adding {} whitespaces to {}".format(num_whitespaces_to_add, context_sentences[sent_idx - 1]))
                context_sentences[sent_idx - 1] = context_sentences[sent_idx - 1] + (" " * num_whitespaces_to_add)

                cumulative_length += sentence_length + num_whitespaces_to_add
                last_sentence = (start_position_of_sent_in_context, sentence_length)


            #
            #
            # start_sentences = []
            # cumulative_length = 0
            # for sent in context_sentences:
            #     sentence_length = len(sent)
            #     start_sentences.append((full_context.find(sent, cumulative_length), sentence_length))
            #     cumulative_length += sentence_length
            #
            #
            # # correct the context sentences to include appropriate whitespaces - this protects start and end pos.
            # last_total = (0, 0)
            # for i, (s, l) in enumerate(start_sentences):
            #     print(s, l)
            #     # add whitespaces
            #     print("adding {} whitespaces to {}".format(s - (last_total[0] + last_total[1]), context_sentences[i-1]))
            #     context_sentences[i-1] = context_sentences[i-1] + (" " * (s - (last_total[0] + last_total[1])))
            #     last_total = (s, l)

            context_sent_lengths = [len(sent) for sent in context_sentences]

            if sum(context_sent_lengths) != len(full_context):
                # print('len full context', len(full_context))
                # print('full context', (full_context))
                # print('sum(context_sent_lengths)', sum(context_sent_lengths))
                #
                # print(context_sentences)

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

                    # squad_example = SquadExample(
                    #     qas_id=question_id,
                    #     question_text=question,
                    #     context_text=passage['context'],
                    #     answer_text=answer_text,
                    #     title="test",
                    #     start_position_character=answer['answer_start'],
                    #     is_impossible=is_impossible
                    # )
                    # dataset.append(squad_example)


                    dataset.append(SQuADExample(question_id, question, short_context, full_context, answer_text, normalised_answer_start, normalised_answer_end, is_impossible))

            break   # todo remove
    return dataset


def read_bioasq(path_to_file: Path):
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

    with open(path_to_file, 'rb') as f:
        bioasq_dict = json.load(f)

    articles_file_name = "pubmed_{}".format(str(path_to_file).split('/').pop())
    articles_file_path = Path(path_to_file / '../{}'.format(articles_file_name)).resolve()

    with open(articles_file_path, 'rb') as f:
        articles_dict = json.load(f)

    def match_answer_to_passage(answer: str, passage: str) -> list:
        # remove punctuation

        answer = answer.lower()
        passage = passage.lower()

        def remove_punctuation(text: str) -> tuple:
            position_list = []

            for char_position, char in enumerate(text):
                # this character needs to be removed if in punctuation, but we should preserve its position
                if char not in string.punctuation:
                    position_list.append(char_position)

            return "".join([c for idx, c in enumerate(text) if idx in position_list]), position_list

        p_answer, answer_positions = remove_punctuation(answer)
        p_passage, passage_positions = remove_punctuation(passage)

        if p_answer in p_passage:
            match_cleaned_starts = [m.start() for m in re.finditer(p_answer, p_passage)]
            match_raw_starts = [passage_positions[m] for m in match_cleaned_starts]
            match_raw_ends = [passage_positions[m + len(p_answer) - 1] for m in match_cleaned_starts]

            return [[start, end] for start, end in zip(match_raw_starts, match_raw_ends)]

    def process_factoid(data):
        question_id = data["id"]
        question = data["body"]
        answer = data["exact_answer"]
        snippets = data["snippets"]

        for snippet in snippets:
            article_id = snippet["document"].split('/').pop()
            section = snippet["beginSection"] if snippet["beginSection"] != "sections.0" else "abstract"

            if snippet["beginSection"] != snippet["endSection"]:
                raise Exception('{} is not {}'.format(snippet["beginSection"], snippet["endSection"]))

            try:
                article = articles_dict[article_id]
                # if the corresponding article is not here - we need to skip it.
                # todo is it ok that some articles aren't here? probably not - let's try and find them.
            except KeyError:
                continue

            paragraph = article[section]
            # print("\nquestion:", question)
            # print("paragraph:", paragraph)

            # parsed_answer = [remove_stopwords_and_stem(sub_answer) for sub_answer in answer]

            matches = match_answer_to_passage("".join(answer), snippet['text'])

            # if matches is not None:
            #     print([snippet['text'][pair[0]:pair[1]+1] for pair in matches])

            beginOffset, endOffset = int(snippet["offsetInBeginSection"]), int(snippet["offsetInEndSection"])
            # print('clipped section:', paragraph[beginOffset:endOffset])
            # print('snippet text:', snippet['text'])
            # print('exact answer:', answer)
            # print('type', data["type"])
            # print("begin", snippet["beginSection"])
            # print("end", snippet["endSection"])






        sub_dataset = []


        for snippet in snippets:
            context = snippet["text"]
            sub_dataset.append({"id": question_id,
                                "context": context,
                                "question": question,
                                "answer": answer})

        print(sub_dataset)
        return sub_dataset

    def process_list():
        pass

    def process_yesno():
        pass

    fc_map = {
        "factoid": process_factoid,
        "yesno": process_yesno,
        # "list": process_list,
    }

    dataset = []

    for data_point in bioasq_dict['questions']:
        question_type = data_point["type"]

        # todo remove
        if question_type in ['list', 'yesno', 'summary']:
            continue


        try:
            fc = fc_map[question_type]
        except KeyError:
            if question_type == "summary":  # summary questions not required
                continue
            raise KeyError("Question type {} is not in fc_map".format_map(question_type))

        context_answer_pairs = fc(data_point)
        dataset.extend(context_answer_pairs)


        # for snippet in snippets:
        #     print(snippet)
        #     context = snippet["text"]
        #     print(data_point["exact_answer"])
        #     # answer = fc()
        #     # dataset.append({"context": context, "question": question, "answer": answer})
        #
        # # for snippet in data_point["snippets"]:
        # #     context =
        #
        # context = passage['context']
        # for qa in passage['qas']:
        #     question = qa['question']
        #     for answer in qa['answers']:
        #         add_end_idx(answer, context)
        #         dataset.append({"context": context, "question": question, "answer": answer})
        #
        #
        # print("body", data_point["body"])
        # print("documents", data_point["documents"])
        # print("ideal_answer", data_point["ideal_answer"])
        # print("concepts", data_point["concepts"])
        # print("type", data_point["type"])
        # print("id", data_point["id"])


    # snippets = question["snippets"]
    # question_type = question["type"]
    # question_answer = question["type"]

    # print(bioasq_dict)
    # print("Keys", bioasq_dict.keys())

    # print(questions)




def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] + 1)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


dataset_to_fc = {
    "squad": read_squad,
    "bioasq": read_bioasq
}


if __name__ == "__main__":
    # todo delete this section after testing
    base_path = Path(__file__).parent
    squad_dir = (base_path / '../datasets/squad/dev-v2.0.json').resolve()
    data = read_squad(squad_dir)
    print("length of data", len(data))

    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # train_contexts = [d["context"] for d in data]
    # train_questions = [d["question"] for d in data]
    # train_answers = [d["answer"] for d in data]
    #
    # train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    # add_token_positions(train_encodings, train_answers)
    # # print(train_encodings)

    # base_path = Path(__file__).parent
    # data_dir = (base_path / '../datasets/bioasq/training8b.json').resolve()
    # data = read_bioasq(data_dir)
    # print("length of data", len(data))

    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # train_contexts = [d["context"] for d in data]
    # train_questions = [d["question"] for d in data]
    # train_answers = [d["answer"] for d in data]
    #
    # train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    # add_token_positions(train_encodings, train_answers)
    # print(train_encodings)



