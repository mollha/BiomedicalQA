import ast
import re
import string
from pathlib import Path
import unidecode
from torch.utils.data import DataLoader
from BiomedicalQA.Code.src.build_checkpoints import build_finetuned_from_checkpoint
from BiomedicalQA.Code.src.data_processing import convert_examples_to_features, QADataset, collate_wrapper
from BiomedicalQA.Code.src.evaluation import evaluate_factoid, evaluate_list, evaluate_yesno
from BiomedicalQA.Code.src.models import get_model_config
import torch
from torch import cuda
import spacy
from BiomedicalQA.Code.src.read_data import BinaryExample, FactoidExample, pre_tokenize

nlp = spacy.load('en_core_web_sm')


def make_example(question, context, qtype, testing=True):
    total_examples = []
    data_point = {
        "id": "test_question",
        "body": question,
        "snippets": [{"text": context}],
        "type": qtype,
    }

    def match_answer_to_passage(answer: str, passage: str) -> list:
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
            return [[start, end + 1] for start, end in zip(match_raw_starts, match_raw_ends)]

        return []  # answer does not appear in passage

    def process_factoid_or_list_question(data):
        question_id = "test_question"
        question = data["body"]
        q_type = data["type"]
        answer_list = None if "exact_answer" not in data else data["exact_answer"]
        snippets = data["snippets"]
        examples_from_question = []

        if type(snippets) == str:  # sometimes, snippets list a string representation of a list
            snippets = ast.literal_eval(snippets)  # convert to list from string

        flattened_answer_list = []
        if answer_list is not None:
            for answer in answer_list:  # flatten list of lists
                if type(answer) == list:
                    flattened_answer_list.extend([sub_answer for sub_answer in answer])
                else:
                    flattened_answer_list.append(answer)

        if len(flattened_answer_list) == 0:
            flattened_answer_list.append(None)

        for snippet in snippets:
            context = snippet['text']
            # As we match to the context paragraph ourselves, we can remove whitespace without affecting start/end pos'
            context = context.strip()

            for answer in flattened_answer_list:  # for each of our candidate answers
                answer = None if answer is None else unidecode.unidecode(answer.lower())  # normalize the answer
                # if we're testing, we don't care about start and end positions
                if testing:
                    context = context.lower()
                    accent_stripped_context = unidecode.unidecode(context)

                    # Create an example for every match
                    examples_from_question.append(
                        FactoidExample(question_id, q_type, question, accent_stripped_context,
                                       answer, -1, -1)
                    )
                else:  # we're training, so we need to find where the answer matches in the passage.
                    matches = match_answer_to_passage("".join(answer), context)

                    if len(matches) == 0:  # there are no matches in the passage and the question is impossible
                        # check to see if we have a configured start_pos and end_pos
                        if 'start_pos' in snippet and 'end_pos' in snippet:
                            # get the values and check if they are valid
                            if snippet['start_pos'] is None or snippet['end_pos'] is None:
                                continue
                            possible_answer = context[snippet['start_pos']: snippet['end_pos']]
                            if unidecode.unidecode(possible_answer.lower()) == answer:
                                matches = [[snippet['start_pos'], snippet['end_pos']]]
                            else:
                                continue
                        else:
                            continue

                    # we have at least one match, iterate through each match
                    # and create an example for each get the matching text
                    for start_pos, end_pos in matches:
                        # pre-tokenize and correct answer start and end if necessary
                        pre_processed_context, answer_start, answer_end = pre_tokenize(context, start_pos, end_pos)

                        # We need to verify that the start and end positions actually point to the answer before we
                        # expect our feature creation code to find the correctly tokenized positions.
                        matching_text = pre_processed_context[answer_start:answer_end]
                        if answer != matching_text:
                            continue

                        # In the case where we match answers to the context, we know that context contains the answer
                        # if the question is not impossible. We don't need to check this like in SQuAD.
                        # Create an example for every match
                        examples_from_question.append(
                            FactoidExample(question_id, q_type, question, pre_processed_context,
                                           matching_text, start_pos, end_pos)
                        )
        return examples_from_question

    def process_yesno_question(data):
        question = data["body"]
        snippets = data["snippets"]
        answer = None if "exact_answer" not in data else data["exact_answer"].lower()

        if type(snippets) == str:  # sometimes, snippets list a string representation of a list
            snippets = ast.literal_eval(snippets)  # convert to list from string

        examples_from_question = []
        for snippet in snippets:
            snippet_text = snippet['text']
            example = BinaryExample(question_id="test_question",
                                    question=question,
                                    context=snippet_text,
                                    answer=answer)
            examples_from_question.append(example)
        return examples_from_question

    fc_map = {
        "factoid": process_factoid_or_list_question,
        "yesno": process_yesno_question,
        "list": process_factoid_or_list_question,
    }
    question_type = data_point["type"]
    fc = fc_map[question_type]

    example_list = fc(data_point)  # apply the right function for the question type
    total_examples.extend(example_list)  # collate examples
    return total_examples


base_path = Path(__file__).parent


number_of_factoid_predictions = 5
number_of_list_predictions = 100
torch.backends.cudnn.benchmark = torch.cuda.is_available()

# ---- Set torch backend and set seed ----
torch.backends.cudnn.benchmark = torch.cuda.is_available()

model_size = "base"
max_length = 128 if model_size == "small" else 256
batch_size = 128 if model_size == "small" else 32

yes_no_checkpoint = "archive/small_yesno_0_0_1_1" if model_size == "small" else "archive/base_yesno_0_0_1_0"
factoid_checkpoint = "archive/small_factoid,list_0_0_17_192" if model_size == "small" else "archive/base_factoid,list_0_0_10_800"


device = "cuda" if cuda.is_available() else "cpu"
base_path = Path(__file__).parent
base_checkpoint_dir = (base_path / '../../checkpoints').resolve()
pretrain_checkpoint_dir = (base_checkpoint_dir / 'pretrain').resolve()
finetune_checkpoint_dir = (base_checkpoint_dir / 'finetune').resolve()
all_datasets_dir = (base_checkpoint_dir / '../datasets').resolve()
predictions_dir = (base_checkpoint_dir / '../predictions').resolve()

config = get_model_config(model_size, pretrain=False)
config["num_warmup_steps"] = 100  # dummy value to avoid an error when building fine-tuned checkpoint.
config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
config["dataset"] = "bioasq"
yesno_model, _, _, electra_tokenizer, config = build_finetuned_from_checkpoint(model_size, config["device"],
                                                                                  pretrain_checkpoint_dir,
                                                                                  finetune_checkpoint_dir,
                                                                                  ("", yes_no_checkpoint), "yesno",
                                                                                  config)

factoid_model, _, _, _, _ = build_finetuned_from_checkpoint(model_size, config["device"],
                                                                                  pretrain_checkpoint_dir,
                                                                                  finetune_checkpoint_dir,
                                                                                  ("", factoid_checkpoint), "factoid",
                                                                                  config)


def process_question(question, context, question_type):
    raw_dataset = make_example(question, context, question_type)

    # get only the data relevant to this specific question type
    print("Converting raw text to features.")
    features = convert_examples_to_features(raw_dataset, electra_tokenizer, config["max_length"])
    print("Created {} features of length {}.".format(len(features), config["max_length"]))
    test_dataset = QADataset(features)
    data_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=collate_wrapper)

    # ------ START THE EVALUATION PROCESS ------
    if question_type == "factoid":
        results_by_question_id, _ = evaluate_factoid(factoid_model, data_loader, electra_tokenizer,
                                                                  training=False,
                                                                  dataset="bioasq")

        return {"prediction": results_by_question_id["test_question"]["predictions"]}
    elif question_type == "yesno":
        results_by_question_id, _ = evaluate_yesno(yesno_model, data_loader)

        return {"prediction": [results_by_question_id["test_question"]["predictions"]]}
    elif question_type == "list":
        results_by_question_id, _ = evaluate_list(factoid_model, data_loader, electra_tokenizer,
                                                               training=False,
                                                               dataset="bioasq")
        print(results_by_question_id)
        return {"prediction": results_by_question_id["test_question"]["predictions"]}


