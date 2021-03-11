import json
from pathlib import Path
import string
import spacy
import re
import ast
import unidecode

nlp = spacy.load('en_core_web_sm')


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


if __name__ == "__main__":
    base_path = Path(__file__).parent

    dataset_dir = (base_path / '../datasets/bioasq/raw_data').resolve()
    path_to_file = str(dataset_dir) + '/enriched_training8b.json'  # TODO CHANGE
    path_to_new_file = str(dataset_dir) + '/new_enriched_training8b.json'

    with open(path_to_file, 'rb') as f:
        bioasq_dict = json.load(f)

    stop_now = False
    q_num = 0
    for data in bioasq_dict['questions']:
        q_num += 1
        print('Question number', q_num)
        if stop_now:
            break
        q_type = data["type"]

        # we don't care about summary, yesno and list questions
        if q_type in ['summary', 'yesno', 'list']:
            continue

        question_id = data["id"]
        question = data["body"]
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
            if stop_now:
                break

            context = snippet['text']
            # As we match to the context paragraph ourselves, we can remove whitespace without affecting start/end pos'
            context = context.strip()

            all_answer_matches = []
            for answer in flattened_answer_list:  # for each of our candidate answers
                answer = None if answer is None else unidecode.unidecode(answer.lower())  # normalize the answer

                matches = match_answer_to_passage("".join(answer), context)
                all_answer_matches.extend(matches)

            if len(all_answer_matches) == 0:  # there are no matches in the passage and the question is impossible
                # check for a snippet start position
                if 'start_pos' in snippet and 'end_pos' in snippet:
                    # get
                    continue

                else: # find them
                    print('\nQID of question', data["id"])
                    print('Question: ', data["body"])
                    print('Type:', data["type"])
                    print('Answer:', data["exact_answer"])
                    print('Snippet:', context)
                    matching_section = input('Input Matching Section: ').strip()

                    if matching_section == "STOPNOW":
                        stop_now = True
                        break

                    if matching_section == "":
                        print('Skipping this one - cannot locate answer')
                        continue

                    data["exact_answer"] = [x if type(x) == list else [x] for x in data["exact_answer"]]
                    data["exact_answer"].append([matching_section])
                    flattened_answer_list.append(matching_section)

                    matches = match_answer_to_passage(matching_section, context)

                    if len(matches) == 0:
                        snippet['start_pos'] = "null"
                        snippet["end_pos"] = "null"
                        print('Skipping this one - cannot locate answer')
                        continue

                    print(matches)

                    # get first match
                    first_match = matches[0]
                    snippet['start_pos'] = first_match[0]
                    snippet["end_pos"] = first_match[1]

    open(path_to_new_file, 'w').close()
    with open(path_to_new_file, 'a') as outfile:
        json.dump(bioasq_dict, outfile)
