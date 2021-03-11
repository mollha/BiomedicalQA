import json
from pathlib import Path
import string
import spacy
import re
import ast
import unidecode

nlp = spacy.load('en_core_web_sm')



if __name__ == "__main__":
    base_path = Path(__file__).parent

    dataset_dir = (base_path / '../datasets/bioasq/raw_data').resolve()
    path_to_file = str(dataset_dir) + '/BioASQ-task9bPhaseB-testset1.json'  # TODO CHANGE
    path_to_new_file = str(dataset_dir) + '/9B1_golden.json'

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
        if q_type in ['summary', 'factoid', 'list']:
            continue

        question_id = data["id"]
        question = data["body"]
        answer_list = None if "exact_answer" not in data else data["exact_answer"]
        snippets = data["snippets"]
        examples_from_question = []

        if type(snippets) == str:  # sometimes, snippets list a string representation of a list
            snippets = ast.literal_eval(snippets)  # convert to list from string

        print('\nQuestion:', question)
        print('Snippets:')
        for i, s in enumerate(snippets):
            print(i, s["text"])
        pred_answer = None
        while pred_answer is None:
            pred_answer = input('Answer: ').strip()
            if pred_answer not in {'y', 'n'}:
                pred_answer = None
            else:
                break

        if pred_answer == 'y':
            data["exact_answer"] = "yes"
        elif pred_answer == 'n':
            data["exact_answer"] = "no"


    open(path_to_new_file, 'w').close()
    with open(path_to_new_file, 'a') as outfile:
        json.dump(bioasq_dict, outfile)
