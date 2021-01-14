from pathlib import Path
import pandas as pd
import json

# convert the BioAsq structured data into squad structure
squad_struct = {
    "version": "",
    "data":
    [
        {
            "title": "",
            "paragraphs": []
         }
    ]
}


def convert_bioasq_to_squad(path_to_dataset: Path):
    """
    Convert datasets in the bioasq format to the squad format.

    The Squad V2.0 structure is as follows:
    {"version": "", "data" : [
        "title": "", "paragraphs": [{
            "qas" : [{
                "question": "", "id": "", "is_impossible": bool, "answers": [{
                    "text": "", "answer_start": ""
                }, ...]
            }, ...],
            "context: ""
        }]
    ]}



    :return:
    """

    bioasq_version = "8b"   # this is to match the version key in squad
    squad_struct["version"] = bioasq_version
    squad_struct["data"][0]["title"] = bioasq_version
    squad_questions = squad_struct["data"][0]["paragraphs"]

    def process_question(question: dict):
        print(question.keys())
        # create a new question for each snippet
        snippets = question["snippets"]

        for snippet in snippets:
            snippet_text = snippet["text"]
            snippet_start_idx = snippet["offsetInBeginSection"]
            snippet_end_idx = snippet["offsetInEndSection"]
            print(snippet.keys())
            print(snippet_start_idx, snippet_end_idx)

            formatted_question = {"qas": [{
                "question": question["body"],
                "id": question["id"],
                "is_impossible": False,  # assume no impossible answers in BioASQ
                "answers": [{
                    "text": "", "answer_start": ""
                }]
            }],
                "context": snippet
            }

            squad_questions.append(formatted_question)

        # "qas": [{
        #     "question": "", "id": "", "is_impossible": bool, "answers": [{
        #         "text": "", "answer_start": ""
        #     }, ...]
        # }, ...]





        quit()


    with open(path_to_dataset) as json_file:
        json_data = json.load(json_file)

        for q in json_data["questions"]:
            process_question(q)



if __name__ == "__main__":
    # Log Process ID

    base_path = Path(__file__).parent
    json_data_dir = (base_path / '../datasets/BioA/BioASQ-training8b').resolve()
    training8b = (json_data_dir / 'training8b.json').resolve()

    convert_bioasq_to_squad(training8b)


