import json
from pathlib import Path


# ------------ READ DATASETS INTO THEIR CORRECT FORMAT ------------
def read_squad(path_to_file: Path):
    """
    Read the squad data into three categories of contexts, questions and answers

    This function is adapted from the Huggingface SQuAD tutorial at:
    https://huggingface.co/transformers/custom_datasets.html#qa-squad
    :param path_to_file: path to file containing squad data
    :return:
    """

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

    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    add_end_idx(answer, context)
                    dataset.append({"context": context, "question": question, "answer": answer})


    return dataset


if __name__ == "__main__":
    # todo delete this section after testing
    base_path = Path(__file__).parent
    squad_dir = (base_path / '../datasets/SQuAD/dev-v2.0.json').resolve()
    data = read_squad(squad_dir)
    print("data", data)


