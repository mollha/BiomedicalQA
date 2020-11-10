import torch
import pandas as pd


def read_bioasq():
    raw_file_path = './Datasets/BioASQ/PreparedData/prepared_data.csv'
    data_frame = pd.read_csv(raw_file_path)

    contexts = []
    questions = []
    answers = []

    for index, row in data_frame.iterrows():
        context = row['snippets']
        print(context)
        question = row['body']

        for answer in row['ideal_answer']:
            contexts.append(context)
            questions.append(question)
            answers.append(answer)

        if index == 100:
            break

    return contexts, questions, answers


train_contexts, train_questions, train_answers = read_bioasq()


class BioASQDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


# train_dataset = BioASQDataset(train_encodings)
# val_dataset = BioASQDataset(val_encodings)
