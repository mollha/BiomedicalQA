import json
from pathlib import Path
from transformers import DistilBertTokenizerFast


# ------------ READ DATASETS INTO THEIR CORRECT FORMAT ------------
def read_squad(path_to_file: Path):
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


    def process_factoid(data):
        question_id = data["id"]
        question = data["body"]
        answer = data["exact_answer"]
        snippets = data["snippets"]

        print("id", question_id)
        print('type', data["type"])

        for snippet in snippets:
            article_id = snippet["document"].split('/').pop()
            section = snippet["beginSection"] if snippet["beginSection"] != "sections.0" else "abstract"

            if snippet["beginSection"] != snippet["endSection"]:
                raise Exception('{} is not {}'.format(snippet["beginSection"], snippet["endSection"]))


            try:
                article = articles_dict[article_id]
                # todo is it ok that some articles aren't here? probably not - let's try and find them.
            except KeyError:
                continue

            print(article)
            print("section name", section)
            paragraph = article[section]
            print(paragraph)

            beginOffset, endOffset = int(snippet["offsetInBeginSection"]), int(snippet["offsetInEndSection"])
            print('\nclipped section:', paragraph[beginOffset:endOffset])

            print('exact answer:', answer)
            print("begin", snippet["beginSection"])
            print("end", snippet["endSection"])



            # print('offsetInEnd', snippet["offsetInEndSection"])
            # print('offsetInBegin', snippet["offsetInBeginSection"])
            # print('snippet', snippet["text"])
            # print('length of text', len(snippet["text"]))
            print('keys', snippet.keys())
            print(article_id)
            print("docs", snippet["document"])




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
    # base_path = Path(__file__).parent
    # squad_dir = (base_path / '../datasets/squad/dev-v2.0.json').resolve()
    # data = read_squad(squad_dir)
    # print("length of data", len(data))
    #
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # train_contexts = [d["context"] for d in data]
    # train_questions = [d["question"] for d in data]
    # train_answers = [d["answer"] for d in data]
    #
    # train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    # add_token_positions(train_encodings, train_answers)
    # # print(train_encodings)

    base_path = Path(__file__).parent
    data_dir = (base_path / '../datasets/bioasq/training8b.json').resolve()
    data = read_bioasq(data_dir)
    # print("length of data", len(data))

    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # train_contexts = [d["context"] for d in data]
    # train_questions = [d["question"] for d in data]
    # train_answers = [d["answer"] for d in data]
    #
    # train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    # add_token_positions(train_encodings, train_answers)
    # print(train_encodings)



