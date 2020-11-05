import pandas as pd
import json

raw_file_path = '../Datasets/BioASQ/RawData/raw_data.json'


def get_summary_stats(data_frame: pd.DataFrame):
    column_names = [x for x in data_frame.columns]
    number_of_questions = len(data_frame)

    total_question_length = 0

    type_dict = {}

    for index, row in data_frame.iterrows():
        for col in column_names:
            content = row[col]

            if col == 'body':
                total_question_length += len(content)

            if col == 'type':
                if content in type_dict:
                    type_dict[content] += 1
                else:
                    type_dict[content] = 0

    # ------ PRINT TH£ SUMMARY STATS ------
    print('------- Question Types -------')
    for key in type_dict.keys():
        print(key + ':', type_dict[key])

    print('\n------- Question Length -------')
    print('Average Question Length:', round(total_question_length / number_of_questions, 2))


if __name__ == '__main__':
    # --------- READING DATA ----------

    with open(raw_file_path, 'r') as f:
        data = f.readlines()

    cleaned_data = [x.strip() for x in data]
    data = "".join(cleaned_data)
    res = json.loads(data)['questions']

    data_frame = pd.DataFrame.from_dict(res, orient='columns')


    get_summary_stats(data_frame)

    #drop_columns = ['concepts', 'id']  # columns to remove
    #for column in drop_columns:
        #data_frame = data_frame.drop([column], axis=1)
    data_frame.to_csv(path_or_buf='../Datasets/BioASQ/PreparedData/prepared_data.csv', index=False)




def convert_bio_asq_json():
    pass


def get_summary_stats():
    pass






if __name__ == "__main__":
    pass