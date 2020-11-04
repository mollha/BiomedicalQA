import pandas as pd
import json

raw_file_path = '../Datasets/BioASQ/RawData/raw_data.json'


def get_summary_stats(data_frame: pd.DataFrame):
    column_names = [x for x in data_frame.columns]

    type_dict = {}

    for index, row in data_frame.iterrows():
        for col in column_names:

            if col == 'type':
                question_type = row[col]
                if question_type in type_dict:
                    type_dict[question_type] += 1
                else:
                    type_dict[question_type] = 0

    print(type_dict)


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