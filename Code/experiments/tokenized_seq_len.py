import pathlib
from transformers import ElectraTokenizerFast


def tokenize_dataset(tokenizer_list: list, list_path_to_csv_files: list):
    avg_len = [[0, 0]] * len(tokenizer_list)

    # iterate through every file
    for file in list_path_to_csv_files:
        print("Parsing file {}.".format(file))

        first_line = True
        with open(file, 'r') as current_file: # open the file
            # read lines one at a time, skipping the first line
            for line in current_file.readlines():
                if first_line:
                    first_line = False
                    continue

                # tokenize the line for every tokenizer
                for idx in range(len(tokenizer_list)):
                    tokenized_line = tokenizer_list[idx].tokenize(line)
                    avg_len[idx] = [avg_len[idx][0] + len(tokenized_line), avg_len[idx][1] + 1]

    average_lengths = []
    for num_tokens, num_samples in avg_len:
        avg_length = num_tokens / num_samples
        average_lengths.append(avg_length)

    return average_lengths


def find_text_files(directory):
    return [str(file) for file in list(directory.glob('*.csv'))]


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent

    # locate our datasets
    processed_data_directory = (base_path / '../datasets/PubMed/processed_data').resolve()
    text_files = find_text_files(processed_data_directory)

    # locate our tokenizer directory
    pre_trained_tok_dir = (base_path / "../pre_training/bio_tokenizer").resolve()

    tokenizer_paths = [str((pre_trained_tok_dir / "bio_electra_tokenizer_general_vocab").resolve()),
                       str((pre_trained_tok_dir / "bio_electra_tokenizer_pubmed_vocab").resolve()),
                       str((pre_trained_tok_dir / "bio_electra_tokenizer_combined_vocab").resolve())]

    tokenizer_list = []

    vocab_names = ["general_vocab", "pubmed_vocab", "combined_vocab"]
    for tokenizer_path in tokenizer_paths:
        tokenizer = ElectraTokenizerFast.from_pretrained(tokenizer_path)
        tokenizer_list.append(tokenizer)

    average_length_list = tokenize_dataset(tokenizer_list, text_files)

    for idx in range(len(vocab_names)):
        print("Vocab file '{}' has average length {}.".format(vocab_names[idx], average_length_list[idx]))


# ---------- RESULTS ----------
# AVG LENGTH FOR GENERAL VOCAB = 225.04
# AVG LENGTH FOR PUBMED VOCAB = 192.78
# AVG LENGTH FOR COMBINED VOCAB = 190.88

