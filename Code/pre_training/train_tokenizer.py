from tokenizers import Tokenizer, BertWordPieceTokenizer
from tokenizers.models import BPE

from transformers import WordpieceTokenizer
import pathlib
from transformers import ElectraTokenizerFast, ElectraTokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer



"""
--- Code for training a Biomedical-domain tokenizer ----

The Google Pre-trained Electra Tokenizer is trained on plain-english text; hence,
many biomedical terms are absent from its vocabulary. Consequently, many of the output
tokens of the tokenizer are ['UNK'] i.e. unknown.

Based on the HuggingFace tutorial at:
https://huggingface.co/docs/tokenizers/python/latest/pipeline.html#all-together-a-bert-tokenizer-from-scratch
"""
base_path = pathlib.Path(__file__).parent


def find_text_files(directory):
    return [str(file) for file in list(directory.glob('*.csv'))]


def train_bio_tokenizer_from_scratch(path_to_output: str):
    """
    Generates the electra-pubmed-vocab.txt file in bio_tokenizer. The general-vocab file should be downloaded from:
    https://huggingface.co/google/electra-small-discriminator/resolve/main/vocab.txt

    :param path_to_output:
    :return:
    """
    processed_data_directory = (base_path / '../datasets/PubMed/processed_data').resolve()
    text_files = find_text_files(processed_data_directory)

    bio_tokenizer = Tokenizer(WordPiece())
    trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    bio_tokenizer.train(trainer, text_files)

    # Initialize a tokenizer
    # bio_tokenizer = BertWordPieceTokenizer()
    # bio_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    # bio_tokenizer.pre_tokenizer = Whitespace()
    # bio_tokenizer.post_processor = TemplateProcessing(
    #     single="[CLS] $A [SEP]",
    #     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    #     special_tokens=[
    #         ("[CLS]", 1),
    #         ("[SEP]", 2),
    #     ],
    # )

    # Customize training
    # bio_tokenizer.train(files=text_files, vocab_size=52_000, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # bio_tokenizer.train(trainer, text_files)
    # bio_tokenizer.save_model(path_to_output, "electra-pubmed")
    bio_tokenizer.model.save(path_to_output, "electra-pubmed")
    # bio_tokenizer.save_pretrained(path_to_output + '/bio_electra_tokenizer')


def train_bio_tokenizer_from_pretrained(path_to_output: str):
    processed_data_directory = (base_path / '../datasets/PubMed/processed_data').resolve()
    text_files = find_text_files(processed_data_directory)

    bio_tokenizer = Tokenizer(WordPiece())
    bio_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    bio_tokenizer.pre_tokenizer = Whitespace()
    bio_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = WordPieceTrainer(
        vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    bio_tokenizer.train(trainer, text_files)
    bio_tokenizer.model.save(path_to_output, "electra-pubmed")


def merge_vocabularies(path_to_vocab1: str, path_to_vocab2: str, path_to_output: str):
    file1 = open(path_to_vocab1, 'r')
    file2 = open(path_to_vocab2, 'r')

    open(path_to_output + "/combined-vocab.txt", 'w')
    file3 = open(path_to_output + "/combined-vocab.txt", 'a')

    lines1 = file1.readlines()
    lines2 = file2.readlines()

    print(len(lines1))
    print(len(lines2))

    lines1.extend(lines2)
    print(len(lines1))
    vocab_set = set()

    for line in lines1:
        line = line.strip()
        vocab_set.add(line)

    for item in vocab_set:
        file3.write(item + '\n')

    # create a tokenizer with the general vocabulary
    # tokenizer = ElectraTokenizerFast.from_pretrained(f'google/electra-small-discriminator')
    #
    # file = open(path_to_new_vocab, 'r')
    # lines = file.readlines()
    # print("\nExtending the vocabulary of the pre-trained {} tokenizer to include words from the PubMed corpus."
    #       .format(f'google/electra-small-discriminator'))
    #
    # for idx in range(len(lines)):
    #     lines[idx] = lines[idx].strip()
    #
    # lines1 = lines[0: (len(lines) // 2)]
    # lines2 = lines[len(lines) // 2:]
    # lines_list = [lines1, lines2]
    #
    # for line_list in lines_list:
    #     tokenizer.add_tokens(line_list)

    # tokenizer.model.save(path_to_output, "complete")
    # file.close()

    pass


def create_tokenizers(additional_vocab_path, path_to_output: str):
    file = open(additional_vocab_path)
    lines = file.readlines()

    for idx in range(len(lines)):
        lines[idx] = lines[idx].strip()


    sizes = ["small", "base", "large"]

    for size in sizes:
        print("\nExtending the vocabulary of the pre-trained {} tokenizer to include words from the PubMed corpus."
              .format(f'google/electra-{size}-generator'))
        electra_tokenizer = ElectraTokenizerFast.from_pretrained(f'google/electra-{size}-generator')
        electra_tokenizer.add_tokens(lines)
        electra_tokenizer.save_pretrained(path_to_output + '/bio_electra_{}_tokenizer'.format(size))

    file.close()


if __name__ == "__main__":
    path_to_output_dir = (base_path / "bio_tokenizer").resolve()
    pathlib.Path(path_to_output_dir).mkdir(exist_ok=True, parents=True)

    path_to_electra_pubmed = pathlib.Path(path_to_output_dir / "electra-pubmed-vocab.txt").resolve()
    path_to_general_vocab = pathlib.Path(path_to_output_dir / "general-vocab.txt").resolve()
    path_to_combined_vocab = pathlib.Path(path_to_output_dir / "combined-vocab.txt").resolve()

    if not pathlib.Path.exists(path_to_electra_pubmed):
        # create the electra-pubmed vocab
        train_bio_tokenizer_from_scratch(str(path_to_output_dir))

    if not pathlib.Path.exists(path_to_general_vocab):
        raise ValueError("Get electra-small-discriminator vocab first.")

    if not pathlib.Path.exists(path_to_combined_vocab):
        merge_vocabularies(str(path_to_general_vocab), str(path_to_electra_pubmed), str(path_to_output_dir)),

    vocab_dict = {"pubmed_vocab": path_to_electra_pubmed, "general_vocab": path_to_general_vocab,
                  "combined_vocab": path_to_combined_vocab}

    for vocab_name in vocab_dict:
        vocab_path = vocab_dict[vocab_name]
        bio_tokenizer = ElectraTokenizer(vocab_file=vocab_path)

        tokenizer_path = pathlib.Path(path_to_output_dir / "bio_electra_tokenizer_{}".format(vocab_name)).resolve()
        bio_tokenizer.save_pretrained(str(tokenizer_path))

    # path_to_tokenizer = str(pathlib.Path(path_to_output_dir / "bio_electra_tokenizer").resolve())

    # electra_tokenizer = ElectraTokenizerFast.from_pretrained(str(path_to_output_dir))


    # path_to_vocab = (path_to_output_dir / "electra-pubmed-vocab.txt").resolve()
    # vocab_exists = path_to_vocab.exists()
    #
    # print("Path to {} {}, {} vocab.".format(str(path_to_vocab), "exists" if vocab_exists else "does not exist",
    #                                         "re-using" if vocab_exists else "creating"))
    # if not vocab_exists:
    #     train_bio_tokenizer(str(path_to_output_dir))
    #
    # create_tokenizers(path_to_vocab, str(path_to_output_dir))
