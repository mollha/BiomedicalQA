from tokenizers import Tokenizer
import pathlib
import pandas as pd
from transformers import ElectraTokenizerFast
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders

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


def train_bio_tokenizer():
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
    bio_tokenizer.model.save("bio_tokenizer", "electra-pubmed")


def create_tokenizers(additional_vocab_path):
    file = open(additional_vocab_path)
    lines = file.readlines()

    sizes = ["small", "base", "large"]

    for size in sizes:
        print("\nExtending the vocabulary of the pre-trained {} tokenizer to include words from the PubMed corpus."
              .format(f'google/electra-{size}-generator'))
        electra_tokenizer = ElectraTokenizerFast.from_pretrained(f'google/electra-{size}-generator')
        for line in lines:
            electra_tokenizer.add_tokens(line)

        model_files = electra_tokenizer.model.save("bio_tokenizer", "electra-duo-corpus")
        electra_tokenizer.model = WordPiece.from_file(*model_files, unk_token="[UNK]")
        electra_tokenizer.save("bio_tokenizer/electra-duo-corpus.json")
        electra_tokenizer.save_pretrained('bio_tokenizer/bio_electra_{}_tokenizer'.format(size))

    file.close()


if __name__ == "__main__":
    path_to_vocab = (base_path / "bio_tokenizer/electra-pubmed-vocab.txt").resolve()
    vocab_exists = path_to_vocab.exists()

    print("Path to {} {}, {} vocab.".format(str(path_to_vocab), "exists" if vocab_exists else "does not exist",
                                            "re-using" if vocab_exists else "creating"))
    if not vocab_exists:
        train_bio_tokenizer()

    create_tokenizers(path_to_vocab)
