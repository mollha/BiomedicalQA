import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import CustomObjectScope
import tensorflow_hub as hub
from bert import tokenization

BERT_URL = 'https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1'
module = hub.Module(BERT_URL)

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


text_to_vectorise = "This is the text"


tokenizer = create_tokenizer_from_hub_module()
tokenizer.tokenize(text_to_vectorise)



