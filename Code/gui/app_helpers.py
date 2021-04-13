import os
import string
from pathlib import Path
from modelling import CostSensitiveSequenceClassification
from os import path

import torch
from torch import cuda, device, load, softmax, argmax, tensor, no_grad, stack
from transformers import (
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraConfig, ElectraTokenizerFast
)

class ELECTRAModel(torch.nn.Module):
    def __init__(self, generator, discriminator, electra_tokenizer):
        super().__init__()
        self.generator, self.discriminator = generator, discriminator
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.)
        self.electra_tokenizer = electra_tokenizer

    def to(self, *args, **kwargs):
        " Also set dtype and device of contained gumbel distribution if needed. "
        super().to(*args, **kwargs)
        a_tensor = next(self.parameters())
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=a_tensor.device,
                                                                          dtype=a_tensor.dtype),
                                                             torch.tensor(1., device=a_tensor.device,
                                                                          dtype=a_tensor.dtype))

    def forward(self, masked_inputs, is_mlm_applied, labels):
        """
        masked_inputs (Tensor[int]): (B, L)
        is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability
        labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
        """
        # this assumes that padding has taken place ALREADY
        # the attention mask is the tensortext of true and false
        # the token_type_ids is the tensor of zeros and ones

        attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs)
        gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[0]  # (B, L, vocab size)
        # reduce size to save space and speed
        mlm_gen_logits = gen_logits[is_mlm_applied, :]  # ( #mlm_positions, vocab_size)

        with torch.no_grad():
            pred_toks = self.sample(mlm_gen_logits)  # ( #mlm_positions, )

            # produce inputs for discriminator
            generated = masked_inputs.clone()  # (B,L)
            generated[is_mlm_applied] = pred_toks  # (B,L)

            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone()  # (B,L)
            is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied])  # (B,L)

        disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0]  # (B, L)

        return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

    def _get_pad_mask_and_token_type(self, input_ids):
        """
        Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask
        and token_type_ids and won't be unnecessarily large,
        thus, preventing cpu processes loading batches from consuming lots of cpu memory and slow down the machine.
        """
        # get a TensorText object (like a list) of booleans indicating whether each
        # input token is a pad token
        attention_mask = input_ids != self.electra_tokenizer.pad_token_id

        # create a list of lists containing zeros and ones.
        # for every boolean in the attention mask, replace this with 0 if positive and 1 if negative.
        token_type_ids = torch.tensor([[int(not boolean) for boolean in sub_mask] for sub_mask in iter(attention_mask)],
                                      device=input_ids.device)
        return attention_mask, token_type_ids

    def sample(self, logits):
        """
        Implement gumbel softmax as there is a bug in torch.nn.functional.gumbel_softmax when fp16
        (https://github.com/pytorch/pytorch/issues/41663) is used.

        Gumbel softmax is equal to what the official ELECTRA code does
        i.e. standard gumbel dist. = -ln(-ln(standard uniform dist.))

        :param logits:
        :return:
        """
        return (logits.float() + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)


word_nums = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

def contains_k(text: str):
    # Given a piece of text, check if text contains k. e.g. give two examples of...
    # split on whitespace
    for word in text.split():
        if word in word_nums:
            return word_nums.index(word)
        if word.isdigit():
            return int(word)
    return None

def combine_tokens(token_list: list) -> str:
    build_string = []

    for token in token_list:
        if '#' not in token and token not in string.punctuation:  # the start of a word and not punctuation
            if len(build_string) > 0 and build_string[-1] != "-":
                build_string.append(" " + token)
            else:
                build_string.append(token)
        else:
            raw_token = token.strip().lstrip('#')
            build_string.append(raw_token)

    return ''.join(build_string).strip()

base_path = Path(__file__).parent

model_paths = {
    'yesno': "models/yesno_model/small_yesno_3_129918_29_103",  # "yesno_model/small_yesno_26_11229_19_103",
    'factoid': "models/factoid_model/small_factoid,list_3_149918_8_0"
}
path_to_biotokenizer = "models/bio_tokenizer/bio_electra_tokenizer_pubmed_vocab"

model_size = "small"
max_length = 128
batch_size = 128
device = "cuda" if cuda.is_available() else "cpu"




def load_model(modeL_type, tokenizer):
    model_path = model_paths[modeL_type]
    path_to_checkpoint = (base_path / model_path).resolve()
    path_to_discriminator = os.path.join(path_to_checkpoint, "discriminator")

    discriminator_config = ElectraConfig.from_pretrained(f'google/electra-{model_size}-discriminator')
    discriminator_config.vocab_size = tokenizer.vocab_size

    ElectraForQuestionAnswering(discriminator_config)

    if modeL_type == "factoid":
        qa_model = ElectraForQuestionAnswering(config=discriminator_config)
    elif modeL_type == "yesno":
        qa_model = CostSensitiveSequenceClassification(config=discriminator_config)
    else:
        raise Exception('Model type is either factoid or yesno.')

    qa_model = qa_model.from_pretrained(path_to_discriminator)


    qa_model.to(device)
    return qa_model


def load_tokenizer():
    electra_tokenizer = ElectraTokenizerFast.from_pretrained(path_to_biotokenizer)
    return electra_tokenizer


tokenizer = load_tokenizer()

print('Loading yes/no model')
yesno_model = load_model("yesno", tokenizer=tokenizer)

print("Loading factoid model")
factoid_model = load_model("factoid", tokenizer=tokenizer)




# -------- FUNCTIONS FOR PROCESSING QUESTIONS -------------
def process_yesno(question, context):
    yesno_model.eval()
    with no_grad():
        tokenized_input = tokenizer(question, context, padding="max_length", truncation="only_second",
                                    max_length=max_length)
        print(tokenized_input)

        inputs = {
            "input_ids": stack(batch_size*[tensor(tokenized_input["input_ids"], device=device)]),
            "attention_mask": stack(batch_size*[tensor(tokenized_input["attention_mask"], device=device)]),
            "token_type_ids": stack(batch_size*[tensor(tokenized_input["token_type_ids"], device=device)]),
        }

        print(inputs["input_ids"])
        outputs = yesno_model(**inputs)
        logits = outputs.logits  # assumes using CPU
        class_probabilities = softmax(logits, dim=1)
        print(class_probabilities)
        predicted_label = argmax(class_probabilities[0])

    return "Yes" if predicted_label == 1 else "No"



def process_factoid_list(question, context, q_type):
    factoid_model.eval()
    with no_grad():
        special_tokens_ids = {tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
        feature_list = []

        # -- Perform a doc stride here, this will give us more predictions to work with --
        tokenized_question = tokenizer.tokenize(question)
        tokenized_context = tokenizer.tokenize(context)

        # [CLS] QUESTION [SEP] SHORT_CONTEXT [SEP]
        num_question_tokens = len(tokenized_question)
        num_additional_tokens = 3  # refers to the [CLS] tokens and [SEP] tokens used when combining question & context
        num_context_tokens = max_length - num_question_tokens - num_additional_tokens  # e.g. SHORT_CONTEXT length

        question_input_ids = tokenizer.convert_tokens_to_ids(tokenized_question)
        question_attention_mask = len(question_input_ids) * [1]  # pay attention to all question input ids.
        question_token_type_ids = len(question_input_ids) * [0]  # token type ids are 0 for question tokens.

        context_input_ids = tokenizer.convert_tokens_to_ids(tokenized_context)
        context_attention_mask = len(context_input_ids) * [1]  # pay attention to all context input ids.
        context_token_type_ids = len(context_input_ids) * [1]  # token type ids are 1 for context tokens.

        for left_clip in range(0, len(context_input_ids), 20):
            clipped_input_ids = context_input_ids[left_clip: min(left_clip + num_context_tokens, len(context_input_ids))]
            clipped_attention_mask = context_attention_mask[left_clip: min(left_clip + num_context_tokens, len(context_attention_mask))]
            clipped_token_type_ids = context_token_type_ids[left_clip: min(left_clip + num_context_tokens, len(context_token_type_ids))]

            all_input_ids = [tokenizer.cls_token_id] + question_input_ids + [tokenizer.sep_token_id] + clipped_input_ids + [tokenizer.sep_token_id]
            all_attention_mask = [1] + question_attention_mask + [1] + clipped_attention_mask + [1]
            all_token_type_ids = [0] + question_token_type_ids + [0] + clipped_token_type_ids + [1]

            # pad the end with zeros if we have shorter length
            all_input_ids.extend([tokenizer.pad_token_id] * (max_length - len(all_input_ids)))  # add the padding token
            all_attention_mask.extend([0] * (max_length - len(all_attention_mask)))  # do not attend to padded tokens
            all_token_type_ids.extend([0] * (max_length - len(all_token_type_ids)))  # part of the context

            # If it is not included, for impossible instances the target prediction
            # for both start and end (tokenized) position is 0, i.e. the [CLS] token
            # This is -1 for examples and 0 for features, as tokenized pos in features & char pos in examples
            tokenized_input = {
                "input_ids": tensor(all_input_ids, device=device),
                "attention_mask": tensor(all_attention_mask, device=device),
                "token_type_ids": tensor(all_token_type_ids, device=device),
            }

            feature_list.append(tokenized_input)

        num_valid = len(feature_list)
        num_remaining = batch_size - num_valid

        input_ids_sublist = stack([f["input_ids"] for f in feature_list] + (num_remaining * [feature_list[0]["input_ids"]]))
        attention_mask_sublist = stack([f["attention_mask"] for f in feature_list] + (num_remaining * [feature_list[0]["attention_mask"]]))
        token_type_ids_sublist = stack([f["token_type_ids"] for f in feature_list] + (num_remaining * [feature_list[0]["token_type_ids"]]))

        inputs = {
            "input_ids": input_ids_sublist,
            "attention_mask": attention_mask_sublist,
            "token_type_ids": token_type_ids_sublist,
        }


        # model outputs are always tuples in transformers
        outputs = factoid_model(**inputs)
        start_logits, end_logits = outputs.start_logits[:num_valid], outputs.end_logits[:num_valid]


        answer_starts, start_indices = torch.topk(start_logits, k=100, dim=1)
        answer_ends, end_indices = torch.topk(end_logits, k=100, dim=1)

        start_end_positions = [x for x in zip(start_indices, end_indices)]
        start_end_probabilities = [x for x in zip(answer_starts, answer_ends)]

        index = 0
        (starts_tensor, ends_tensor) = start_end_positions[0]
        probabilities_of_starts, probabilities_of_ends = start_end_probabilities[index]
        sub_start_end_positions = zip(starts_tensor, ends_tensor)  # zip the start and end positions

        sub_start_end_probabilities = list(zip(probabilities_of_starts, probabilities_of_ends))
        input_ids = inputs["input_ids"][index]

        list_of_predictions = []  # gather all of the predictions for this question
        for sub_index, (s, e) in enumerate(
                sub_start_end_positions):  # convert the start and end positions to answers.
            # get the probabilities associated with this prediction
            probability_of_start, probability_of_end = sub_start_end_probabilities[sub_index]

            if e <= s:  # if end position is less than or equal to start position, skip this pair
                continue

            clipped_ids = [t for t in input_ids[int(s):int(e)] if t not in special_tokens_ids]
            clipped_tokens = tokenizer.convert_ids_to_tokens(clipped_ids, skip_special_tokens=True)
            predicted_answer = combine_tokens(clipped_tokens)

            if len(predicted_answer) > 100:
                continue  # if length is more than 100 and we are evaluating on bioasq, skip this pair

            # put our prediction in the list, alongside the probabilities (pred, start_prob + end_prob)
            # if neither start probability or end probability are negative
            # if probability_of_start > 0 and probability_of_end > 0:
            list_of_predictions.append(
                (predicted_answer, probability_of_start.item() + probability_of_end.item()))

        pred_lists = list_of_predictions
        pred_lists.sort(key=lambda val: val[1], reverse=True)

        if q_type == "list":
            k = 100 if contains_k(question) is None else contains_k(question)

            # For each list question, each participating system will have to return a single list* of entity names, numbers,
            # or similar short expressions, jointly taken to constitute a single answer (e.g., the most common symptoms of
            # a disease). The returned list will have to contain no more than 100 entries of no more than
            # 100 characters each.
            best_predictions = []
            num_best_predictions = 0

            # decide what our probability threshold is going to be
            # we only want to do this if k is not 100 (i.e. default)
            if k == 100:  # perform probability thresholding
                # find the prediction with the highest probability
                prediction, highest_probability = pred_lists[0]  # most probable
                probability_threshold = highest_probability / 0.85 if highest_probability < 0 else highest_probability * 0.85
                pred_lists = [(pred, prob) for pred, prob in pred_lists if prob >= probability_threshold]

            for pred, probability in pred_lists:
                if num_best_predictions >= k:
                    break

                # don't put repeats in our list.
                cleaned_best_pred = [p.replace(" ", "").strip().lower() for p in best_predictions]
                if pred not in best_predictions and pred.replace(" ", "").strip().lower() not in cleaned_best_pred:
                    num_best_predictions += 1
                    best_predictions.append(pred)

            return best_predictions

        elif q_type == "factoid":
            # For each factoid question in BioASQ, each participating system will have to return a list of up to 5 entity names
            # (e.g., up to 5 names of drugs), numbers, or similar short expressions, ordered by decreasing confidence.
            k = 5

            # pred_lists[: min(len(pred_lists), k)]  # take up to k of the best predictions
            best_predictions = []
            best_probabilities = []
            num_best_predictions = 0
            for pred, probability in pred_lists:
                if num_best_predictions >= k:
                    break

                # don't put repeats in our list.
                if pred not in best_predictions:
                    num_best_predictions += 1
                    best_predictions.append(pred)
                    best_probabilities.append(probability)

            return best_predictions


def process_question(question, context, question_type):
    if question_type == "yesno":
        preds = process_yesno(question, context)
        return {"prediction": [preds]}
    else:
        preds = process_factoid_list(question, context, question_type)
        return {"prediction": preds}

