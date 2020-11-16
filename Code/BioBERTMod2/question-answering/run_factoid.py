# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils_qa import transform_n2b_factoid, eval_bioasq_standard, to_list

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Save checkpoint and log every X updates steps.
update_steps = 500



def get_default_settings():

    return {
        # Required parameters
        "model_name_or_path": None,  # Path to pretrained model or model identifier from huggingface.co/models
        "output_dir": None,  # The output directory where the model checkpoints and predictions will be written.
        "golden_file": None,  # BioASQ official golden answer file
        "official_eval_dir": './scripts/bioasq_eval',  # BioASQ official golden answer file

        # Other parameters
        "train_file": None, # "The input training file. If a data dir is specified, will look for the file there. If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        "predict_file": None, # The input evaluation file. If a data dir is specified, will look for the file there" If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        "per_gpu_eval_batch_size": 8, # Batch size per GPU/CPU for evaluation.
        "num_train_epochs": 3.0, # Total number of training epochs to perform.
        "eval_all_checkpoints": False,  # Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    }





def train(train_dataset, model, tokenizer, model_type, model_path, device, evaluate_all_checkpoints, output_dir):
    """ Train the model """
    train_batch_size = 8
    num_training_epochs = 1
    learning_rate = 8e-6 # The initial learning rate for Adam.
    weight_decay = 0.0  # Weight decay if we apply some.
    adam_epsilon = 1e-8 # Epsilon for Adam optimizer.
    max_grad_norm = 1.0  # Max gradient norm.
    version_2_with_negative = False

    overwrite_output_directory = True

    if (os.path.exists(output_dir) and os.listdir(
            output_dir) and not overwrite_output_directory):
        raise ValueError(
            "Output directory ({}) already exists. Set overwrite_output_directory to True to overcome.".format(
                output_dir
            )
        )

    # Create a SummaryWriter()
    tb_writer = SummaryWriter()


    # Random Sampler used during training.
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                  batch_size=train_batch_size)


    t_total = len(train_dataloader) // 1 * num_training_epochs

    # Prepare optimizer and schedule (linear warm up and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(model_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(model_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

    # Start Training!
    logger.info("---------- BEGIN TRAINING ----------")
    logger.info("Dataset Size = {}\nNumber of Epochs = {}".format(len(train_dataset), num_training_epochs))
    logger.info("Instantaneous batch size per GPU = %d", train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size
    )
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(model_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = model_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader))
            steps_trained_in_current_epoch = global_step % (len(train_dataloader))

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_training_epochs), desc="Epoch")

    # Added here for reproducibility
    # TODO SET SEED HERE AGAIN

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }


            if model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * 0).to(device)}
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % 1 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if update_steps > 0 and global_step % update_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if evaluate_all_checkpoints:
                        results = evaluate(model, tokenizer, model_type, output_dir, device, evaluate_all_checkpoints)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / update_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if update_steps > 0 and global_step % update_steps == 0:
                    output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    training_arguments = {}

                    torch.save(training_arguments, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)


    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(model, tokenizer, model_type, output_dir, device, dataset, examples, features, prefix=""):
    eval_batch_size = 12
    n_best_size = 20 # The total number of n-best predictions to generate in the nbest_predictions.json output file.
    max_answer_length = 30 # The maximum length of an answer that can be generated. This is needed because the start " and end predictions are not conditioned on one another.
    uncased_model = False # Set this flag if you are using an uncased model.
    verbose_logging = False # If true, all of the warnings related to data processing will be printed. " "A number of warnings are expected for a normal SQuAD evaluation.",
    version_2_with_negative = False,  # If true, the SQuAD examples contain some that do not have an answer.
    null_score_diff_threshold = 0.0 # If null_score - best_non_null is greater than the threshold predict null.
    golden_file = "../datasets/QA/BioASQ/7B_golden.json" # "gdrive/My Drive/BioBERT/datasets/QA/BioASQ/7B_golden.json"
    official_eval_dir = "./scripts/bioasq_eval" # "gdrive/My Drive/BioBERT/question-answering/scripts/bioasq_eval" # BioASQ official golden answer file


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sequential Sampler used during evaluation.
    eval_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * 0).to(device)}
                    )

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))

    if version_2_with_negative:
        output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None


    predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            uncased_model,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            verbose_logging,
            version_2_with_negative,
            null_score_diff_threshold,
            tokenizer,
        )

    # Transform the prediction into the BioASQ format
    logger.info("***** Transform the prediction file into the BioASQ format {} *****".format(prefix))
    transform_n2b_factoid(output_nbest_file, output_dir)

    # Evaluate with the BioASQ official evaluation code
    pred_file = os.path.join(output_dir, "BioASQform_BioASQ-answer.json")
    eval_score = eval_bioasq_standard(str(5), pred_file, golden_file, official_eval_dir)

    print("** BioASQ-factoid Evaluation Results ************************************")
    print(f"   S. Accuracy = {float(eval_score[1])*100:.2f}")
    print(f"   L. Accuracy = {float(eval_score[2])*100:.2f}")
    print(f"   MRR         = {float(eval_score[3])*100:.2f}")
    

