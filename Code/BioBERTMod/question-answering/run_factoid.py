""" Finetuning HuggingFace Models, such as BioBERT and Electra for question-answering on SQuAD and BioASQ."""

import glob
import os
import random
from time import time

import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult
from utils_qa import transform_n2b_factoid, eval_bioasq_standard, load_and_cache_examples, EnvironmentSettings

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

output_directory = "./output"
model_name_or_path = "google/electra-small-discriminator"  # "dmis-lab/biobert-base-cased-v1.1",
logging_steps = 500
evaluate_during_training = False

def set_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, tokenizer, settings, prefix=""):
    device = torch.device("cpu")

    eval_dataset, examples, features = load_and_cache_examples(args, tokenizer, model_name_or_path, evaluate=True, output_examples=True)
    eval_batch_size = settings.unpack_eval_settings()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create a DataLoader
    # randomly sample the evaluation dataset with the specified batch size
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=eval_batch_size)

    # Evaluation
    print("\n-------- Conducting Evaluation {} --------".format(prefix))
    print("Sample Size = {}\n Batch size = {}\n".format(len(eval_dataset), eval_batch_size))

    result_list = []
    start_time = time()

    # Iterate through batches of evaluation data and collect results
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            feature_indices = batch[3]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output
            result_list.append(SquadResult(unique_id, start_logits, end_logits))

    # -------- COLLATE SUMMARY STATISTICS --------
    evaluation_time = time() - start_time
    print("Evaluation time: {} seconds ({} seconds per example)".format(evaluation_time, evaluation_time / len(eval_dataset)))

    # Compute predictions
    output_prediction_file = os.path.join(output_directory, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_directory, "nbest_predictions_{}.json".format(prefix))

    if args["version_2_with_negative"]:
        output_null_log_odds_file = os.path.join(output_directory, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    compute_predictions_logits(
        examples,
        features,
        result_list,
        args["n_best_size"],
        args["max_answer_length"],
        args["do_lower_case"],
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        True,   # verbose logging
        args["version_2_with_negative"],
        args["null_score_diff_threshold"],
        tokenizer
    )

    ## Transform the prediction into the BioASQ format
    print("***** Transform the prediction file into the BioASQ format {} *****".format(prefix))
    transform_n2b_factoid(output_nbest_file, output_directory)

    ## Evaluate with the BioASQ official evaluation code
    pred_file = os.path.join(output_directory, "BioASQform_BioASQ-answer.json")
    eval_score = eval_bioasq_standard(str(5), pred_file, args["golden_file"], args["official_eval_dir"])

    print("** BioASQ-factoid Evaluation Results ************************************")
    print(f"   S. Accuracy = {float(eval_score[1]) * 100:.2f}")
    print(f"   L. Accuracy = {float(eval_score[2]) * 100:.2f}")
    print(f"   MRR         = {float(eval_score[3]) * 100:.2f}")


def save_model_checkpoint(model, tokenizer, optimizer, scheduler, global_step, args):
    output_dir = os.path.join(output_directory, "checkpoint-{}".format(global_step))
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    print("Saving model checkpoint to", output_dir)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    print("Saving optimizer and scheduler states to", output_dir)



def train(args, train_dataset, model, tokenizer, settings):
    device = torch.device("cpu")
    number_of_epochs, max_steps, training_batch_size, learning_rate,\
    adam_epsilon, checkpoint_period, gradient_acc_steps, weight_decay = settings.unpack_train_settings()

    """ Train the model """
    tensorboard_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=training_batch_size)

    # If we have set an upper bound on the number of steps
    if max_steps > 0:
        t_total = max_steps
        number_of_epochs = max_steps // (len(train_dataloader) // gradient_acc_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_acc_steps * number_of_epochs

    # --- Prepare optimizer and schedule (linear warmup and decay)
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
    if os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path, "scheduler.pt")))

    # Conduct training
    print("-------- Performing Training --------")
    print("Sample Size = {}\n Number of Epochs = {}".format(len(train_dataset), number_of_epochs))
    print("Batch Size (w. parallel, distributed & accumulation) = %d", training_batch_size * gradient_acc_steps)
    print("Gradient Accumulation steps = %d", gradient_acc_steps)
    print("Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(model_name_or_path):
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // gradient_acc_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_acc_steps)

            print("Continuing training from checkpoint, will skip to saved global_step")
            print("Continuing training from epoch %d", epochs_trained)
            print("Continuing training from global step %d", global_step)
            print("Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            print("Beginning fine-tuning.")

    training_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(number_of_epochs), desc="Epoch")

    # Added here for reproducibility
    set_seed(42)

    for _ in train_iterator:
        # Configure the Progress Bar
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

            outputs = model(**inputs)

            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0] / gradient_acc_steps if gradient_acc_steps > 1 else outputs[0]
            loss.backward()

            training_loss += loss.item()
            if (step + 1) % gradient_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if logging_steps > 0 and global_step % logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if evaluate_during_training:
                        results = evaluate(args, model, tokenizer, settings)
                        for key, value in results.items():
                            tensorboard_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tensorboard_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tensorboard_writer.add_scalar("loss", (training_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = training_loss

                # ------- Save model checkpoint --------
                if global_step % checkpoint_period == 0:
                    save_model_checkpoint(model, tokenizer, optimizer, scheduler, global_step, args)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break

        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    tensorboard_writer.close()

    return global_step, training_loss / global_step


def get_args():
    return {
        "train_file": "../datasets/QA/BioASQ/BioASQ-train-factoid-7b.json",
        "max_seq_length": 384,
        "golden_file": None,
        "official_eval_dir": './scripts/bioasq_eval',
        "data_dir": None,
        "predict_file": None,
        "config_name": "",
        "tokenizer_name": "google/electra-small-discriminator", #"",
        "version_2_with_negative": True,
        "null_score_diff_threshold": 0.0,
        "doc_stride": 128,
        "max_query_length": 64,
        "do_lower_case": False,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "n_best_size": 20,
        "max_answer_length": 30,
        "lang_id": 0,
        "logging_steps": 500,
        "eval_all_checkpoints": False,
        "overwrite_output_dir": False,
        "overwrite_cache": False,
    }


def main():

    args = get_args()
    perform_training = True
    perform_evaluation = True
    default_settings = EnvironmentSettings()

    if args["doc_stride"] >= args["max_seq_length"] - args["max_query_length"]:
        print(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(output_directory)
        and os.listdir(output_directory)
        and perform_training
        and not args["overwrite_output_dir"]
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                output_directory
            )
        )

    # ------ SET UP DISTRIBUTED TRAINING -----
    # Setup CUDA, GPU & distributed training
    device = torch.device("cpu")

    # Set seed
    set_seed(42)

    # ------- LOAD PRE-TRAINED MODELS AND TOKENIZER --------

    config = AutoConfig.from_pretrained(
        args["config_name"] if args["config_name"] else model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args["tokenizer_name"] if args["tokenizer_name"] else model_name_or_path,
        do_lower_case=args["do_lower_case"]
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config
    )

    model.to(device)

    print("Training/evaluation parameters", args)

    # Training
    if perform_training:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, 200)))

        global_step, training_loss = train(args, train_dataset, model, tokenizer, default_settings)
        print(" global_step = {}, average loss = {}".format(global_step, training_loss))

    # Save the trained model and the tokenizer
    if perform_training:
        print("Saving model checkpoint to", output_directory)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_directory)
        tokenizer.save_pretrained(output_directory)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_directory, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(output_directory)  # , force_download=True)
        tokenizer = AutoTokenizer.from_pretrained(output_directory, do_lower_case=args["do_lower_case"])
        model.to(device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if perform_evaluation:
        if perform_training:
            print("Loading checkpoints saved during training for evaluation")
            checkpoints = [output_directory]

            if args["eval_all_checkpoints"]:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(output_directory + "/**/" + WEIGHTS_NAME, recursive=True))
                )
        else:
            print("Loading checkpoint {} for evaluation".format(model_name_or_path))
            checkpoints = [model_name_or_path]

        print("Evaluate the following checkpoints:", checkpoints)
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(device)

            # Evaluate
            evaluate(args, model, tokenizer, default_settings, prefix=global_step)


if __name__ == "__main__":
    main()
