import os
import timeit
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING, AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadResult
from utils_qa import transform_n2b_factoid, eval_bioasq_standard, to_list

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Save checkpoint and log every X updates steps.
update_steps = 500


def train(train_dataset, model, tokenizer, model_info, device, output_dir, settings, dataset_info):
    """ Train the model """

    # Create a SummaryWriter()
    tb_writer = SummaryWriter()

    version_2_with_negative = False
    overwrite_output_directory = True

    if (os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite_output_directory):
        raise ValueError(
            "Output directory ({}) already exists. Set overwrite_output_directory to True to overcome.".format(
                output_dir
            )
        )


    # Random Sampler used during training.
    data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=settings["batch_size"])

    # Prepare optimizer and schedule (linear warm up and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": settings["decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=settings["epsilon"], lr=settings["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=len(data_loader) // settings["epochs"])

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(model_info["model_path"], "optimizer.pt")) and os.path.isfile(
            os.path.join(model_info["model_path"], "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_info["model_path"], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_info["model_path"], "scheduler.pt")))

    print("---------- BEGIN TRAINING ----------")
    print("Dataset Size = {}\nNumber of Epochs = {}".format(len(train_dataset), settings["epochs"]))
    print("Batch size = {}\n".format(settings["batch_size"]))

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(model_info["model_path"]):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = model_info["model_path"].split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(data_loader))
            steps_trained_in_current_epoch = global_step % (len(data_loader))

            print("Continuing training from checkpoint, will skip to saved global_step")
            print("Continuing training from epoch %d", epochs_trained)
            print("Continuing training from global step %d", global_step)
            print("Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            print("Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(settings["epochs"]), desc="Epoch")

    # Added here for reproducibility
    # TODO SET SEED HERE AGAIN

    for _ in train_iterator:
        epoch_iterator = tqdm(data_loader, desc="Iteration")
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

            if model_info["model_type"] in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if model_info["model_type"] in ["xlnet", "xlm"]:
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
                global_step += 1

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                # Log metrics
                if update_steps > 0 and global_step % update_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    # Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
                    if settings["evaluate_all_checkpoints"]:
                        results = evaluate(model, tokenizer, model_info["model_type"], output_dir, device, settings["evaluate_all_checkpoints"], dataset_info)
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
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    print("Saving model checkpoint, optimizer and scheduler states to {}".format(output_dir))

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(model, tokenizer, model_type, output_dir, device, test_set, examples, features, dataset_info, eval_settings, prefix=""):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sequential Sampler used during evaluation.
    data_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=eval_settings["batch_size"])

    print("----- Running evaluation {} -----".format(prefix))
    print("Num examples = {}\nBatch size = {}\n".format(len(test_set), eval_settings["batch_size"]))

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(data_loader, desc="Evaluating"):
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

    eval_time = timeit.default_timer() - start_time
    print("Evaluation done in total %f secs (%f sec per example)".format(eval_time, eval_time / len(test_set)))

    # Compute predictions
    output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))

    if eval_settings["version_2_with_negative"]:
        output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    compute_predictions_logits(
        examples,
        features,
        all_results,
        eval_settings["n_best_size"],
        eval_settings["max_answer_length"],
        uncased_model,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,
        eval_settings["version_2_with_negative"],
        0.0,
        tokenizer,
    )

    # Transform the prediction into the BioASQ format
    print("----- Transform the prediction file into the BioASQ format {} -----".format(prefix))
    transform_n2b_factoid(output_nbest_file, output_dir)

    # Evaluate with the BioASQ official evaluation code
    pred_file = os.path.join(output_dir, "BioASQform_BioASQ-answer.json")
    eval_score = eval_bioasq_standard(str(5), pred_file, dataset_info["golden_file"], dataset_info["official_eval_dir"])

    print("** BioASQ-factoid Evaluation Results ************************************")
    print(f"   S. Accuracy = {float(eval_score[1]) * 100:.2f}")
    print(f"   L. Accuracy = {float(eval_score[2]) * 100:.2f}")
    print(f"   MRR         = {float(eval_score[3]) * 100:.2f}")
