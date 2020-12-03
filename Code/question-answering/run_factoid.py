import os
from time import time
from torch import save, load, ones, int64, nn, no_grad
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING, AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadResult
from utils_qa import transform_n2b_factoid, eval_bioasq_standard, to_list
from torch.utils.tensorboard import SummaryWriter


""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Save checkpoint and log every X updates steps.
def save_model(model, tokenizer, optimizer, scheduler, settings, global_step, tr_loss, save_dir):

    # ------------- SAVE FINE-TUNED TOKENIZER AND MODEL -------------
    save_dir = os.path.join(save_dir, "checkpoint-{}".format(global_step))

    # Take care of distributed/parallel training
    saving_model = model.module if hasattr(model, "module") else model
    saving_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # save training settings with trained model
    save(settings, os.path.join(save_dir, "train_settings.bin"))
    save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    print("Saving model checkpoint, optimizer and scheduler states to {}".format(save_dir))
    print("global_step = {}, average loss = {}".format(global_step, tr_loss))


def train(train_dataset, model, tokenizer, model_info, device, save_dir, settings, dataset_info):
    """ Train the model """

    version_2_with_negative = False
    overwrite_output_directory = False

    if os.path.exists(save_dir) and os.listdir(save_dir) and not overwrite_output_directory:
        raise ValueError(
            "Output directory ({}) already exists. Set overwrite_output_directory to True to overcome.".format(
                save_dir
            )
        )

    print(train_dataset)

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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(data_loader) // settings["epochs"])

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(model_info["model_path"], "optimizer.pt")) and os.path.isfile(
            os.path.join(model_info["model_path"], "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(load(os.path.join(model_info["model_path"], "optimizer.pt")))
        scheduler.load_state_dict(load(os.path.join(model_info["model_path"], "scheduler.pt")))

    print("\n---------- BEGIN TRAINING ----------")
    print("Dataset Size = {}\nNumber of Epochs = {}\nBatch size = {}\n"
          .format(len(train_dataset), settings["epochs"], settings["batch_size"]))

    global_step, epochs_trained = 1, 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(model_info["model_path"]):
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = model_info["model_path"].split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(data_loader))
            steps_trained_in_current_epoch = global_step % (len(data_loader))

            print("Continuing training from checkpoint, will skip to saved global_step")
            print("Continuing training from epoch {} and global step {}".format(epochs_trained, global_step))
            print("Skip the first {} steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            print("Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(settings["epochs"]), desc="Epoch")

    # Added here for reproducibility
    # TODO SET SEED HERE AGAIN

    tb_writer = SummaryWriter()  # Create a SummaryWriter()
    for epoch_number in train_iterator:
        epoch_iterator = tqdm(data_loader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            print(type(batch))
            print("batch size", len(batch))

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # train model one step
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
                        {"langs": (ones(batch[0].shape, dtype=int64) * 0).to(device)}
                    )

            outputs = model(**inputs)

            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            loss.backward()

            tr_loss += loss.item()

            if (step + 1) % 1 == 0:
                nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
                global_step += 1

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                # Log metrics
                if settings["update_steps"] > 0 and global_step % settings["update_steps"] == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    # Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
                    if settings["evaluate_all_checkpoints"]:
                        results = evaluate(model, tokenizer, model_info["model_type"], save_dir, device, settings["evaluate_all_checkpoints"], dataset_info)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / settings["update_steps"], global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if settings["update_steps"] > 0 and global_step % settings["update_steps"] == 0:
                    save_model(model, tokenizer, optimizer, scheduler, settings, global_step, tr_loss / global_step, save_dir)

    tb_writer.close()

    # ------------- SAVE FINE-TUNED MODEL -------------
    save_model(model, tokenizer, optimizer, scheduler, settings, global_step, tr_loss / global_step, save_dir)


def evaluate(model, tokenizer, model_type, save_dir, device, test_set, examples, features, dataset_info, eval_settings, prefix=""):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Sequential Sampler used during evaluation.
    data_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=eval_settings["batch_size"])

    print("----- Running evaluation {} -----".format(prefix))
    print("Num examples = {}\nBatch size = {}\n".format(len(test_set), eval_settings["batch_size"]))

    all_results = []
    start_time = time()

    for batch in tqdm(data_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with no_grad():
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
                        {"langs": (ones(batch[0].shape, dtype=int64) * 0).to(device)}
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

    eval_time = time() - start_time
    print("Evaluation done in total %f secs (%f sec per example)".format(eval_time, eval_time / len(test_set)))

    # Compute predictions
    output_prediction_file = os.path.join(save_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(save_dir, "nbest_predictions_{}.json".format(prefix))

    output_null_log_odds_file = None
    if eval_settings["version_2_with_negative"]:
        output_null_log_odds_file = os.path.join(save_dir, "null_odds_{}.json".format(prefix))


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
    transform_n2b_factoid(output_nbest_file, save_dir)

    # Evaluate with the BioASQ official evaluation code
    pred_file = os.path.join(save_dir, "BioASQform_BioASQ-answer.json")
    eval_score = eval_bioasq_standard(str(5), pred_file, dataset_info["golden_file"], dataset_info["official_eval_dir"])

    print("** BioASQ-factoid Evaluation Results ************************************")
    print(f"   S. Accuracy = {float(eval_score[1]) * 100:.2f}")
    print(f"   L. Accuracy = {float(eval_score[2]) * 100:.2f}")
    print(f"   MRR         = {float(eval_score[3]) * 100:.2f}")
