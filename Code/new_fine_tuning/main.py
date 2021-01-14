import os
from pathlib import Path


config = {
    "batch_size": 8,
    "epochs": 1,
    "learning_rate": 8e-6,  # The initial learning rate for Adam.
    "decay": 0.0,  # Weight decay if we apply some.
    "epsilon": 1e-8,  # Epsilon for Adam optimizer.
    "max_grad_norm": 1.0,  # Max gradient norm.
    "evaluate_all_checkpoints": False,
    "update_steps": 500,
    "size": "small"
}


# ---------- DEFINE MAIN FINE-TUNING LOOP ----------
def fine_tune(finetuning_dataset, model, scheduler, optimizer, settings, checkpoint_name="recent"):
    checkpoint_dir = (base_path / 'checkpoints/fine_tune').resolve()
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)



# def train(train_dataset, model, tokenizer, model_info, device, save_dir, settings, dataset_info):
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

            if "electra" in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if "electra" in ["xlnet", "xlm"]:
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
                        results = evaluate(model, tokenizer, "electra", save_dir, device, settings["evaluate_all_checkpoints"], dataset_info)
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


if __name__ == "__main__":
    # Log Process ID
    print(f"Process ID: {os.getpid()}\n")

    base_path = Path(__file__).parent


    # # Override general config with model specific config, for models of different sizes
    # model_specific_config = get_model_config(config['size'])
    # config = {**model_specific_config, **config}

    # Set torch backend and set seed
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])


    generator, discriminator, electra_tokenizer = build_electra_model(config['size'])
    electra_model = ELECTRAModel(generator, discriminator, electra_tokenizer)
