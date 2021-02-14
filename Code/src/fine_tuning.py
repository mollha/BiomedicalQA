import argparse
from read_data import dataset_to_fc
from tqdm import tqdm
from models import *
from utils import *
from evaluation import evaluate_factoid, evaluate_yesno, evaluate_list
from data_processing import *
from build_checkpoints import build_finetuned_from_checkpoint
from torch.utils.data import DataLoader, RandomSampler

# ------------- DEFINE TRAINING AND EVALUATION SETTINGS -------------
config = {
    'seed': 0,
    'losses': [],
    'avg_loss': [0, 0],
    'num_workers': 3 if torch.cuda.is_available() else 0,
    "max_epochs": 5,  # can override the val in config
    "current_epoch": 0,  # track the current epoch in config for saving checkpoints
    "steps_trained": 0,  # track the steps trained in config for saving checkpoints
    "global_step": -1,  # total steps over all epochs
    "update_steps": 2000,  # set this really high for now.
    "pretrained_settings": {
        "epochs": 0,
        "steps": 0,
    },
}


def fine_tune(train_dataloader, eval_dataloader_dict, qa_model, scheduler, optimizer, settings, checkpoint_dir):
    qa_model.to(settings["device"])

    print('len train dataloader', len(train_dataloader))  # todo remove these once it makes sense
    print('len test dataloader', len(eval_dataloader_dict))  # todo remove these once it makes sense

    # ------------------ PREPARE TO START THE TRAINING LOOP ------------------
    sys.stderr.write("\n---------- BEGIN FINE-TUNING ----------")
    sys.stderr.write(
        "\nDevice = {}\nModel Size = {}\nTotal Epochs = {}\nStart training from Epoch = {}\n"
        "Start training from Step = {}\nBatch size = {}\nCheckpoint Steps = {}\nMax Sample Length = {}\n\n"
            .format(settings["device"].upper(), settings["size"], settings["max_epochs"], settings["current_epoch"],
                    settings["steps_trained"], settings["batch_size"], settings["update_steps"],
                    settings["max_length"]))

    qa_model.zero_grad()
    set_seed(settings["seed"])  # Added here for reproducibility
    train_iterator = trange(settings["current_epoch"], int(settings["max_epochs"]), desc="Epoch")  # Resume from epoch
    steps_trained = settings["steps_trained"]  # resume training

    for epoch_number in train_iterator:
        step_iterator = tqdm(train_dataloader, desc="Step")  # get a tqdm iterator of the dataloader
        settings["current_epoch"] = epoch_number  # update the epoch number with the current epoch

        for training_step, batch in enumerate(step_iterator):
            # If resuming training from a checkpoint, overlook previously trained steps.
            if steps_trained > 0:
                steps_trained -= 1
                continue  # skip this step

            qa_model.train()  # make sure we are in .train() mode

            if "factoid" in settings["question_type"] or "list" in settings["question_type"]:
                inputs = {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.attention_mask,
                    "token_type_ids": batch.token_type_ids,
                    "start_positions": batch.answer_start,
                    "end_positions": batch.answer_end,
                }
            elif "yesno" in settings["question_type"]:
                inputs = {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.attention_mask,
                    "token_type_ids": batch.token_type_ids,
                    "labels": batch.labels,
                }
                # print('batch labels', batch.labels)
            else:
                raise Exception("Question type list must be contain factoid, list or yesno.")

            outputs = qa_model(**inputs)  # put inputs through the model and get outputs
            loss = outputs[0]  # Collect loss from outputs
            loss.backward()  # back-propagate

            # update the average loss statistics
            settings["avg_loss"][0] += float(loss.item())
            settings["avg_loss"][1] += 1

            nn.utils.clip_grad_norm_(qa_model.parameters(), 1.)

            # --- perform updates ---
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            qa_model.zero_grad()  # zero_grad clears old gradients from the last step

            settings["steps_trained"] = training_step
            settings["global_step"] += 1

            # Save checkpoint every settings["update_steps"] steps
            if settings["global_step"] > 0 and settings["update_steps"] > 0 and settings["global_step"] % settings[
                "update_steps"] == 0:
                sys.stderr.write("\n{} steps trained in current epoch, {} steps trained overall.\n"
                                 .format(settings["steps_trained"], settings["global_step"]))
                save_checkpoint(qa_model, optimizer, scheduler, settings, checkpoint_dir, pre_training=False)

        # TODO our metric results dictionary will be empty if we're evaluating with non-golden bioasq.
        # just look at the statistics at the end of an epoch

        # if "factoid" in settings["question_type"] or "list" in settings["question_type"]:

        sys.stderr.write("\n{} steps trained in current epoch, {} steps trained overall."
                         .format(settings["steps_trained"], settings["global_step"]))

        for eval_dataset_name in eval_dataloader_dict:  # evaluate each of the evaluation datasets
            loader_all_question_types = eval_dataloader_dict[eval_dataset_name]
            sys.stderr.write("\nEvaluating on test-set {}".format(eval_dataset_name))

            for qt in loader_all_question_types:
                eval_dataloader = loader_all_question_types[qt]

                if "factoid" in qt:
                    metric_results = evaluate_factoid(qa_model, eval_dataloader, electra_tokenizer, k, training=True)
                elif "list" in qt:
                    metric_results = {}  # todo do nothing for now - need to add list qs
                elif "yesno" in qt:
                    metric_results = evaluate_yesno(qa_model, eval_dataloader, training=True)
                else:
                    raise Exception("Question type in config must be factoid, list or yesno.")

                sys.stderr.write("\nGathering metrics for {} questions".format(qt))
                if len(metric_results) > 0:
                    # Our metric results dictionary will be empty if we're evaluating with non-golden bioasq.
                    sys.stderr.write("\nCurrent evaluation metrics are {}".format(metric_results))

    # update loss function statistics
    settings["losses"].append(settings["avg_loss"][0] / settings["avg_loss"][1])  # bank stats
    settings["avg_loss"] = [0, 0]  # reset stats
    # ------------- SAVE FINE-TUNED MODEL -------------
    save_checkpoint(qa_model, optimizer, scheduler, settings, checkpoint_dir, pre_training=False)


if __name__ == "__main__":
    # Log the process ID
    print(f"Process ID: {os.getpid()}\n")

    # -- Parse command line arguments (checkpoint name and model size) ---
    parser = argparse.ArgumentParser(description='Overwrite default fine-tuning settings.')
    parser.add_argument("--size", default="small", choices=['small', 'base', 'large'], type=str,
                        help="The size of the electra model e.g. 'small', 'base' or 'large")
    parser.add_argument("--p-checkpoint", default="recent", type=str,
                        help="The name of the pre-training checkpoint to use e.g. small_15_10230.")
    parser.add_argument("--f-checkpoint", default="", type=str,
                        help="The name of the fine-tuning checkpoint to use e.g. small_factoid_15_10230_2_30487")
    parser.add_argument("--question-type", default=["factoid", "list"], choices=[['factoid'], ['yesno'], ['list'], ["factoid", "list"]], type=list,
                        help="Types supported by fine-tuned model - e.g. choose one of [['factoid'], ['yesno'], ['list'], ['factoid', 'list']]")
    parser.add_argument("--dataset", default="bioasq", choices=['squad', 'bioasq'], type=str,
                        help="The name of the dataset to use in training e.g. squad")
    parser.add_argument("--k", default="5", type=int,
                        help="K-best predictions are selected for factoid and list questions (between 1 and 100)")

    args = parser.parse_args()
    config['size'] = args.size
    config["question_type"] = args.question_type
    args.f_checkpoint = args.f_checkpoint if args.f_checkpoint != "empty" else ""  # deals with slurm script
    args.question_type = ".".join(args.question_type)

    sys.stderr.write("\n--- ARGUMENTS ---")
    sys.stderr.write(
        "\nPre-training checkpoint: {}\nFine-tuning checkpoint: {}\nModel Size: {}\nQuestion Type: {}\nDataset: {}\nK: {}"
        .format(args.p_checkpoint, args.f_checkpoint, args.size, args.question_type, args.dataset, args.k))

    # ------- Check the validity of the arguments passed via command line -------
    try:  # check that the value of k is actually a number
        k = int(args.k)
        if k < 1 or k > 100:  # check that k is at least 1 and at most 100
            raise ValueError
    except ValueError:
        raise Exception("k must be an integer between 1 and 100. Got {}".format(args.k))

    if args.f_checkpoint != "" and args.f_checkpoint != "recent":
        if args.size not in args.f_checkpoint:
            raise Exception(
                "If using a fine-tuned checkpoint, the model size of the checkpoint must match provided model size."
                "e.g. --f-checkpoint small_factoid_15_10230_12_20420 --size small")
        if args.question_type not in args.f_checkpoint:
            raise Exception(
                "If using a fine-tuned checkpoint, the question type of the checkpoint must match question type."
                "e.g. --f-checkpoint small_factoid_15_10230_12_20420 --question-type factoid")
    elif args.p_checkpoint != "recent" and args.size not in args.p_checkpoint:
        raise Exception(
            "If not using the most recent checkpoint, the model size of the checkpoint must match model size."
            "e.g. --p-checkpoint small_15_10230 --size small")

    # ---- Set torch backend and set seed ----
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    set_seed(config["seed"])  # set seed for reproducibility

    # ---- Override general config with model specific config, for models of different sizes ----
    model_specific_config = get_model_config(config['size'], pretrain=False)
    config = {**model_specific_config, **config}

    # ---- Find path to checkpoint directory - create the directory if it doesn't exist ----
    base_path = Path(__file__).parent
    checkpoint_name = (args.p_checkpoint, args.f_checkpoint)

    selected_dataset = args.dataset.lower()
    base_checkpoint_dir = (base_path / '../checkpoints').resolve()
    pretrain_checkpoint_dir = (base_checkpoint_dir / 'pretrain').resolve()
    finetune_checkpoint_dir = (base_checkpoint_dir / 'finetune').resolve()

    # create the fine-tune directory if it doesn't exist already
    Path(finetune_checkpoint_dir).mkdir(exist_ok=True, parents=True)  # create the fine-tune directory

    # -- Set device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    sys.stderr.write("\nDevice: {}\n".format(config["device"].upper()))

    # get the tokenizer to prepare the data
    _, _, electra_tokenizer, _ = build_electra_model(config['size'], get_config=True)

    # -- Load the data and prepare it in squad format
    dataset_function = dataset_to_fc[selected_dataset]

    # ---- Find the path(s) to the dataset file(s) ----
    dataset_dir = (base_checkpoint_dir / '../datasets').resolve()
    train_dataset_file_paths = [(dataset_dir / (selected_dataset + "/" + d_path)).resolve() for d_path in
                                datasets[selected_dataset]["train"]]
    test_dataset_file_paths = [(dataset_dir / (selected_dataset + "/" + d_path)).resolve() for d_path in
                               datasets[selected_dataset]["test"]]

    sys.stderr.write("\nTraining files are '{}'\nEvaluation files are '{}'"
                     .format(datasets[selected_dataset]["train"], datasets[selected_dataset]["test"]))

    # ----- PREPARE THE TRAINING DATASET -----
    sys.stderr.write("\nReading raw train dataset for '{}'".format(selected_dataset))
    raw_train_dataset = dataset_function(train_dataset_file_paths)

    # combine the features from these datasets.
    train_features = []
    for qt in config["question_type"]:  # iterate through our chosen question types
        raw_train_dataset_by_question = raw_train_dataset[qt]
        sub_train_features = convert_examples_to_features(raw_train_dataset_by_question, electra_tokenizer,
                                                          config["max_length"])
        train_features.extend(sub_train_features)

    print("Created {} train features of length {}.".format(len(train_features), config["max_length"]))
    train_dataset = QADataset(train_features)

    # Random Sampler used during training.
    # We create a single data_loader for training.
    train_data_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=config["batch_size"],
                                   collate_fn=collate_wrapper)

    # ----- PREPARE THE EVALUATION DATASET -----
    sys.stderr.write("\nReading raw test dataset(s) for '{}'".format(selected_dataset))

    # Once populated, the test_data_loader_dict will contain {"dataset_path": {"factoid": dataloader, "list": datalo...}
    test_data_loader_dict = {}

    for test_dataset_file_path in test_dataset_file_paths:  # iterate over each path separately.
        test_data_loader_dict[test_dataset_file_path] = {}  # initialise empty dict for this dataset path

        raw_test_dataset = dataset_function([test_dataset_file_path], testing=True)

        for qt in config["question_type"]:
            raw_test_dataset_by_question = raw_test_dataset[qt]  # how do we also get list and factoid together?
            test_features = convert_examples_to_features(raw_test_dataset_by_question, electra_tokenizer, config["max_length"])

            print("Created {} test features of length {} from {} questions.".format(len(test_features), config["max_length"], qt))
            test_dataset = QADataset(test_features)

            # Create a dataloader for each of the test datasets.
            test_data_loader = DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=collate_wrapper)
            test_data_loader_dict[test_dataset_file_path][qt] = test_data_loader

    config["num_warmup_steps"] = len(train_data_loader) // config["max_epochs"]
    electra_for_qa, optimizer, scheduler, electra_tokenizer, \
    config = build_finetuned_from_checkpoint(config["size"], config["device"], pretrain_checkpoint_dir,
                                             finetune_checkpoint_dir, checkpoint_name, config["question_type"], config)

    print(config)

    # ------ START THE FINE-TUNING LOOP ------
    fine_tune(train_data_loader, test_data_loader_dict, electra_for_qa, scheduler, optimizer, config,
              finetune_checkpoint_dir)
