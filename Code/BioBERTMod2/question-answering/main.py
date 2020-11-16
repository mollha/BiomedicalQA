from torch import cuda, device, manual_seed, save, load
from os import path
import logging
from random import seed
from numpy import random
from glob import glob

from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)

from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor
from run_factoid import train, evaluate


# Create logger for detailed logging.
logger = logging.getLogger(__name__)
logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )


def set_seed(seed_value, use_cuda):
    if use_cuda:
        cuda.manual_seed_all(seed_value)

    seed(seed_value)
    random.seed(seed_value)
    manual_seed(seed_value)


def load_pretrained_model_tokenizer(config_name, cache_dir, model_path, uncased_model, device):

    # Load pre-trained model and tokenizer

    config = AutoConfig.from_pretrained(
        config_name if config_name else model_path,
        cache_dir=cache_dir if cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, # model_path same as tokenizer name
        do_lower_case=uncased_model,
        cache_dir=cache_dir if cache_dir else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None,
    )

    model.to(device)
    return model, tokenizer


def load_and_cache_examples(tokenizer, model_path, train_file, evaluate=False, output_examples=False):
    overwrite_cached_features = True
    max_seq_length = 384 # The maximum total input sequence length after WordPiece tokenization. Sequences " "longer than this will be truncated, and sequences shorter than this will be padded."
    predict_file = "gdrive/My Drive/BioBERT/datasets/QA/BioASQ/BioASQ-test-factoid-7b.json" # "..datasets/QA/BioASQ/BioASQ-test-factoid-7b.json" #
    version_2_with_negative = False
    threads = 1 # "multiple threads for converting example to features
    doc_stride = 128 # When splitting up a long document into chunks, how much stride to take between chunks
    max_query_length = 64 # The maximum number of tokens for the question. Questions longer than this will " "be truncated to this length.
    data_dir = None # The input data dir. Should contain the .json files for the task. If no data dir or train/predict files are specified, will run with tensorflow_datasets."

    # Load data features from cache or dataset file
    input_dir = data_dir if data_dir else "."
    cached_features_file = path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_path.split("/"))).pop(),
            str(max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if path.exists(cached_features_file) and not overwrite_cached_features:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not data_dir and ((evaluate and not predict_file) or (not evaluate and not train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(data_dir, filename=predict_file)
            else:
                examples = processor.get_train_examples(data_dir, filename=train_file)


        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=threads,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)


    if output_examples:
        return dataset, examples, features
    return dataset




if __name__ == "__main__":
    # output_dir: The output directory where the model checkpoints and predictions will be written.
    doc_stride = 128 # When splitting up a long document into chunks, how much stride to take between chunks
    max_query_length = 64 # The maximum number of tokens for the question. Questions longer than this will " "be truncated to this length.
    max_seq_length = 384 # The maximum total input sequence length after WordPiece tokenization. Sequences " "longer than this will be truncated, and sequences shorter than this will be padded."
    uncased_model = False # Set this flag if you are using an uncased model.
    config_name = "" # Pretrained config name or path if not the same as model_name
    cache_dir = "" # Where do you want to store the pre-trained models downloaded from s3
    train_file = "gdrive/My Drive/BioBERT/datasets/QA/BioASQ/BioASQ-train-factoid-7b.json" # "../datasets/QA/BioASQ/BioASQ-train-factoid-7b.json" #


    # DECIDE WHETHER TO TRAIN, EVALUATE, OR BOTH.
    train_model = True
    evaluate_model = True

    model_type = "bert"  # Ensure that lowercase model is used
    train_model_path = "dmis-lab/biobert-base-cased-v1.1"
    output_dir = "gdrive/My Drive/BioBERT/question-answering/output" # "./output"

    # Setup CUDA, GPU & distributed training
    device = device("cuda" if cuda.is_available() else "cpu")
    gpu_available = bool(cuda.device_count() > 0)
    print("Device: {}, GPU available: {}".format(device, gpu_available))

    if doc_stride >= max_seq_length - max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    set_seed(0, gpu_available)  # fix seed for reproducibility

    model, tokenizer = load_pretrained_model_tokenizer(config_name, cache_dir, train_model_path, uncased_model, device)
    evaluate_all_checkpoints = False

    # Training
    if train_model:
        train_dataset = load_and_cache_examples(tokenizer, train_model_path, train_file, evaluate=False, output_examples=False)

        global_step, tr_loss = train(train_dataset, model, tokenizer, model_type, train_model_path, device,
                                     evaluate_all_checkpoints, output_dir)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save the trained model and the tokenizer

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        training_arguments = {}
        # Good practice: save your training arguments together with the trained model
        save(training_arguments, path.join(output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(output_dir)  # , force_download=True)
        tokenizer = AutoTokenizer.from_pretrained(output_dir, do_lower_case=uncased_model)
        model.to(device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}

    if evaluate_model:
        if train_model:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [output_dir]
            if evaluate_all_checkpoints:
                checkpoints = list(
                    path.dirname(c)
                    for c in sorted(glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", output_dir)
            checkpoints = [output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(device)

            dataset, examples, features = load_and_cache_examples(tokenizer, output_dir, train_file, evaluate=True,
                                                                  output_examples=True)

            # Evaluate
            evaluate(model, tokenizer, model_type, output_dir, device, dataset, examples, features, prefix=global_step)
