import os

# ------------------ SPECIFY GENERAL MODEL CONFIG ------------------
config = {
    'seed': 0,
    "eval_batch_size": 12,
    "n_best_size": 20,  # The total number of n-best predictions to generate in the nbest_predictions.json output file.
    "max_answer_length": 30,  # maximum length of a generated answer
    "version_2_with_negative": False,  # If true, the SQuAD examples contain some that do not have an answer.
}

def evaluate(finetuned_model, test_dataset):


    metrics = {

    }

    return metrics


if __name__ == "__main__":

    # get finetuned model
    finetuned_checkpoint = {}

    # get test dataset
    test_dataset = {}

    # what happens if we start from a checkpoint here (vs passing a checkpoint from fine-tuning)
    evaluate(finetuned_checkpoint, test_dataset)