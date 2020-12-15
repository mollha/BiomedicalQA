from fastai.losses import CrossEntropyLossFlat
import torch.nn as nn
import torch


def confusion_matrix(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)

    Attribution: https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


class ELECTRALoss:
    """
    Generator loss function: Cross-Entropy loss flat
    Discriminator loss function: Binary Cross-Entropy with Logits
    """

    def __init__(self, loss_weights=(1.0, 50.0)):
        self.loss_weights = loss_weights
        self.generator_loss_function = CrossEntropyLossFlat()
        self.discriminator_loss_function = nn.BCEWithLogitsLoss()

        self.mid_epoch_stats = {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "avg_disc_loss": [0, 0],
            "avg_gen_loss": [0, 0],
            "avg_combined_loss": [0, 0]
        }

        self.generator_losses = []
        self.discriminator_losses = []
        self.combined_losses = []
        self.discriminator_accuracy = []
        self.discriminator_precision = []
        self.discriminator_recall = []

        self.sigmoid_fc = nn.Sigmoid()

    def __call__(self, pred, targ_ids):
        # model outputs (i.e. pred) are always tuple in transformers (see doc)
        mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
        disc_logits = disc_logits.masked_select(non_pad)  # -> 1d tensor
        is_replaced = is_replaced.masked_select(non_pad)  # -> 1d tensor

        # Logits is the tensor that is being mapped to probabilities by the Softmax

        # calculate discriminator accuracy
        # get tensor of correct and incorrect answers
        disc_predictions = torch.round(self.sigmoid_fc(disc_logits.float()))

        true_positives, false_positives, \
        true_negatives, false_negatives = confusion_matrix(disc_predictions, is_replaced.float())

        gen_loss = self.generator_loss_function(mlm_gen_logits.float(), targ_ids[is_mlm_applied])
        disc_loss = self.discriminator_loss_function(disc_logits.float(), is_replaced.float())

        generator_loss = gen_loss * self.loss_weights[0]
        discriminator_loss = disc_loss * self.loss_weights[1]

        # Update mid_epoch statistics
        self.mid_epoch_stats["avg_disc_loss"][0] += float(disc_loss.data)
        self.mid_epoch_stats["avg_disc_loss"][1] += 1

        self.mid_epoch_stats["avg_gen_loss"][0] += float(gen_loss.data)
        self.mid_epoch_stats["avg_gen_loss"][1] += 1

        self.mid_epoch_stats["avg_combined_loss"][0] += float(generator_loss.data) + float(discriminator_loss.data)
        self.mid_epoch_stats["avg_combined_loss"][1] += 1

        self.mid_epoch_stats["true_positives"] += true_positives
        self.mid_epoch_stats["false_positives"] += false_positives
        self.mid_epoch_stats["true_negatives"] += true_negatives
        self.mid_epoch_stats["false_negatives"] += false_negatives

        return generator_loss + discriminator_loss

    def update_statistics(self):
        """
        Collate mid-epoch statistics to summarise statistics over the last epoch.
        :return: None
        """
        avg_disc_loss = self.mid_epoch_stats["avg_disc_loss"]
        avg_gen_loss = self.mid_epoch_stats["avg_gen_loss"]
        avg_combined_loss = self.mid_epoch_stats["avg_combined_loss"]

        true_positives = self.mid_epoch_stats["true_positives"]
        false_positives = self.mid_epoch_stats["false_positives"]
        true_negatives = self.mid_epoch_stats["true_negatives"]
        false_negatives = self.mid_epoch_stats["false_negatives"]

        self.generator_losses.append(avg_gen_loss[0] / avg_gen_loss[1])
        self.discriminator_losses.append(avg_disc_loss[0] / avg_disc_loss[1])
        self.combined_losses.append(avg_combined_loss[0] / avg_combined_loss[1])

        self.discriminator_accuracy.append((true_positives + false_positives) /
                                           (true_positives + false_positives + true_negatives + false_negatives))
        self.discriminator_precision.append(true_positives / (true_positives + false_positives))
        self.discriminator_recall.append(true_positives / (true_positives + false_negatives))

        # reset epoch-level stats
        self.mid_epoch_stats = {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "avg_disc_loss": [0, 0],
            "avg_gen_loss": [0, 0],
            "avg_combined_loss": [0, 0]
        }

    def get_statistics(self):
        return self.combined_losses, self.generator_losses, self.discriminator_losses, self.discriminator_accuracy, \
               self.discriminator_precision, self.discriminator_recall
