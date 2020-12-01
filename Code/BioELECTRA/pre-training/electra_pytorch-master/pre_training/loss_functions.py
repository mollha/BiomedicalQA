from fastai.losses import CrossEntropyLossFlat
import torch.nn as nn


class ELECTRALoss:
    """
    Generator loss function: Cross-Entropy loss flat
    Discriminator loss function: Binary Cross-Entropy with Logits
    """

    def __init__(self, loss_weights=(1.0, 50.0)):
        self.loss_weights = loss_weights
        self.generator_loss_function = CrossEntropyLossFlat()
        self.discriminator_loss_function = nn.BCEWithLogitsLoss()

    def __call__(self, pred, targ_ids):
        # model outputs (i.e. pred) are always tuple in transformers (see doc)
        mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
        disc_logits = disc_logits.masked_select(non_pad)  # -> 1d tensor
        is_replaced = is_replaced.masked_select(non_pad)  # -> 1d tensor

        gen_loss = self.generator_loss_function(mlm_gen_logits.float(), targ_ids[is_mlm_applied])
        disc_loss = self.discriminator_loss_function(disc_logits.float(), is_replaced.float())

        return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]

