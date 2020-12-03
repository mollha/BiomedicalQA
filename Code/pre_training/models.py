import torch
import os
from pathlib import Path
from torch import nn
import torch.nn.functional
import datetime

# ------------------ DEFINE CONFIG FOR ELECTRA MODELS AS SPECIFIED IN PAPER ------------------
# i.e. Vanilla ELECTRA Model settings are outlined in the paper: https://arxiv.org/abs/2003.10555

small_config = {
    "mask_prob": 0.15,
    "lr": 5e-4,
    "batch_size": 128,
    "max_steps": 10 ** 6,
    "max_length": 128,
    "generator_size_divisor": 4
}

base_config = {
    "mask_prob": 0.15,
    "lr": 2e-4,
    "batch_size": 256,
    "max_steps": 766 * 1000,
    "max_length": 512,
    "generator_size_divisor": 3
}

large_config = {
    "mask_prob": 0.25,
    "lr": 2e-4,
    "batch_size": 2048,
    "max_steps": 400 * 1000,
    "max_length": 512,
    "generator_size_divisor": 4
}


def get_model_config(model_size: str) -> dict:
    """
    Given a model size (e.g. small, base or large), return a dictionary containing the vanilla
    ELECTRA settings for this model, as outlined in the ELECTRA paper.

    Link to paper - https://arxiv.org/abs/2003.10555.

    :param model_size: a string representing the size of the model
    :return: a dictionary containing the model config
    """
    assert (model_size in ["small", "base", "large"])

    index = ['small', 'base', 'large'].index(model_size)
    return [small_config, base_config, large_config][index]


# ------------------ LOAD AND SAVE MODEL CHECKPOINTS ------------------
def load_checkpoint(path_to_checkpoint: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler) -> tuple:
    """
    Given a path to a checkpoint directory, the model, optimizer, scheduler and training settings
    are loaded from this directory, ready to continue pre-training.

    The structure of the checkpoint directory is expected to be as follows:
    ----> path/to/checkpoint/
                ﹂model.pt
                ﹂optimizer.pt
                ﹂scheduler.pt
                ﹂train_settings.bin

    :param path_to_checkpoint: a string pointing to the location of the directory containing a model checkpoint
    :param model: model skeleton to be populated with saved (pre-trained) model state
    :param optimizer: optimizer skeleton to be populated with saved (pre-trained) optimizer state
    :param scheduler: scheduler skeleton to be populated with saved (pre-trained) scheduler state
    :return: model, optimizer and scheduler in their pre-trained states and previous model settings
    """
    # Load in optimizer, tokenizer and scheduler states
    path_to_optimizer = os.path.join(path_to_checkpoint, "optimizer.pt")
    if os.path.isfile(path_to_optimizer):
        optimizer.load_state_dict(torch.load(path_to_optimizer))

    path_to_scheduler = os.path.join(path_to_checkpoint, "scheduler.pt")
    if os.path.isfile(path_to_scheduler):
        scheduler.load_state_dict(torch.load(path_to_scheduler))

    path_to_model = os.path.join(path_to_checkpoint, "model.pt")
    if os.path.isfile(path_to_model):
        model.load_state_dict(torch.load(path_to_model))

    settings = torch.load(os.path.join(path_to_checkpoint, "train_settings.bin"))

    print("Re-instating settings from model saved on {} at {}.".format(settings["saved_on"], settings["saved_at"]))
    print("Resuming training from epoch {} and step: {}\n"
          .format(settings["current_epoch"], settings["steps_trained"]))

    return model, optimizer, scheduler, settings


def save_checkpoint(model, optimizer, scheduler, settings, checkpoint_dir):
    now = datetime.datetime.now()
    today = datetime.date.today()

    # WE NEED TO CREATE A NEW CHECKPOINT NAME
    checkpoint_name = "{}_{}_{}".format(settings["size"], settings["current_epoch"], settings["steps_trained"])

    # save the time and date of when the model was last saved
    settings["saved_on"] = today.strftime("%d/%m/%y")
    settings["saved_at"] = now.strftime("%H:%M:%S")

    # ------------- save pre-trained optimizer, scheduler and model -------------
    save_dir = os.path.join(checkpoint_dir, checkpoint_name)
    Path(save_dir).mkdir(exist_ok=False, parents=True)

    # save training settings with trained model
    torch.save(settings, os.path.join(save_dir, "train_settings.bin"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    # the tokenizer state is saved with the model

    print("Saving model checkpoint, optimizer and scheduler states to {}".format(save_dir))


# --------------------------------------------------------------------------------


class ELECTRAModel(nn.Module):

    def __init__(self, generator, discriminator, electra_tokenizer):
        super().__init__()
        self.generator, self.discriminator = generator, discriminator
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.)
        self.electra_tokenizer = electra_tokenizer

    def to(self, *args, **kwargs):
        " Also set dtype and device of contained gumbel distribution if needed. "
        super().to(*args, **kwargs)
        a_tensor = next(self.parameters())
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=a_tensor.device,
                                                                          dtype=a_tensor.dtype),
                                                             torch.tensor(1., device=a_tensor.device,
                                                                          dtype=a_tensor.dtype))

    def forward(self, masked_inputs, is_mlm_applied, labels):
        """
        masked_inputs (Tensor[int]): (B, L)
        is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability
        labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
        """
        # this assumes that padding has taken place ALREADY
        # the attention mask is the tensortext of true and false
        # the token_type_ids is the tensor of zeros and ones

        attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs)

        gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[0]  # (B, L, vocab size)
        # reduce size to save space and speed
        mlm_gen_logits = gen_logits[is_mlm_applied, :]  # ( #mlm_positions, vocab_size)

        with torch.no_grad():
            pred_toks = self.sample(mlm_gen_logits)  # ( #mlm_positions, )

            # produce inputs for discriminator
            generated = masked_inputs.clone()  # (B,L)
            generated[is_mlm_applied] = pred_toks  # (B,L)

            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone()  # (B,L)
            is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied])  # (B,L)

        disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0]  # (B, L)

        return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

    def _get_pad_mask_and_token_type(self, input_ids):
        """
        Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask
        and token_type_ids and won't be unnecessarily large,
        thus, preventing cpu processes loading batches from consuming lots of cpu memory and slow down the machine.
        """

        # get a TensorText object (like a list) of booleans indicating whether each
        # input token is a pad token
        attention_mask = input_ids != self.electra_tokenizer.pad_token_id

        # create a list of lists containing zeros and ones.
        # for every boolean in the attention mask, replace this with 0 if positive and 1 if negative.
        token_type_ids = torch.tensor([[int(not boolean) for boolean in sub_mask] for sub_mask in iter(attention_mask)],
                                      device=input_ids.device)
        return attention_mask, token_type_ids

    def sample(self, logits):
        """
        Implement gumbel softmax as there is a bug in torch.nn.functional.gumbel_softmax when fp16
        (https://github.com/pytorch/pytorch/issues/41663) is used.

        Gumbel softmax is equal to what the official ELECTRA code does
        i.e. standard gumbel dist. = -ln(-ln(standard uniform dist.))

        :param logits:
        :return:
        """
        return (logits.float() + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)