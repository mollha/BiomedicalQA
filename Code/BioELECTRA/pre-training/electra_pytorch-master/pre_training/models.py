import torch
from torch import nn
import torch.nn.functional

""" Vanilla ELECTRA settings
    """

small_config = {
        "mask_prob": 0.15,
        "lr": 5e-4,
        "batch_size": 128,
        "steps": 10 ** 6,
        "max_length": 128,
        "generator_size_divisor": 4
    }

base_config = {
        "mask_prob": 0.15,
        "lr": 2e-4,
        "batch_size": 256,
        "steps": 766 * 1000,
        "max_length": 512,
        "generator_size_divisor": 3
    }

large_config = {
        "mask_prob": 0.25,
        "lr": 2e-4,
        "batch_size": 2048,
        "steps": 400 * 1000,
        "max_length": 512,
        "generator_size_divisor": 4
    }


def get_model_config(model_size):
    assert (model_size in ["small", "base", "large"])

    index = ['small', 'base', 'large'].index(model_size)
    return [small_config, base_config, large_config][index]


class ELECTRAModel(nn.Module):

  def __init__(self, generator, discriminator, electra_tokenizer):
    super().__init__()
    self.generator, self.discriminator = generator, discriminator
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
    self.electra_tokenizer = electra_tokenizer

  def to(self, *args, **kwargs):
    "Also set dtype and device of contained gumbel distribution if needed"
    super().to(*args, **kwargs)
    a_tensor = next(self.parameters())
    device, dtype = a_tensor.device, a_tensor.dtype
    dtype = torch.float32
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

  def forward(self, masked_inputs, is_mlm_applied, labels):
    """
    masked_inputs (Tensor[int]): (B, L)
    sentA_lenths (Tensor[int]): (B, L)
    is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability
    labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
    """
    # this assumes that padding has taken place ALREADY
    # the attention mask is the tensortext of true and false
    # the token_type_ids is the tensor of zeros and ones

    attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs)

    gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[0] # (B, L, vocab size)
    # reduce size to save space and speed
    mlm_gen_logits = gen_logits[is_mlm_applied, :] # ( #mlm_positions, vocab_size)

    with torch.no_grad():
      pred_toks = self.sample(mlm_gen_logits)  # ( #mlm_positions, )

      # produce inputs for discriminator
      generated = masked_inputs.clone()  # (B,L)
      generated[is_mlm_applied] = pred_toks  # (B,L)

      # produce labels for discriminator
      is_replaced = is_mlm_applied.clone() # (B,L)
      is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied]) # (B,L)

    disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0] # (B, L)

    return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

  def _get_pad_mask_and_token_type(self, input_ids):
    """
    Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large,
    thus, preventing cpu processes loading batches from consuming lots of cpu memory and slow down the machine.
    """

    # get a TensorText object (like a list) of booleans indicating whether each
    # input token is a pad token
    attention_mask = input_ids != self.electra_tokenizer.pad_token_id

    #
    # # get the padded length of the sequences
    # # e.g. (128, 60) if input ids is TensorText, or torch.Size([128, 48]) (if input_ids is a tensor)
    # seq_len = input_ids.shape[1]

    # create a list of lists containing zeros and ones.
    # for every length, x, in sentA_lengths, create a list beginning with x zeros,
    # then add seq_len - x ones.

    # token_type_ids = torch.tensor([ ([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],
    #                               device=input_ids.device)

    # token_type_ids = torch.tensor([[0 if boolean else 1 for boolean in sub_mask] for sub_mask in iter(attention_mask)],
    #                                 device=input_ids.device)

    token_type_ids = torch.tensor([[int(not boolean) for boolean in sub_mask] for sub_mask in iter(attention_mask)],
                                  device=input_ids.device)


    # print("are my tensors equal?", torch.all(torch.eq(token_type_ids, token_type_ids_2)))
    # print(token_type_ids)
    # print(token_type_ids_2)

    return attention_mask, token_type_ids

  def sample(self, logits):
    "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
    return (logits.float() + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)