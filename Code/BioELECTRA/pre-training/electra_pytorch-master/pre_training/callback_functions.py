import time
from statistics import mean, stdev
import torch
from torch import nn
from fastai.text.all import *
from functools import partial
from fastai.callback.core import Callback, CancelFitException, CancelEpochException
from fastai.data.core import delegates, TfmdDL  # data loader
from fastcore.basics import store_attr



"""
Modified from HuggingFace/transformers (https://github.com/huggingface/transformers/blob/0a3d0e02c5af20bfe9091038c4fd11fb79175546/src/transformers/data/data_collator.py#L102). 
It is a little bit faster cuz 
- intead of a[b] a on gpu b on cpu, tensors here are all in the same device
- don't iterate the tensor when create special tokens mask
And
- doesn't require huggingface tokenizer
- cost you only 550 µs for a (128,128) tensor on gpu, so dynamic masking is cheap   
"""
def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1,
                orginal_prob=0.1, ignore_index=-100):
    """
    Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence.
    * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
    """

    device = inputs.device
    labels = inputs.clone()

    # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)

    for sp_id in special_token_indices:
        special_tokens_mask = special_tokens_mask | (inputs == sp_id)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    mlm_mask = torch.bernoulli(probability_matrix).bool()
    labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

    # mask (mlm_probability * (1-replace_prob-orginal_prob))
    mask_prob = 1 - replace_prob - orginal_prob
    mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
    inputs[mask_token_mask] = mask_token_index

    # replace with a random token (mlm_probability * replace_prob)
    if int(replace_prob) != 0:
        rep_prob = replace_prob / (replace_prob + orginal_prob)
        replace_token_mask = torch.bernoulli(
            torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
        inputs[replace_token_mask] = random_words[replace_token_mask]

    # do nothing (mlm_probability * orginal_prob)
    pass

    return inputs, labels, mlm_mask



class MaskedLM():
    def __init__(self, mask_tok_id, special_tok_ids, vocab_size, ignore_index=-100, **kwargs):
        self.ignore_index = ignore_index

        # assumes for_electra is true
        self.mask_tokens = partial(mask_tokens, mask_token_index=mask_tok_id, special_token_indices=special_tok_ids,
                                   vocab_size=vocab_size, ignore_index=-100, **kwargs)


    def mask_batch(self, inputs) -> tuple:
        """
        Compute the masked inputs - in ELECTRA, MLM is used, therefore the raw batches should
        not be passed to the model.
        :return: None

        ---- Attributes of Learner: ----
        xb: last input drawn from self.dl (current DataLoader used for iteration), potentially modified by callbacks
        yb: last target drawn from self.dl (potentially modified by callbacks).
        --------------------------------
        """

        input_ids, sent_lengths = inputs
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids)

        # return self.learn.xb, self.learn.yb
        return (masked_inputs, sent_lengths, is_mlm_applied, labels), (labels,)



class MaskedLMCallback(Callback):
    " MaskedLM Callback class handling tweaks of the training loop by changing a `Learner` in various events"

    @delegates(mask_tokens)
    def __init__(self, mask_tok_id, special_tok_ids, vocab_size, ignore_index=-100, **kwargs):
        self.ignore_index = ignore_index

        # assumes for_electra is true
        self.mask_tokens = partial(mask_tokens, mask_token_index=mask_tok_id, special_token_indices=special_tok_ids,
                                   vocab_size=vocab_size, ignore_index=-100, **kwargs)

    def before_batch(self) -> None:
        """
        Compute the masked inputs - in ELECTRA, MLM is used, therefore the raw batches should
        not be passed to the model.
        :return: None

        ---- Attributes of Learner: ----
        xb: last input drawn from self.dl (current DataLoader used for iteration), potentially modified by callbacks
        yb: last target drawn from self.dl (potentially modified by callbacks).
        --------------------------------
        """

        input_ids, sent_lengths = self.xb
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids)
        self.learn.xb, self.learn.yb = (masked_inputs, sent_lengths, is_mlm_applied, labels), (labels,)

    @delegates(TfmdDL.show_batch)
    def show_batch(self, dl, idx_show_ignored, verbose=True, **kwargs):
        input_ids, sent_lengths = dl.one_batch()
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids.clone())
        # check
        assert torch.equal(is_mlm_applied, labels != self.ignore_index)
        assert torch.equal((~is_mlm_applied * masked_inputs + is_mlm_applied * labels), input_ids)
        # change symbol to show the ignored position

        labels[labels == self.ignore_index] = idx_show_ignored
        # some notice to help understand the masking mechanism
        if verbose:
            print("We won't count loss from position where y is ignore index")
            print("Notice 1. Positions have label token in y will be either [Mask]/other token/orginal token in x")
            print("Notice 2. Special tokens (CLS, SEP) won't be masked.")
            print("Notice 3. Dynamic masking: every time you run gives you different results.")
        # show
        tfm_b = (masked_inputs, sent_lengths, is_mlm_applied, labels)
        dl.show_batch(b=tfm_b, **kwargs)


class GradientClipping(Callback):
    """
    Gradient clipping prevents exploding gradients in DNNs
    """
    def __init__(self, clip: float = 0.1):
        """

        :param clip:
        """
        self.clip = clip
        assert self.clip
    def after_backward(self):
        if hasattr(self, 'scaler'): self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)


class RunSteps(Callback):
    toward_end = True

    def __init__(self, n_steps, save_points=None, base_name=None, no_val=True):
        """
        Args:
          `n_steps` (`Int`): Run how many steps, could be larger or smaller than `len(dls.train)`
          `savepoints`
          - (`List[Float]`): save when reach one of percent specified.
          - (`List[Int]`): save when reache one of steps specified
          `base_name` (`String`): a format string with `{percent}` to be passed to `learn.save`.
        """
        if save_points is None:
            save_points = []
        else:
            assert '{percent}' in base_name
            save_points = [s if isinstance(s, int) else int(n_steps * s) for s in save_points]
            for sp in save_points: assert sp != 1, "Are you sure you want to save after 1 steps, instead of 1.0 * num_steps ?"
            assert max(save_points) <= n_steps
        store_attr('n_steps,save_points,base_name,no_val', self)

    def before_train(self):
        # fix pct_train (cuz we'll set `n_epoch` larger than we need)
        self.learn.pct_train = self.train_iter / self.n_steps

    def after_batch(self):
        # fix pct_train (cuz we'll set `n_epoch` larger than we need)
        self.learn.pct_train = self.train_iter / self.n_steps
        # when to save
        if self.train_iter in self.save_points:
            percent = (self.train_iter / self.n_steps) * 100
            self.learn.save(self.base_name.format(percent=f'{percent}%'))
        # when to interrupt
        if self.train_iter == self.n_steps:
            raise CancelFitException

    def after_train(self):
        if self.no_val:
            if self.train_iter == self.n_steps:
                pass  # CancelFit is raised, don't overlap it with CancelEpoch
            else:
                raise CancelEpochException
