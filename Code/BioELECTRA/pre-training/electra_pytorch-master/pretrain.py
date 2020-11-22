import os, sys, random
from pathlib import Path
from functools import partial
from datetime import datetime, timezone, timedelta
from IPython.core.debugger import set_trace as bk
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.tensor as T
import datasets
from fastai.text.all import *
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
from hugdatafast import *
from _utils.utils import *
from _utils.would_like_to_pr import *


""" Vanilla ELECTRA settings
"""


class ELECTRADataProcessor(object):
    """Given a stream of input text, creates pretraining examples."""

    def __init__(self, hf_dset, tokenizer, max_length, text_col='text', lines_delimiter='\n',
                 minimize_data_size=True, apply_cleaning=True):
        self.tokenizer = tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length

        self.hf_dset = hf_dset
        self.text_col = text_col
        self.lines_delimiter = lines_delimiter
        self.minimize_data_size = minimize_data_size
        self.apply_cleaning = apply_cleaning

    def map(self, **kwargs):
        "Some settings of datasets.Dataset.map for ELECTRA data processing"
        num_proc = kwargs.pop('num_proc', os.cpu_count())
        return self.hf_dset.my_map(
            function=self,
            batched=True,
            remove_columns=self.hf_dset.column_names,  # this is must b/c we will return different number of rows
            disable_nullable=True,
            input_columns=[self.text_col],
            writer_batch_size=10 ** 4,
            num_proc=num_proc,
            **kwargs
        )

    def __call__(self, texts):
        if self.minimize_data_size:
            new_example = {'input_ids': [], 'sentA_length': []}
        else:
            new_example = {'input_ids': [], 'input_mask': [], 'segment_ids': []}

        for text in texts:  # for every doc

            for line in re.split(self.lines_delimiter, text):  # for every paragraph

                if re.fullmatch(r'\s*', line): continue  # empty string or string with all space characters
                if self.apply_cleaning and self.filter_out(line): continue

                example = self.add_line(line)
                if example:
                    for k, v in example.items(): new_example[k].append(v)

            if self._current_length != 0:
                example = self._create_example()
                for k, v in example.items(): new_example[k].append(v)

        return new_example

    def filter_out(self, line):
        if len(line) < 80: return True
        return False

    def clean(self, line):
        # () is remainder after link in it filtered out
        return line.strip().replace("\n", " ").replace("()", "")

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = self.clean(line)
        tokens = self.tokenizer.tokenize(line)
        tokids = self.tokenizer.convert_tokens_to_ids(tokens)
        self._current_sentences.append(tokids)
        self._current_length += len(tokids)
        if self._current_length >= self._target_length:
            return self._create_example()
        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self._target_length - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in self._current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (len(first_segment) == 0 or
                    len(first_segment) + len(sentence) < first_segment_target_length or
                    (len(second_segment) == 0 and
                     len(first_segment) < first_segment_target_length and
                     random.random() < 0.5)):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(0, self._max_length -
                                             len(first_segment) - 3)]

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0
        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return self._make_example(first_segment, second_segment)

    def _make_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a tf.train.Example."""
        input_ids = [self.tokenizer.cls_token_id] + first_segment + [self.tokenizer.sep_token_id]
        sentA_length = len(input_ids)
        segment_ids = [0] * sentA_length
        if second_segment:
            input_ids += second_segment + [self.tokenizer.sep_token_id]
            segment_ids += [1] * (len(second_segment) + 1)

        if self.minimize_data_size:
            return {
                'input_ids': input_ids,
                'sentA_length': sentA_length,
            }
        else:
            input_mask = [1] * len(input_ids)
            input_ids += [0] * (self._max_length - len(input_ids))
            input_mask += [0] * (self._max_length - len(input_mask))
            segment_ids += [0] * (self._max_length - len(segment_ids))
            return {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            }


# define config here
config = {
    'device': "cuda:0" if torch.cuda.is_available() else "cpu:0",
    'seed': 0,
    'adam_bias_correction': False,
    'schedule': 'original_linear',
    'sampling': 'fp32_gumbel',  # choose from 'fp32_gumbel', 'fp16_gumbel', 'multinomial'
    'electra_mask_style': True,
    'size': 'small',
    'num_workers': 3,
}

# Check and Default
name_of_run = 'Electra_Seed_{}'.format(config["seed"])

# Setting of different sizes
i = ['small', 'base', 'large'].index(config['size'])

config["mask_prob"] = [0.15, 0.15, 0.25][i]
config["lr"] = [5e-4, 2e-4, 2e-4][i]
config["bs"] = [128, 256, 2048][i]
config["steps"] = [10**6, 766*1000, 400*1000][i]
config["max_length"] = [128, 512, 512][i]
generator_size_divisor = [4, 3, 4][i]

disc_config = ElectraConfig.from_pretrained(f'google/electra-{config["size"]}-discriminator')
gen_config = ElectraConfig.from_pretrained(f'google/electra-{config["size"]}-generator')
# note that public electra-small model is actually small++ and don't scale down generator size 
gen_config.hidden_size = int(disc_config.hidden_size/generator_size_divisor)
gen_config.num_attention_heads = disc_config.num_attention_heads//generator_size_divisor
gen_config.intermediate_size = disc_config.intermediate_size//generator_size_divisor
electra_tokenizer = ElectraTokenizerFast.from_pretrained(f'google/electra-{config["size"]}-generator')

# logger was removed entirely

# Path to data
Path('./datasets', exist_ok=True)
Path('./checkpoints/pretrain').mkdir(exist_ok=True, parents=True)
edl_cache_dir = Path("./datasets/electra_dataloader")
edl_cache_dir.mkdir(exist_ok=True)

# Print info
print(f"process id: {os.getpid()}")

# %%

# creating this partial function is the first place that electra_tokenizer is used.
ELECTRAProcessor = partial(ELECTRADataProcessor, tokenizer=electra_tokenizer, max_length=config["max_length"])
# todo check the type of the object that is returned by line 227


print('Load in dataset')
# dataset = datasets.load_dataset('csv', cache_dir='./datasets', data_files={'train': ['my_train_file_1.csv', 'my_train_file_2.csv']})['train']
dataset = datasets.load_dataset('csv', cache_dir='./datasets', data_files='./datasets/fibro_abstracts.csv')['train']



print('Load/create data from dataset for ELECTRA')
# apply_cleaning is true by default e.g. ELECTRAProcessor(dataset, apply_cleaning=False) if no cleaning
e_dataset = ELECTRAProcessor(dataset).map(cache_file_name=f'electra_customdataset_{config["max_length"]}.arrow', num_proc=1)


merged_dsets = {'train': e_dataset}
hf_dsets = HF_Datasets(merged_dsets, cols={'input_ids':TensorText,'sentA_length': noop},
                       hf_toker=electra_tokenizer, n_inp=2)
dls = hf_dsets.dataloaders(bs=config["bs"], num_workers=config["num_workers"], pin_memory=False,
                           shuffle_train=True,
                           srtkey_fc=False,
                           cache_dir='./datasets/electra_dataloader', cache_name='dl_{split}.json')

# # 2. Masked language model objective
# 2.1 MLM objective callback

# %%
"""
Modified from HuggingFace/transformers (https://github.com/huggingface/transformers/blob/0a3d0e02c5af20bfe9091038c4fd11fb79175546/src/transformers/data/data_collator.py#L102). 
It is a little bit faster cuz 
- intead of a[b] a on gpu b on cpu, tensors here are all in the same device
- don't iterate the tensor when create special tokens mask
And
- doesn't require huggingface tokenizer
- cost you only 550 µs for a (128,128) tensor on gpu, so dynamic masking is cheap   
"""
def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1, orginal_prob=0.1, ignore_index=-100):
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
    special_tokens_mask = special_tokens_mask | (inputs==sp_id)
  probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  mlm_mask = torch.bernoulli(probability_matrix).bool()
  labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

  # mask  (mlm_probability * (1-replace_prob-orginal_prob))
  mask_prob = 1 - replace_prob - orginal_prob
  mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
  inputs[mask_token_mask] = mask_token_index

  # replace with a random token (mlm_probability * replace_prob)
  if int(replace_prob) != 0:
    rep_prob = replace_prob/(replace_prob + orginal_prob)
    replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    inputs[replace_token_mask] = random_words[replace_token_mask]

  # do nothing (mlm_probability * orginal_prob)
  pass

  return inputs, labels, mlm_mask


class MaskedLMCallback(Callback):
  "MaskedLM Callback class handling tweaks of the training loop by changing a `Learner` in various events"

  @delegates(mask_tokens)
  def __init__(self, mask_tok_id, special_tok_ids, vocab_size, ignore_index=-100, **kwargs):
    self.ignore_index = ignore_index

    # assumes for_electra is true
    self.mask_tokens = partial(mask_tokens,
                               mask_token_index=mask_tok_id,
                               special_token_indices=special_tok_ids,
                               vocab_size=vocab_size,
                               ignore_index=-100,
                               **kwargs)

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
    assert torch.equal((~is_mlm_applied *masked_inputs + is_mlm_applied * labels), input_ids)
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


# %%
mlm_cb = MaskedLMCallback(mask_tok_id=electra_tokenizer.mask_token_id,
                          special_tok_ids=electra_tokenizer.all_special_ids,
                          vocab_size=electra_tokenizer.vocab_size,
                          mlm_probability=config["mask_prob"],
                          replace_prob=0.0 if config["electra_mask_style"] else 0.1,
                          orginal_prob=0.15 if config["electra_mask_style"] else 0.1)
#mlm_cb.show_batch(dls[0], idx_show_ignored=electra_tokenizer.convert_tokens_to_ids(['#'])[0])

# 3. ELECTRA (replaced token detection objective)
# see details in paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

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
    if config["sampling"] == 'fp32_gumbel': dtype = torch.float32
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

  def forward(self, masked_inputs, sentA_lenths, is_mlm_applied, labels):
    """
    masked_inputs (Tensor[int]): (B, L)
    sentA_lenths (Tensor[int]): (B, L)
    is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
    labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
    """
    attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs, sentA_lenths)

    gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[0] # (B, L, vocab size)
    # reduce size to save space and speed
    mlm_gen_logits = gen_logits[is_mlm_applied, :] # ( #mlm_positions, vocab_size)

    with torch.no_grad():
      # sampling
      pred_toks = self.sample(mlm_gen_logits)  # ( #mlm_positions, )

      # produce inputs for discriminator
      generated = masked_inputs.clone()  # (B,L)
      generated[is_mlm_applied] = pred_toks  # (B,L)

      # produce labels for discriminator
      is_replaced = is_mlm_applied.clone() # (B,L)
      is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied]) # (B,L)

    disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0] # (B, L)

    return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

  def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
    """
    Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
    """
    attention_mask = input_ids != self.electra_tokenizer.pad_token_id
    seq_len = input_ids.shape[1]
    token_type_ids = torch.tensor([ ([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],
                                  device=input_ids.device)
    return attention_mask, token_type_ids

  def sample(self, logits):
    "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
    if config["sampling"] == 'fp32_gumbel':
      return (logits.float() + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)
    elif config["sampling"] == 'fp16_gumbel': # 5.06 ms
      return (logits + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)
    elif config["sampling"] == 'multinomial': # 2.X ms
      return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()


class ELECTRALoss():
  """
  Generator loss function: Cross-Entropy loss flat
  Discriminator loss function: BCE with Logits
  """

  def __init__(self, loss_weights=(1.0, 50.0)):
    self.loss_weights = loss_weights
    self.generator_loss_function = CrossEntropyLossFlat()
    self.discriminator_loss_function = nn.BCEWithLogitsLoss()

  def __call__(self, pred, targ_ids):
    mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
    disc_logits = disc_logits.masked_select(non_pad)  # -> 1d tensor
    is_replaced = is_replaced.masked_select(non_pad)  # -> 1d tensor

    gen_loss = self.generator_loss_function(mlm_gen_logits.float(), targ_ids[is_mlm_applied])
    disc_loss = self.discriminator_loss_function(disc_logits.float(), is_replaced.float())

    return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]


# # 5. Train
# Seed & PyTorch benchmark
# TODO SET TO TRUE IF USING CUDA
torch.backends.cudnn.benchmark = False

def set_seed(seed_value):
    dls[0].rng = random.Random(seed_value)  # for fastai dataloader
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

set_seed(config["seed"])

# Generator and Discriminator
generator = ElectraForMaskedLM(gen_config)
discriminator = ElectraForPreTraining(disc_config)
discriminator.electra.embeddings = generator.electra.embeddings
generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

# ELECTRA training loop
electra_model = ELECTRAModel(generator, discriminator, electra_tokenizer)


# Optimizer
if config["adam_bias_correction"]:
    opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)
else:
    opt_func = partial(Adam_no_bias_correction, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)


# Learner
dls.to(torch.device(config["device"]))

learn = Learner(dls, electra_model,
                loss_func=ELECTRALoss(),
                opt_func=opt_func,
                path='./checkpoints',
                model_dir='pretrain',
                cbs=[mlm_cb, RunSteps(config["steps"], [0.0625, 0.125, 0.25, 0.5, 1.0], name_of_run+"_{percent}")],
                )


# Mixed precison and Gradient clip
learn.to_native_fp16(init_scale=2.**11)
learn.add_cb(GradientClipping(1.))

# Print time and run name
print(f"{name_of_run} , starts at {datetime.now()}")

# Learning rate schedule
lr_schedule = ParamScheduler({'lr': partial(linear_warmup_and_decay,
                                            lr_max=config["lr"],
                                            warmup_steps=10000,
                                            total_steps=config["steps"],)})

# Run
learn.fit(9999, cbs=[lr_schedule])


