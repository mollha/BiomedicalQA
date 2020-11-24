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

from mask_inputs import MaskedLMCallback
from data_processing import ELECTRADataProcessor
from models import ELECTRAModel

from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
from hugdatafast import *
from _utils.utils import *
from _utils.would_like_to_pr import *


""" Vanilla ELECTRA settings
"""

# define config here
config = {
    'device': "cuda:0" if torch.cuda.is_available() else "cpu:0",
    'seed': 0,
    'adam_bias_correction': False,
    'schedule': 'original_linear',
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
mlm_cb = MaskedLMCallback(mask_tok_id=electra_tokenizer.mask_token_id,
                          special_tok_ids=electra_tokenizer.all_special_ids,
                          vocab_size=electra_tokenizer.vocab_size,
                          mlm_probability=config["mask_prob"],
                          replace_prob=0.0 if config["electra_mask_style"] else 0.1,
                          orginal_prob=0.15 if config["electra_mask_style"] else 0.1)

# mlm_cb.show_batch(dls[0], idx_show_ignored=electra_tokenizer.convert_tokens_to_ids(['#'])[0])


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
torch.backends.cudnn.benchmark = torch.cuda.is_available()


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


