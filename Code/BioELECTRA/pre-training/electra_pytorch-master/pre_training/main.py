from callback_functions import MaskedLMCallback, GradientClipping, RunSteps
from data_processing import ELECTRADataProcessor
from loss_functions import ELECTRALoss
from models import ELECTRAModel, get_model_config
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
from hugdatafast import *
from _utils.utils import *


if __name__ == "__main__":

    # define config here
    config = {
        'device': "cuda:0" if torch.cuda.is_available() else "cpu:0",
        'seed': 0,
        'adam_bias_correction': False,
        'schedule': 'original_linear',
        'electra_mask_style': True,
        'size': 'small',
        'num_workers': 3 if torch.cuda.is_available() else 0,           # this might be wrong - it initially was just 3
    }

    # Check and Default
    name_of_run = 'Electra_Seed_{}'.format(config["seed"])

    # merge general config with model specific config
    # Setting of different sizes
    model_specific_config = get_model_config(config['size'])
    config = {**config, **model_specific_config}

    discriminator_config = ElectraConfig.from_pretrained(f'google/electra-{config["size"]}-discriminator')
    generator_config = ElectraConfig.from_pretrained(f'google/electra-{config["size"]}-generator')

    # note that public electra-small model is actually small++ and don't scale down generator size
    generator_config.hidden_size = int(discriminator_config.hidden_size/config["generator_size_divisor"])
    generator_config.num_attention_heads = discriminator_config.num_attention_heads//config["generator_size_divisor"]
    generator_config.intermediate_size = discriminator_config.intermediate_size//config["generator_size_divisor"]
    electra_tokenizer = ElectraTokenizerFast.from_pretrained(f'google/electra-{config["size"]}-generator')

    # Path to data
    Path('../datasets', exist_ok=True)
    Path('./checkpoints/pretrain').mkdir(exist_ok=True, parents=True)
    edl_cache_dir = Path("../datasets/electra_dataloader")
    edl_cache_dir.mkdir(exist_ok=True)

    # Print info
    print(f"process id: {os.getpid()}")

    # creating this partial function is the first place that electra_tokenizer is used.
    ELECTRAProcessor = partial(ELECTRADataProcessor, tokenizer=electra_tokenizer, max_length=config["max_length"])

    print('Load in the dataset.')
    dataset = datasets.load_dataset('csv', cache_dir='../datasets', data_files='./datasets/fibro_abstracts.csv')['train']

    print('Create or load cached ELECTRA-compatible data.')
    # apply_cleaning is true by default e.g. ELECTRAProcessor(dataset, apply_cleaning=False) if no cleaning
    e_dataset = ELECTRAProcessor(dataset).map(cache_file_name=f'electra_customdataset_{config["max_length"]}.arrow', num_proc=1)

    hf_dsets = HF_Datasets({'train': e_dataset}, cols={'input_ids': TensorText, 'sentA_length': noop},
                           hf_toker=electra_tokenizer, n_inp=2)

    # data loader
    dls = hf_dsets.dataloaders(bs=config["bs"], num_workers=config["num_workers"], pin_memory=False,
                               shuffle_train=True,
                               srtkey_fc=False,
                               cache_dir='../datasets/electra_dataloader', cache_name='dl_{split}.json')


    # # 2. Masked language model objective
    # 2.1 MLM objective callback
    mlm_cb = MaskedLMCallback(mask_tok_id=electra_tokenizer.mask_token_id,
                              special_tok_ids=electra_tokenizer.all_special_ids,
                              vocab_size=electra_tokenizer.vocab_size,
                              mlm_probability=config["mask_prob"],
                              replace_prob=0.0 if config["electra_mask_style"] else 0.1,
                              orginal_prob=0.15 if config["electra_mask_style"] else 0.1)

    # mlm_cb.show_batch(dls[0], idx_show_ignored=electra_tokenizer.convert_tokens_to_ids(['#'])[0])

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
    generator = ElectraForMaskedLM(generator_config)
    discriminator = ElectraForPreTraining(discriminator_config)
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

    # Learner is the basic fast ai class for handling the training loop
    # dls: data loaders
    # model: the model to train
    # loss_func: the loss function to use
    # opt_func: used to create an optimiser when Learner.fit is called
    # lr: is the default learning rate
    # :
    learn = Learner(dls, electra_model,
                    loss_func=ELECTRALoss(),
                    opt_func=opt_func,
                    path='./checkpoints',
                    model_dir='pretrain',
                    cbs=[mlm_cb, RunSteps(config["steps"], [0.0625, 0.125, 0.25, 0.5, 1.0], name_of_run+"_{percent}")],
                    )

    # Mixed precison and Gradient clip
    learn.to_native_fp16(init_scale=2.**11)

    # add callback
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