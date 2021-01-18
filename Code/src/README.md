# Pre-training ELECTRA to incorporate closed-domain vocabulary.

We are using the Electra Model from the `huggingface transformers` library, instantiating the model with pre-trained weights.
https://huggingface.co/transformers/model_doc/electra.html

Pre-training on Biomedical text allows us to incorporate relevant language for Biomedical Question Answering.




- visualise.py: draw graphs of statistics collected by the custom loss function during pre-training.