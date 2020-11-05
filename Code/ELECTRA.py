from transformers import ElectraModel, ElectraTokenizer
import torch

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraModel.from_pretrained('google/electra-small-discriminator')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

print(last_hidden_states)