from transformers import ElectraModel, ElectraTokenizer
import torch

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraModel.from_pretrained('google/electra-small-discriminator')

# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids)
#
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
#
# print(last_hidden_states)

question_text = "This is my question"
answer_text = "This is my answer"

# input_ids = tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)
input_ids = tokenizer.encode(question_text, answer_text, add_special_tokens=True)

print('The input has a total of {:} tokens.'.format(len(input_ids)))

tokens = tokenizer.convert_ids_to_tokens(input_ids)

for token, id in zip(tokens, input_ids):

    if id == tokenizer.sep_token_id:
        print('')

    print('{:<12} {:>6,}'.format(token, id))

    if id == tokenizer.sep_token_id:
        print('')
