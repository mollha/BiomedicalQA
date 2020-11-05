from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")

model = AutoModelForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")