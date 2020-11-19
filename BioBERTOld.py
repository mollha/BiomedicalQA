from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel, pipeline

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")

qa_model = AutoModelForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)





# Allocate a pipeline for question-answering
question_answerer = pipeline('question-answering')
question_answerer({
    'question': 'What is the name of the repository ?',
    'context': 'Pipeline have been included in the huggingface/transformers repository'
 })
print(outputs)