# BioELECTRA: An Efficient Approach to Biomedical Question Answering

Biomedical Question Answering system, developed as part of my Masters' Thesis

Project Supervised by Dr Noura Al-Moubayed

The repository is organised as follows:
- **Code**:<br />
    &nbsp;&nbsp;&nbsp;|-- **checkpoints**: saved model checkpoints<br />
    &nbsp;&nbsp;&nbsp;|-- **datasets**: pretraining and finetuning datasets<br />
    &nbsp;&nbsp;&nbsp;|-- **src**: source code<br />
- **ShellScripts**: slurm scripts for running experiments on NCC
- **Papers**:<br />
    &nbsp;&nbsp;&nbsp;|-- **LiteratureSurvey.pdf**<br />
    &nbsp;&nbsp;&nbsp;|-- **ProjectLog.pdf**<br />
    &nbsp;&nbsp;&nbsp;|-- **ProjectPaper.pdf**<br />
  
Note:
- ProjectLog.pdf is a log of project progress including notes from supervisor meetings.
- requirements.txt contains the python packages required to run this project.

Datasets are not provided due to copyright, however, they can be acquired from the following locations:
- Pubmed Pretraining Corpus: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
- BoolQ: https://github.com/google-research-datasets/boolean-questions
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
- BioASQ: http://participants-area.bioasq.org/datasets/