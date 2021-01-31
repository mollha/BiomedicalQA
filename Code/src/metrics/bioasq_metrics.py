""" ----------- BIOASQ EVALUATION METRICS -----------
BioASQ metrics differ for different question types:
    - yes/no questions (produce a single yes or no answer per question)
        - accuracy -> correct answers / total questions
        - precision and recall
        - f1_score -> measured independently for "yes" (f1_y) and "no" (f1_n) answers
        - maF1 -> macro averaged f-measure, ((f1_y) + (f1_n)) / 2

    - factoid questions (produce a list of candidate answers per question)
        - strict accuracy -> question correct if first element in list matches expected answer
        - lenient accuracy -> question correct if any element in list matches expected answer
        - MRR (official eval metric)

    - list questions (produce a list of answers per question - up to 100)
        - for every answer list, compute precision, recall and f measure
        - compute mean avg precision, recall and f-measure (official eval metric)
"""