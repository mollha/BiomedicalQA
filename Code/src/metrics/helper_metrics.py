import string


# --------- METRIC HELPER FUNCTIONS ---------
def check_match(prediction, expected):
    """
    Defines the logic for checking if a prediction matches the expected answer.
    We can decide what we constitute as a match.
    :return:
    """

    def transform(text: str) -> str:
        """
        Process text to ensure comparable and consistent format.
        For now, we only convert to lower case.
        :param text: text to transform
        :return: transformed text
        """

        # remove punctuation from the string
        text = text.translate(str.maketrans('', '', string.punctuation))

        # translate characters to lower case
        return text.lower()

    if type(expected) == list:  # we have a list of candidate answers
        # check if it matches any of the correct answers.
        return any([transform(prediction) == transform(exp) for exp in expected])
    return transform(prediction) == transform(expected)
