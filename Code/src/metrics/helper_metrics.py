import string


# --------- METRIC HELPER FUNCTIONS ---------
def check_match(prediction, expected):
    """
    Defines the logic for checking if a prediction matches the expected answer.
    We can decide what we constitute as a match. We aren't being leniant.
    :return:
    """

    def transform(text: str):
        """
        Process text to ensure comparable and consistent format.
        For now, we only convert to lower case.
        :param text: text to transform
        :return: transformed text
        """

        if text is None:
            return None

        # remove punctuation from the string
        text = text.translate(str.maketrans('', '', string.punctuation))

        # translate characters to lower case
        return text.lower()

    def check_equality(a, b):
        if a is None and b is None:
            return True
        return a == b

    if type(expected) == list:  # we have a list of candidate answers
        # check if it matches any of the correct answers.
        return any([check_equality(transform(prediction), transform(exp)) for exp in expected])
    return check_equality(transform(prediction), transform(expected))
