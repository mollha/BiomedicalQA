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

        # translate characters to lower case
        return text.lower()

    return transform(prediction) == transform(expected)
