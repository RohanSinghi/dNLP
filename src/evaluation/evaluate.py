'''Module for evaluation metrics and methods.'''

def calculate_accuracy(predictions, labels):
    """Calculates the accuracy of the predictions.

    :param predictions: The predicted values.
    :type predictions: list
    :param labels: The ground truth labels.
    :type labels: list
    :return: The accuracy.
    :rtype: float
    """
    correct = sum([1 for p, l in zip(predictions, labels) if p == l])
    return correct / len(labels)