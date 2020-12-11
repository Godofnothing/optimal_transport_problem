import operator
from functools import reduce

def concatenate_lists(lists):
    return reduce(operator.add, lists)

def get_inverted_labels(labels):
    inverted_labels = {label : [] for label in labels}
    for idx, label in enumerate(labels):
        inverted_labels[label].append(idx)
        
    return inverted_labels