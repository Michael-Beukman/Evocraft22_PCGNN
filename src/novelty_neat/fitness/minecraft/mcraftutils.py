import numpy as np


def get_counts_of_array(arr, ignore_zero=True):
    uniques = np.unique(arr)
    alls = (set(uniques))
    if ignore_zero: alls -= {0}
    list_of_counts = []
    for j in alls:
        list_of_counts.append((arr == j).sum())
    return list_of_counts