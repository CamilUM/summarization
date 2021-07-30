from math import ceil

def threshold(scores, thresh):
    """
    Filter sentences given their associated scores and
    the threshold to filter them. Only get those above threshold.

    - scores: [float].
    - thresh: float.
    - return: [int] with selected indices.
    """
    return mask([thresh <= s for s in scores])

def mask(booleans):
    """
    Filter sentences given a binary mask. Only get true ones.

    - booleans: [bool] with binary mask.
    - return: [int] with selected indices.
    """
    return [i for i in range(len(booleans)) if booleans[i]]

def k_top(scores, k):
    """
    Filter sentences given their associated scores and
    the maximum desired sentences. Only get the k-most scored ones.

    - scores: [str].
    - k: int and k > 0.
    - return: [int] with selected indices.
    """
    if k <= 0: return []
    indices = range(len(scores))
    return sorted(sorted(indices, key = lambda x: scores[x], reverse = True)[:k])

def p_top(scores, p):
    """
    Filter sentences given their associated scores and
    the proportion of desired sentences. Only get the p-most scored ones.

    - scores: [float]
    - p: float and 0 < p <= 1.
    - return: [int] with selected indices.
    """
    if p <= 0: return []
    if p >= 1: return range(len(scores))
    return k_top(scores, ceil(p * len(scores)))
