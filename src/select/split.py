import random

def sample(objects, percentage = 0.8, seed = 0):
    """
    - objects: [object].
    - percentage: float.
    - seed: int.
    - return: [int] with indices.
    """
    if objects == []: return []
    random.seed(seed)
    n = int(percentage * len(objects))
    return sorted(random.sample(range(len(objects)), n))

def split(objects, samples):
    """
    - objects: [object].
    - samples: [int] with indices.
    - return: [object] with first part.
    - return: [object] with second part.
    """
    if objects == []: return [], []
    if samples == []: return [], objects
    if len(objects) == len(samples): return objects, []

    # Limit
    n = len(objects)
    # First part
    train = [objects[s] for s in samples if 0 <= s < n]
    # Second part
    samples = sorted(set(range(n)) - set(samples))
    test    = [objects[s] for s in samples]
    return train, test
