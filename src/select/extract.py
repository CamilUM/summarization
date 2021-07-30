def extract(sentences, indices):
    """
    Extract only the indicated sentences.
    Invalid indices are ignored.

    - sentences: [[str]].
    - indices: [int].
    - return: [[str]] with extracted sentences.
    """
    limit = len(sentences)
    return [sentences[i] for i in indices if 0 <= i < limit]
