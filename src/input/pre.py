from src.nlp.lemma import lemmatize
from src.nlp.stopwords import STOPWORDS, SPECIAL

def fix(text):
    """
    Normalize text and delete not important data. Norms:

    - Lowercase (A -> a)
    - Reduce spaces to one ('    ' -> ' ')
    - Trim spaces ('  A  ' -> 'A')
    - End with stop (A -> A.) (A. -> A.)

    - text: str.
    - return: str.
    """
    cleaned = text.lower()
    cleaned = " ".join(cleaned.split())
    cleaned = fullstop(cleaned)
    return cleaned

def fullstop(text):
    return text if text.endswith(".") else text + "."

def clean(text):
    """
    Normalize text and delete not important data. Norms:

    - Lemmatize (am, are, is -> be)
    - Delete stopwords (and, the, ...)
    - Delete special characters (#, &, @, ...)

    - text: [str].
    - return: str.
    """
    tokens = text
    #tokens = lemmatize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [t for t in tokens if t not in SPECIAL]
    return tokens
