import io
import csv
from src.nlp.token import text

def ascsv(originals, summaries, indices):
    """
    - originals: [[[str]]].
    - summaries: {str: [[[str]]]}
    - indices: [int] with which text to include.
    - return: str with CSV format.
    """
    lines = []
    for method, sums in summaries.items():
        for o, s in zip([originals[i] for i in indices], [sums[i] for i in indices]):
            lines.append([method, text(o), text(s)])
    return row(lines)

def row(lines):
    """
    - lines: [[str]].
    - return: str.
    """
    # Iniciar
    out = io.StringIO()
    writer = csv.writer(out, delimiter = ",")

    # CSV
    writer.writerows(lines)

    # String
    return out.getvalue()
