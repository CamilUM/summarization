import json
from src.input.raw import Raw

def read(filename, maximum):
    """
    Get data from JSON files.

    - filename: str with JSON file with a list of dictionaries.
    - maximum: int with maximum length for output list.
    - return: [Raw].
    """
    raws = []
    with open(filename, "r") as f:
        for _, line in zip(range(maximum), f):
            raw = parse_entry(line)
            raws.append(raw)
    return raws

def parse_entry(line):
    """
    - f: str with JSON line with a dictionary with "title", "text", "summary" and "topic" entries.
    - return: Raw.
    """
    d = json.loads(line)
    head     = d["title"]
    body     = d["text"]
    sums     = [ d["summary"] ]
    category = d["topic"]
    return Raw(head, body, sums, category)
