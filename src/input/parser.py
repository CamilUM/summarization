import xml.etree.ElementTree as ET
from src.input.raw import Raw

def read(original, sums):
    """
    Get data from XML files.

    - original: str with original document.
    - sums: [str] with human-generated summaries.
    - return: Raw.
    """
    head, body, category = parse_original(original)
    sums = [parse_summary(s) for s in sums]

    return Raw(head, body, sums, category)

# Don't use outside of this module
def parse_original(filename):
    root = ET.parse(filename).getroot()
    head = root.find("HEAD").find("s").text
    text = root.find("TEXT").findall("s")
    body = [s.text for s in text]
    category = root.find("CATEGORY").find("s").text

    return head, body, category

def parse_summary(filename):
    return ET.parse(filename).getroot().text
