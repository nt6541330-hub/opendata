# open_source_data/data_collect/utils/html.py
import re

_BR_RE = re.compile(r"<br\s*/?>", re.I)
_TAG_RE = re.compile(r"<.*?>", re.S)

def strip_html(text: str) -> str:
    if not text:
        return ""
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    return text.strip()