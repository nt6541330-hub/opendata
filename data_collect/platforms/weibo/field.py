from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class WeiboImage:
    url: str

@dataclass
class WeiboPost:
    note_id: str
    keyword: str
    text_html: str
    images: List[WeiboImage] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    source: str = "m_api"

@dataclass
class WeiboComment:
    cid: str
    text_html: str
    raw: Dict[str, Any] = field(default_factory=dict)