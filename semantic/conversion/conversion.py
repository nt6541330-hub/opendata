# semantic/conversion/conversion.py
"""
conversion.py

作为通用工具模块被导入使用。
核心功能：
1. normalize_date_from_raw: 强大的中文时间字符串解析，支持模糊时间、区间、农历数字转换等。
2. convert_document_simple: 将原始的 interim 文档结构转换为 evolution 演化图谱所需的简化结构。
"""

import re
from datetime import datetime, timedelta
from calendar import monthrange

# ==========================
# 1. 常量与配置
# ==========================
DEFAULT_YEAR = datetime.now().year
MONTH_END_PREF_DAY = 1  # 月末/仅月时的默认日期（1表示月初）
FORCE_DEFAULT_YEAR_FOR_TIME_TAIL = True  # 当含有具体时间(时分秒)但缺年份时，是否强制使用默认年份

# 模糊时间词表
FUZZY = [
    '昨天', '昨日', '今天', '今日', '明天', '明日',
    '上月', '本月', '去年', '今年', '明年',
    '近期', '近段时间', '最近', '近日', '不久前', '日前', '即将到来'
]

# 辅助正则常量
DASH_CLASS = r"[—–\-]"  # em/en/连字符（包括 ASCII 短横）
SEP_MD = rf"(?:{DASH_CLASS}|至|到)"  # “—/–/-/至/到”
SP = r"\s*"  # 可选空格

TODAY = datetime.now().date()
CURR_YEAR = TODAY.year
CURR_MONTH = TODAY.month

# 中文数字映射
_CN_NUM_D = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15, "十六": 16, "十七": 17,
    "十八": 18, "十九": 19, "二十": 20, "二十一": 21, "二十二": 22, "二十三": 23,
    "二十四": 24, "二十五": 25, "二十六": 26, "二十七": 27, "二十八": 28, "二十九": 29,
    "三十": 30, "三十一": 31
}
DAY_CN_ALTS = "|".join(sorted(_CN_NUM_D.keys(), key=len, reverse=True))
DAY_CN_GROUP = "(?:" + DAY_CN_ALTS + ")"
PAT_DAY_CN_ONLY = re.compile("^" + SP + r"(?P<cn>" + DAY_CN_GROUP + r")(?:号|日)" + SP + "$")
PAT_DAY_NUM_ONLY = re.compile("^" + SP + r"(?P<d>\d{1,2})(?:号|日)" + SP + "$")

# ==========================
# 2. 正则表达式表 (PATS)
# ==========================
PATS = [
    # 完整范围： yyyy年m月d日 至 yyyy年m月d日 -> 返回结束端
    ("ymd_range_both_years", re.compile(
        rf"^{SP}(?P<y1>\d{{4}})年(?P<m1>\d{{1,2}})月(?P<d1>\d{{1,2}})日{SP}{SEP_MD}{SP}(?P<y2>\d{{4}})年(?P<m2>\d{{1,2}})月(?P<d2>\d{{1,2}})日{SP}$"
    )),
    # 同年范围： yyyy年m月d日 至 m月d日 -> 返回结束端（当年）
    ("ymd_range_same_year", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m1>\d{{1,2}})月(?P<d1>\d{{1,2}})日{SP}{SEP_MD}{SP}(?P<m2>\d{{1,2}})月(?P<d2>\d{{1,2}})日{SP}$"
    )),
    # 年+月范围（紧凑写法）： 2025年1-9月
    ("ym_month_compact_range", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m1>\d{{1,2}}){DASH_CLASS}(?P<m2>\d{{1,2}})月{SP}$"
    )),
    # 无年但含月日区间： 9月17日至9月19日
    ("md_range_no_year", re.compile(
        rf"^{SP}(?P<m1>\d{{1,2}})月{SP}(?P<d1>\d{{1,2}})日{SP}{SEP_MD}{SP}(?P<m2>\d{{1,2}})月{SP}(?P<d2>\d{{1,2}})日{SP}$"
    )),
    # 月-月无年
    ("m_range_no_year", re.compile(
        rf"^{SP}(?P<m1>\d{{1,2}})月{SP}{SEP_MD}{SP}(?P<m2>\d{{1,2}})月{SP}$"
    )),

    # 年份区间
    ("y_to_y_range", re.compile(rf"^{SP}(?P<y1>\d{{4}})年{SP}(?:至|到|{DASH_CLASS}){SP}(?P<y2>\d{{4}})年{SP}$")),
    ("year_decade", re.compile(rf"^{SP}(?P<y>\d{{4}})年代{SP}$")),
    ("year_after", re.compile(rf"^{SP}(?P<y>\d{{4}})年(?:后|之后){SP}$")),
    ("year_begin", re.compile(rf"^{SP}(?P<y>\d{{4}})年(?:初|年初){SP}$")),
    ("year_end", re.compile(rf"^{SP}(?P<y>\d{{4}})年(?:末|年末){SP}$")),

    ("quarter", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年{SP}(?:(?P<cn>[第]?[一二三四])季度|Q(?P<qn>[1-4])){SP}$",
        re.IGNORECASE
    )),
    ("half_year", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年{SP}(?P<h>上半年|下半年){SP}$"
    )),
    ("ymd_to_md_same_year", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m1>\d{{1,2}})月(?P<d1>\d{{1,2}})日{SP}(?:至|到|{DASH_CLASS}){SP}(?P<m2>\d{{1,2}})月(?P<d2>\d{{1,2}})日{SP}$"
    )),
    ("ymd_to_d", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月(?P<d1>\d{{1,2}})日{SP}(?:至|到|{DASH_CLASS}){SP}(?P<d2>\d{{1,2}})日{SP}$"
    )),
    # 跨年区间
    ("ym_to_ym_cross", re.compile(
        rf"^{SP}(?P<y1>\d{{4}})年(?P<m1>\d{{1,2}})月{SP}(?:至|到|{DASH_CLASS}){SP}(?P<y2>\d{{4}})年(?P<m2>\d{{1,2}})月{SP}$"
    )),
    # 同年区间
    ("ym_to_m_same", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m1>\d{{1,2}})月{SP}(?:至|到|{DASH_CLASS}){SP}(?P<m2>\d{{1,2}})月{SP}$"
    )),
    ("y_to_now", re.compile(rf"^{SP}(?P<y>\d{{4}})年{SP}至今{SP}$")),
    ("ymd_with_clockish", re.compile(
        rf"^{SP}(?P<y>\d{{1,4}})年(?P<m>\d{{1,2}})月(?P<d>\d{{1,2}})日(?:{SP}(?:(?:(?P<h>\d{{1,2}})(?:时|点)(?:许|左右)?)|(?:(?P<h2>\d{{1,2}}):(?P<min>\d{{1,2}})(?::(?P<sec>\d{{1,2}}))?))){SP}$"
    )),
    ("ymd_with_period", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月(?P<d>\d{{1,2}})日(?:上午|下午|晚间|晚上|晚|清晨|早上|中午|傍晚|凌晨|夜间|深夜)?{SP}$"
    )),
    ("ym_month_end", re.compile(rf"^{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月(?:底|末){SP}$")),
    ("iso_date_dash", re.compile(rf"^{SP}(?P<y>\d{{4}})[\-\.\/](?P<m>\d{{1,2}})[\-\.\/](?P<d>\d{{1,2}}){SP}$")),
    ("ymd", re.compile(rf"^{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月(?P<d>\d{{1,2}})日{SP}$")),
    ("ym", re.compile(rf"^{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月{SP}$")),
    ("y_only", re.compile(rf"^{SP}(?P<y>\d{{4}})年{SP}$")),
    ("md", re.compile(rf"^{SP}(?P<m>\d{{1,2}})月{SP}(?P<d>\d{{1,2}})日{SP}$")),
    ("m_only", re.compile(rf"^{SP}(?P<m>\d{{1,2}})月{SP}$")),
    ("multi_years", re.compile(rf"^{SP}(?P<y1>\d{{4}})年.*?(?P<y2>\d{{1,4}})年.*$")),
    ("year_range", re.compile(rf"^{SP}(?P<y1>\d{{4}})年{SP}{DASH_CLASS}{SP}(?P<y2>\d{{4}})年{SP}$")),
    ("year_range_loose", re.compile(
        rf"^{SP}(?P<y1>\d{{4}})(?:年)?{SP}{DASH_CLASS}{SP}(?P<y2>\d{{4}})(?:年)?{SP}$"
    )),
    ("century_decade", re.compile(
        rf"^{SP}(?P<c>\d{{1,2}})世纪(?P<d>[0-9])0年代{SP}$"
    )),
    ("this_year_month", re.compile(rf"^{SP}今年(?P<m>\d{{1,2}})月(?:份)?{SP}$")),
    ("this_year_half", re.compile(rf"^{SP}今年(?P<h>上半年|下半年){SP}$")),
    ("thisyear_half_alt", re.compile(rf"^{SP}(本年)(?P<h>上半年|下半年){SP}$")),
    ("ym_with_xun", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月(?P<x>上旬|中旬|下旬|中下旬|中上旬){SP}$"
    )),
    ("m_with_xun_no_year", re.compile(
        rf"^{SP}(?P<m>\d{{1,2}})月(?P<x>上旬|中旬|下旬){SP}$"
    )),
    ("md_to_d_no_year", re.compile(
        rf"^{SP}(?P<m>\d{{1,2}})月{SP}(?P<d1>\d{{1,2}})日{SP}{SEP_MD}{SP}(?P<d2>\d{{1,2}})日{SP}$"
    )),
    ("ymd_d_to_d", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月(?P<d1>\d{{1,2}}){SP}{DASH_CLASS}{SP}(?P<d2>\d{{1,2}})日{SP}$"
    )),
    ("y_zeromd", re.compile(rf"^{SP}(?P<y>\d{{4}})年0{{2}}月0{{2}}日{SP}$")),
    ("y_first_three_q", re.compile(rf"^{SP}(?P<y>\d{{4}})年前三个季度{SP}$")),
    ("ymd_with_dangri", re.compile(rf"{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月(?P<d>\d{{1,2}})日{SP}当日{SP}$")),
    ("year_before", re.compile(rf"{SP}(?P<y>\d{{4}})年之前{SP}$")),
    ("ym_with_comment", re.compile(rf"{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月{SP}（.*?）{SP}$")),
    ("ym_before", re.compile(rf"{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月之前{SP}$")),
    ("ym_begin", re.compile(rf"{SP}(?P<y>\d{{4}})年(?P<m>\d{{1,2}})月初{SP}$")),
]


# ==========================
# 3. 辅助函数
# ==========================

def _cn_day_to_int(s: str):
    return _CN_NUM_D.get(s)


def _pre_clean(s: str) -> str:
    """轻度清洗输入字符串（全角空格等）"""
    if not isinstance(s, str):
        return ""
    s = s.replace("\u3000", " ")
    return s.strip()


def _last_day_clamped(y: int, m: int, pref_day: int = MONTH_END_PREF_DAY) -> int:
    """返回 min(pref_day, 当月天数)"""
    return min(pref_day, monthrange(y, m)[1])


def _clamp_day(y: int, m: int, d: int) -> int:
    return min(d, monthrange(y, m)[1])


def _search_prev_ym_in_text(s: str, day_pos: int = None):
    """
    在字符串 s 中向前查找最近的 'YYYY年M月' 或 'M月'（返回 (year, month) 或 (None, month)）
    """
    if not s:
        return None, None
    hay = s if day_pos is None else s[:day_pos]
    m = re.search(r"(?P<y>\d{4})年\s*(?P<m>\d{1,2})月", hay)
    if m:
        return int(m.group("y")), int(m.group("m"))
    m2s = list(re.finditer(r"(?P<m>\d{1,2})月", hay))
    if m2s:
        m2 = m2s[-1]
        return None, int(m2.group("m"))
    return None, None


def _infer_year_month_from_event(event: dict, raw: str = None):
    """
    尝试从当前 event 的子事件/其它字段中推断年和月
    """
    if not isinstance(event, dict):
        return None, None

    candidates = []
    # 如果有子事件列表，优先扫描
    for key in ("subevents", "children", "events"):
        lst = event.get(key)
        if isinstance(lst, list):
            for item in lst:
                if isinstance(item, dict):
                    for tkey in (
                    "moment", "period", "time_position_moment", "time_position", "time", "date", "datetime"):
                        v = item.get(tkey)
                        if isinstance(v, str):
                            candidates.append(v)
                    if "raw" in item and isinstance(item["raw"], str):
                        candidates.append(item["raw"])
    # 扫描 event 的字符串字段
    for k, v in event.items():
        if isinstance(v, str) and k not in ("event_id",):
            candidates.append(v)

    for text in candidates:
        if not text: continue
        m = re.search(r"(?P<y>\d{4})年\s*(?P<m>\d{1,2})月", text)
        if m:
            return int(m.group("y")), int(m.group("m"))
        m2s = list(re.finditer(r"(?P<m>\d{1,2})月", text))
        if m2s:
            return None, int(m2s[-1].group("m"))
    return None, None


# ==========================
# 4. 核心解析函数
# ==========================

def normalize_date_from_raw(raw: str, event: dict = None) -> str:
    """
    解析各种日期字符串，返回规范化字符串 YYYY-MM-DD（对缺月日的返回为 YYYY-00-00）。
    对区间统一返回**结束端**。
    event 可选：当 raw 是“日/号”且缺年月时，会尝试从 event 的上下文中推断。
    """
    if not raw:
        return ""
    s = _pre_clean(raw)

    # 模糊词直接近似为最近一天（返回具体日期）
    if any(k in s for k in FUZZY):
        dt = datetime.now()
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"

    # 先尝试中文“几号/几日”
    m_only_cn = PAT_DAY_CN_ONLY.fullmatch(s)
    m_only_num = PAT_DAY_NUM_ONLY.fullmatch(s)

    if m_only_cn or m_only_num:
        if m_only_cn:
            cn = m_only_cn.group("cn")
            d = _cn_day_to_int(cn)
            day_pos = raw.find(cn)
        else:
            d = int(m_only_num.group("d"))
            mpos = re.search(r"(\d{1,2})(?:号|日)", raw)
            day_pos = mpos.start() if mpos else None

        # 1) 同串中向前查找 yyyy年m月 或 最近的 m月
        y_prev, m_prev = _search_prev_ym_in_text(raw, day_pos=day_pos)
        if y_prev and m_prev:
            d0 = _clamp_day(y_prev, m_prev, d)
            return f"{y_prev:04d}-{m_prev:02d}-{d0:02d}"
        if m_prev:
            d0 = _clamp_day(DEFAULT_YEAR, m_prev, d)
            return f"{DEFAULT_YEAR:04d}-{m_prev:02d}-{d0:02d}"

        # 2) 尝试从 event 上下文推断
        if event:
            y_doc, m_doc = _infer_year_month_from_event(event, raw=s)
            if y_doc and m_doc:
                d0 = _clamp_day(y_doc, m_doc, d)
                return f"{y_doc:04d}-{m_doc:02d}-{d0:02d}"
            if m_doc:
                d0 = _clamp_day(DEFAULT_YEAR, m_doc, d)
                return f"{DEFAULT_YEAR:04d}-{m_doc:02d}-{d0:02d}"

        # 3) 回落：默认年 + 当前月
        d0 = _clamp_day(DEFAULT_YEAR, CURR_MONTH, d)
        return f"{DEFAULT_YEAR:04d}-{CURR_MONTH:02d}-{d0:02d}"

    # 按 PATS 优先级匹配
    for name, pat in PATS:
        m = pat.fullmatch(s)
        if not m: continue
        g = m.groupdict()
        try:
            # 含完整年月日范围 -> 返回结束端
            if name in ("ymd_range_both_years", "ymd_range_same_year"):
                y = int(g.get("y2") or g.get("y"))
                return f"{y:04d}-{int(g['m2']):02d}-{int(g['d2']):02d}"

            if name == "ym_month_compact_range":
                return f"{int(g['y']):04d}-{int(g['m2']):02d}-00"

            if name == "md_range_no_year":
                y = datetime.now().year
                return f"{y:04d}-{int(g['m2']):02d}-{int(g['d2']):02d}"

            if name == "m_range_no_year":
                y = datetime.now().year
                return f"{y:04d}-{int(g['m2']):02d}-00"

            if name in ("y_to_y_range", "year_range", "year_range_loose", "y_zeromd", "year_before"):
                y = int(g.get("y2") or g.get("y"))
                return f"{y:04d}-00-00"

            if name == "year_decade":
                y = int(g["y"]) + 9
                return f"{y:04d}-00-00"

            if name in ("year_after", "year_end"):
                return f"{int(g['y']):04d}-12-31"

            if name == "year_begin":
                return f"{int(g['y']):04d}-01-01"

            if name in ("ym_to_ym_cross", "ym_to_m_same"):
                y = int(g.get("y2") or g.get("y"))
                return f"{y:04d}-{int(g['m2']):02d}-00"

            if name == "y_to_now":
                cur = datetime.now().year
                return f"{cur:04d}-01-01"

            if name in (
            "ymd_with_clockish", "ymd_with_period", "ymd", "iso_date_dash", "ymd_d_to_d", "ymd_with_dangri"):
                y = int(g["y"])
                if FORCE_DEFAULT_YEAR_FOR_TIME_TAIL and name in ("ymd_with_clockish", "ymd_with_period"):
                    # 如果年份过小或未指定完整（部分正则允许），使用默认年
                    # 这里简化处理：直接信任正则组，若想强转需判断 g['y']
                    pass
                return f"{y:04d}-{int(g['m']):02d}-{int(g.get('d') or g.get('d2')):02d}"

            if name == "ym_month_end":
                y = int(g["y"]);
                m = int(g["m"])
                d = _last_day_clamped(y, m, MONTH_END_PREF_DAY)
                return f"{y:04d}-{m:02d}-{d:02d}"

            if name in ("ym", "m_only", "this_year_month", "ym_with_comment", "ym_before", "ym_begin"):
                y = int(g.get("y", DEFAULT_YEAR))
                return f"{y:04d}-{int(g['m']):02d}-00"

            if name == "y_only":
                return f"{int(g['y']):04d}-00-00"

            if name == "md":
                return f"{DEFAULT_YEAR:04d}-{int(g['m']):02d}-{int(g['d']):02d}"

            if name == "multi_years":
                y2 = int(g.get("y2") or g.get("y1"))
                return f"{y2:04d}-00-00"

            if name == "century_decade":
                c = int(g["c"]);
                d0 = int(g["d"])
                y = (c - 1) * 100 + d0 * 10
                return f"{y:04d}-00-00"

            if name in ("this_year_half", "thisyear_half_alt", "y_first_three_q"):
                y = int(g.get("y", CURR_YEAR))
                return f"{y:04d}-00-00"

            if name in ("ym_with_xun", "m_with_xun_no_year"):
                y = int(g.get("y", CURR_YEAR))
                m2 = int(g["m"])
                day_map = {"上旬": 5, "中旬": 15, "下旬": 25, "中下旬": 20, "中上旬": 10}
                d0 = _clamp_day(y, m2, day_map[g["x"]])
                return f"{y:04d}-{m2:02d}-{d0:02d}"

            if name == "md_to_d_no_year":
                y = CURR_YEAR
                return f"{y:04d}-{int(g['m']):02d}-{int(g['d2']):02d}"

        except Exception:
            continue

    # 兜底：找第一个 yyyy 年作为结尾年的占位
    m_year_segment = re.search(r"(\d{4})年", s)
    if m_year_segment:
        year = int(m_year_segment.group(1))
        return f"{year:04d}-01-01"

    return ""


# ==========================
# 5. 结构转换函数 (Evolution)
# ==========================

def convert_event_simple(event: dict) -> dict:
    """
    将单个事件转换为简化格式，时间解析优先 moment，再 period。
    """
    rc = event.get("relationship_characteristics", {}) or {}
    roles = []
    for k in ["subject_person", "subject_organization", "related_party_person", "related_party_organization"]:
        v = rc.get(k, {}) or {}
        if v.get("name"):
            roles.append({
                "实体": v.get("name"),
                "角色": v.get("role", ""),
                "实体情感标记": v.get("emotion", "")
            })

    tp = event.get("time_position", {}) or {}
    raw_moment = tp.get("moment", "") or ""
    raw_period = tp.get("period", "") or ""

    source_time_str = raw_moment if raw_moment else raw_period

    # 使用本模块的解析器
    t_time_norm = normalize_date_from_raw(source_time_str, event)

    event_name = event.get("event_name", "") or ""
    if t_time_norm:
        event_id_norm = f"{t_time_norm}-{event_name}"
    else:
        event_id_norm = event_name

    return {
        "事件id": event_id_norm,
        "event_id": event.get("event_id", ""),
        "事件类型t_type": event.get("attribute_characteristics", {}).get("event_type", ""),
        "时间t_time": t_time_norm,
        "地点t_location": event.get("space_position", {}).get("place_name", ""),
        "触发词": event.get("trigger_word", []),
        "参与方t_role": roles
    }


def convert_document_simple(doc: dict) -> dict:
    """把整个文档（多个事件）转换为目标简化结构"""
    return {
        "_id": doc.get("_id", ""),
        "title": doc.get("title", ""),
        "text": doc.get("content", ""),
        "status": "0",
        "事件基本信息": [convert_event_simple(e) for e in doc.get("structured_data", {}).get("events", [])]
    }