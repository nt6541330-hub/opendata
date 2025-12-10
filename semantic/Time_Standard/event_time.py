# semantic/Time_Standard/event_time.py
import re
import traceback
from datetime import datetime, timedelta
from calendar import monthrange
from pymongo import MongoClient

# 【修改点 1】引入统一配置
from config.settings import settings
# 引入同级目录下的转换工具（需确保 semantic/conversion/conversion.py 存在）
from semantic.conversion.conversion import normalize_date_from_raw

# ==========================
# 1. 常量定义 (原 config.time 内容整合)
# ==========================
DEFAULT_YEAR = datetime.now().year
RECENT_DAYS_OFFSET = 3  # 模糊词"近日"对应的回溯天数
UPCOMING_DAYS_OFFSET = 3  # "即将"对应的未来天数
RECENT_PAST_DAYS_OFFSET = 7  # "日前"对应的回溯天数
MONTH_END_PREF_DAY = 1  # 月末/仅月时的默认日期（1表示月初，31表示月末）
FORCE_DEFAULT_YEAR_FOR_TIME_TAIL = True  # 当含有具体时间(时分秒)但缺年份时，是否强制使用默认年份

# 模糊时间词表
FUZZY = [
    '昨天', '昨日', '今天', '今日', '明天', '明日',
    '上月', '本月', '去年', '今年', '明年',
    '近期', '近段时间', '最近', '近日'
]

# 固定节假日映射 (示例，可按需补充)
HOLIDAY_FIXED = {
    "元旦": datetime(DEFAULT_YEAR, 1, 1),
    "劳动节": datetime(DEFAULT_YEAR, 5, 1),
    "国庆": datetime(DEFAULT_YEAR, 10, 1),
    "中秋": datetime(DEFAULT_YEAR, 9, 17),  # 需每年更新，此处仅为示例
}

# ==========================
# 2. 数据库连接
# ==========================
# 【修改点 2】使用 settings 配置
client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]
collection = db[settings.EVENT_NODE_COLLECTION]
FIELD = "time_position_moment"

# ==========================
# 3. 辅助常量与正则
# ==========================
DASH_CLASS = r"[—–\-]"
SEP_MD = rf"(?:{DASH_CLASS}|至|到)"
SP = r"\s*"
CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4}

TODAY = datetime.now().date()
TODAY_DT = datetime(TODAY.year, TODAY.month, TODAY.day)
CURR_YEAR = TODAY.year
CURR_MONTH = TODAY.month

# 中文数字（日）
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

# 正则匹配模式表
PATS = [
    # 完整范围： yyyy年m月d日 至 yyyy年m月d日
    ("ymd_range_both_years", re.compile(
        rf"^{SP}(?P<y1>\d{{4}})年(?P<m1>\d{{1,2}})月(?P<d1>\d{{1,2}})日{SP}{SEP_MD}{SP}(?P<y2>\d{{4}})年(?P<m2>\d{{1,2}})月(?P<d2>\d{{1,2}})日{SP}$"
    )),
    # 同年范围： yyyy年m月d日 至 m月d日
    ("ymd_range_same_year", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m1>\d{{1,2}})月(?P<d1>\d{{1,2}})日{SP}{SEP_MD}{SP}(?P<m2>\d{{1,2}})月(?P<d2>\d{{1,2}})日{SP}$"
    )),
    # 年 + 紧凑月区间： 2025年1-9月
    ("ym_month_compact_range", re.compile(
        rf"^{SP}(?P<y>\d{{4}})年(?P<m1>\d{{1,2}}){DASH_CLASS}(?P<m2>\d{{1,2}})月{SP}$"
    )),
    # 无年但月日范围： 9月17日至9月19日
    ("md_range_no_year", re.compile(
        rf"^{SP}(?P<m1>\d{{1,2}})月{SP}(?P<d1>\d{{1,2}})日{SP}{SEP_MD}{SP}(?P<m2>\d{{1,2}})月{SP}(?P<d2>\d{{1,2}})日{SP}$"
    )),
    # 无年月区间： 4月-5月
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
# 4. 辅助函数
# ==========================

def _cn_day_to_int(s: str):
    return _CN_NUM_D.get(s)


def _pre_clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u3000", " ").strip()
    return s


def _last_day_clamped(y: int, m: int, pref_day: int = MONTH_END_PREF_DAY) -> int:
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


def _infer_year_month_from_doc(doc: dict, raw: str = None):
    """
    尝试从当前 doc 的子事件/其它字段中推断年和月
    """
    if not isinstance(doc, dict):
        return None, None

    candidates = []
    # 优先扫描子事件
    for key in ("subevents", "children", "events"):
        lst = doc.get(key)
        if isinstance(lst, list):
            for it in lst:
                if isinstance(it, dict):
                    for fk in ("time_position_moment", "time", "datetime", "time_position"):
                        v = it.get(fk)
                        if isinstance(v, str):
                            candidates.append(v)
                    if "raw" in it and isinstance(it["raw"], str):
                        candidates.append(it["raw"])
    # 扫描其他字符串字段
    for k, v in doc.items():
        if isinstance(v, str):
            candidates.append(v)

    # 搜索候选词
    for text in candidates:
        if not text: continue
        m = re.search(r"(?P<y>\d{4})年\s*(?P<m>\d{1,2})月", text)
        if m:
            return int(m.group("y")), int(m.group("m"))
        m2s = list(re.finditer(r"(?P<m>\d{1,2})月", text))
        if m2s:
            return None, int(m2s[-1].group("m"))
    return None, None


def normalize_time(raw: str, doc: dict = None):
    """解析各种日期字符串为 datetime；失败返回 None"""
    if not raw:
        return None
    s = _pre_clean(raw)

    # 1. 模糊词处理
    if any(k in s for k in FUZZY):
        return (datetime.now() - timedelta(days=RECENT_DAYS_OFFSET)).replace(hour=0, minute=0, second=0, microsecond=0)
    if re.fullmatch(rf"{SP}(不久前|日前){SP}", s):
        d = TODAY - timedelta(days=RECENT_PAST_DAYS_OFFSET)
        return datetime(d.year, d.month, d.day)
    if re.fullmatch(rf"{SP}即将到来{SP}", s):
        d = TODAY + timedelta(days=UPCOMING_DAYS_OFFSET)
        return datetime(d.year, d.month, d.day)
    if re.fullmatch(rf"{SP}中秋小长假期间{SP}", s):
        if "中秋" in HOLIDAY_FIXED:
            d = HOLIDAY_FIXED["中秋"]
            return datetime(d.year, d.month, d.day)

    # 2. 中文“几号/几日”或数字“日/号”
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

        # 同串前推
        y_prev, m_prev = _search_prev_ym_in_text(raw, day_pos=day_pos)
        if y_prev is not None and m_prev is not None:
            d0 = _clamp_day(y_prev, m_prev, d)
            return datetime(y_prev, m_prev, d0)
        if m_prev is not None:
            d0 = _clamp_day(DEFAULT_YEAR, m_prev, d)
            return datetime(DEFAULT_YEAR, m_prev, d0)

        # 文档上下文推断
        if doc:
            y_doc, m_doc = _infer_year_month_from_doc(doc, raw)
            if y_doc is not None and m_doc is not None:
                d0 = _clamp_day(y_doc, m_doc, d)
                return datetime(y_doc, m_doc, d0)
            if m_doc is not None:
                d0 = _clamp_day(DEFAULT_YEAR, m_doc, d)
                return datetime(DEFAULT_YEAR, m_doc, d0)

        # 默认
        d0 = _clamp_day(DEFAULT_YEAR, CURR_MONTH, d)
        return datetime(DEFAULT_YEAR, CURR_MONTH, d0)

    # 3. 正则列表匹配
    for name, pat in PATS:
        m = pat.fullmatch(s)
        if not m: continue
        g = m.groupdict()
        try:
            # 含完全年月日范围 -> 返回结束端具体日
            if name in ("ymd_range_both_years", "ymd_range_same_year"):
                return datetime(int(g.get("y2", g.get("y"))), int(g["m2"]), int(g["d2"]))

            if name == "ym_month_compact_range":
                return datetime(int(g["y"]), int(g["m2"]), 1)

            if name in ("md_range_no_year", "m_range_no_year"):
                y = datetime.now().year
                d2 = int(g.get("d2", 1))
                return datetime(y, int(g["m2"]), d2)

            if name in ("y_to_y_range", "year_begin", "y_to_now", "y_only", "y_zeromd", "year_before"):
                y = int(g.get("y2") or g.get("y"))
                return datetime(y, 1, 1)

            if name == "year_decade":
                return datetime(int(g["y"]) + 9, 1, 1)

            if name == "year_after":
                return datetime(int(g["y"]), 12, 31)

            if name == "year_end":
                return datetime(int(g["y"]), 12, 31)

            if name in ("ym_to_ym_cross", "ym_to_m_same"):
                y = int(g.get("y2") or g.get("y"))
                return datetime(y, int(g["m2"]), 1)

            if name == "ymd_with_clockish":
                y = int(g["y"]);
                m2 = int(g["m"]);
                d0 = int(g["d"])
                has_time_tail = bool(g.get("h") or g.get("h2") or g.get("min"))
                if FORCE_DEFAULT_YEAR_FOR_TIME_TAIL and has_time_tail:
                    y = DEFAULT_YEAR
                return datetime(y, m2, d0)

            if name == "ymd_with_period":
                y = int(g["y"]);
                m2 = int(g["m"]);
                d0 = int(g["d"])
                period_words = ("上午", "下午", "晚间", "晚上", "晚", "清晨", "早上", "中午", "傍晚", "凌晨", "夜间", "深夜")
                has_period = any(w in s for w in period_words)
                if FORCE_DEFAULT_YEAR_FOR_TIME_TAIL and has_period:
                    y = DEFAULT_YEAR
                return datetime(y, m2, d0)

            if name == "ym_month_end":
                y = int(g["y"]);
                m2 = int(g["m"])
                day = _last_day_clamped(y, m2, MONTH_END_PREF_DAY)
                return datetime(y, m2, day)

            if name in ("iso_date_dash", "ymd", "ymd_d_to_d", "ymd_with_dangri"):
                return datetime(int(g["y"]), int(g["m"]), int(g.get("d") or g.get("d2")))

            if name in ("ym", "m_only", "this_year_month", "ym_with_comment", "ym_before", "ym_begin"):
                y = int(g.get("y", DEFAULT_YEAR))
                return datetime(y, int(g["m"]), 1)

            if name == "md":
                return datetime(DEFAULT_YEAR, int(g["m"]), int(g["d"]))

            if name == "multi_years":
                y2 = int(g.get("y2") or g.get("y1"))
                return datetime(y2, 1, 1)

            if name in ("year_range", "year_range_loose"):
                return datetime(int(g["y2"]), 1, 1)

            if name == "century_decade":
                c = int(g["c"]);
                d0 = int(g["d"])
                return datetime((c - 1) * 100 + d0 * 10, 1, 1)

            if name in ("this_year_half", "thisyear_half_alt", "y_first_three_q"):
                y = int(g.get("y", CURR_YEAR))
                return datetime(y, 3, 1)

            if name in ("ym_with_xun", "m_with_xun_no_year"):
                y = int(g.get("y", CURR_YEAR))
                m2 = int(g["m"])
                day_map = {"上旬": 5, "中旬": 15, "下旬": 25, "中下旬": 20, "中上旬": 10}
                d0 = _clamp_day(y, m2, day_map[g["x"]])
                return datetime(y, m2, d0)

            if name == "md_to_d_no_year":
                return datetime(CURR_YEAR, int(g["m"]), int(g["d2"]))

        except ValueError:
            return None

    return None


def run_update(limit=0):
    """
    把 FIELD 为字符串的记录转成 日期；未识别的统一写成“今天”
    """
    print("[EventTime] 开始时间标准化...")

    # 查找所有时间字段为字符串的记录
    base_query = {FIELD: {"$type": "string"}}
    cursor = collection.find(base_query, limit=limit) if limit > 0 else collection.find(base_query)

    ok = bad = errs = 0
    for doc in cursor:
        try:
            raw = doc.get(FIELD, "")
            raw_str = raw.strip() if isinstance(raw, str) else None

            # 1. 尝试使用 conversion 的通用解析器
            norm_str = normalize_date_from_raw(raw_str, doc)
            if norm_str:
                try:
                    dt = datetime.strptime(norm_str, "%Y-%m-%d")
                except ValueError:
                    dt = None
            else:
                dt = None

            # 2. 如果通用解析失败，使用本地 normalize_time
            if not dt:
                dt = normalize_time(raw_str, doc)

            # 3. 更新数据库
            if dt is None:
                # 无法识别 -> 设为今天
                dt = TODAY_DT
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {FIELD: dt, f"{FIELD}_raw": raw_str}}
                )
                bad += 1
            else:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {FIELD: dt, f"{FIELD}_raw": raw_str}}
                )
                ok += 1
        except Exception as e:
            errs += 1
            traceback.print_exc()

    print(f"[EventTime] 完成：解析成功 {ok} 条，未识别改为今日 {bad} 条，更新失败 {errs} 条。")


if __name__ == "__main__":
    run_update(limit=0)