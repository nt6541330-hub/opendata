# semantic/Hot_topic/hotspot.py
# -*- coding: utf-8 -*-
"""
Hot Events -> Same-Doc Two Reports (classic & wiki)
优化最终版：
1. [性能] 复用 Embedding 模型进行关键词提取，避免循环加载 KeyBERT。
2. [逻辑] 支持全量数据扫描，无视时间窗口。
3. [存储] 移除绘图和文件操作，报告全文存入数据库。
"""

import os
import re
import json
import math
import hashlib
import argparse
import datetime as dt
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from pymongo import MongoClient
from bson import ObjectId
from dateutil import parser as dtparser

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

# 引入统一配置
from config.settings import settings

# ==========================================
# 【环境配置】必须在导入深度学习库前设置
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- 可选库（自动降级） ----
USE_HDBSCAN = True
try:
    import hdbscan
except Exception:
    USE_HDBSCAN = False

USE_KEYBERT = True
try:
    from keybert import KeyBERT
except Exception:
    USE_KEYBERT = False

# ------------------ 工具函数 ------------------
SEP = "。"
SAFEWORDS = {"重磅", "震撼", "史无前例", "惨", "怒", "惊天", "核打击", "歼灭", "血洗"}
SENT_SPLIT_RE = re.compile(r"[。！？；;\n]+")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def norm_text(s: str) -> str:
    if not s: return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"http[s]?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def concat_for_embed(title, content, max_chars=800):
    t = norm_text(title or "")
    c = norm_text(content or "")
    if len(c) > max_chars: c = c[:max_chars]
    return (t + SEP + c).strip()


def parse_date_any(x):
    if x is None: return dt.datetime.utcnow()
    if isinstance(x, dt.datetime): return x
    if isinstance(x, str):
        try:
            return dtparser.parse(x)
        except Exception:
            pass
    return dt.datetime.utcnow()


def to_datestr(d):
    if isinstance(d, dt.datetime): d = d.date()
    return d.strftime("%Y-%m-%d")


def recency_score(latest_date: dt.date, today: dt.date) -> float:
    dd = max(0, (today - latest_date).days)
    return math.exp(- dd / settings.HOTSPOT_RECENCY_TAU_DAYS)


def diversity_ratio(sources: list, total: int) -> float:
    if total == 0: return 0.0
    return len(set([s for s in sources if s])) / total


def stable_event_id(news_ids: list) -> str:
    m = hashlib.md5()
    for x in sorted(news_ids):
        m.update(x.encode("utf-8"))
    return m.hexdigest()


def choose_event_time(dates: list) -> dict:
    if not dates: return {"type": "unknown", "value": None}
    dmin, dmax = min(dates), max(dates)
    if dmin == dmax:
        return {"type": "point", "value": to_datestr(dmin)}
    if (dmax - dmin).days <= 3:
        cnt = Counter(dates)
        peak = max(cnt.items(), key=lambda x: x[1])[0]
        return {"type": "point", "value": to_datestr(peak)}
    return {"type": "range", "value": [to_datestr(dmin), to_datestr(dmax)]}


def derive_source_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def parse_date_from_text(text: str):
    if not text: return None
    s = text.replace("年", "-").replace("月", "-").replace("日", " ").replace("时间：", " ")
    s = s.replace("時", " ").replace("：", ":")
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:\s+(\d{1,2}:\d{2}:\d{2}))?", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        t = m.group(4) or "00:00:00"
        try:
            return dt.datetime.strptime(f"{y:04d}-{mo:02d}-{d:02d} {t}", "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    return None


# -------- 嵌入与聚类 --------
def embed_docs(model: SentenceTransformer, docs):
    texts = [concat_for_embed(d.get("title", ""), d.get("content", "")) for d in docs]
    # batch_size 可根据显存调整
    emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return np.asarray(emb, dtype=np.float32)


def cluster_embeddings(emb: np.ndarray):
    if len(emb) == 0: return np.array([])
    # 降维处理
    n_components = min(50, emb.shape[1])
    if len(emb) > n_components:
        X = PCA(n_components=n_components).fit_transform(emb)
    else:
        X = emb

    min_cluster_size = settings.HOTSPOT_MIN_CLUSTER_SIZE

    if USE_HDBSCAN and len(emb) >= min_cluster_size * 2:
        cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2, metric="euclidean")
        labels = cl.fit_predict(X)
    else:
        k = max(2, len(emb) // max(2, min_cluster_size * 2))
        cl = AgglomerativeClustering(n_clusters=k)
        labels = cl.fit_predict(X)
    return labels


# -------- 关键词与标题 (优化版) --------
try:
    import jieba

    HAS_JIEBA = True
except Exception:
    HAS_JIEBA = False


def _zh_tokenize(s: str):
    toks = [t.strip() for t in jieba.lcut(s, cut_all=False) if t.strip()]
    toks = [t for t in toks if 2 <= len(t) <= 8]
    return toks


# 【优化点】增加 model 参数，复用已加载的模型
def extract_keywords(texts: list, model=None, topk=5) -> list:
    texts = [t for t in texts if t]
    if not texts: return []

    if USE_KEYBERT:
        try:
            # 如果传入了预加载的 model (SentenceTransformer 对象)，直接复用
            if model:
                kb = KeyBERT(model=model)
            else:
                # 否则重新加载 (慢)
                kb = KeyBERT(model=settings.HOTSPOT_EMB_MODEL)

            joined = "。".join(texts)
            # 提取关键词
            cands = kb.extract_keywords(joined, top_n=max(topk, 8), keyphrase_ngram_range=(1, 3), stop_words=None)
            out = [re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", "", w).strip() for w, _ in cands]
            out = [w for w in out if 2 <= len(w) <= 10]
            if out: return out[:topk]
        except Exception:
            pass

    # 降级方案：TF-IDF
    if HAS_JIEBA:
        vec = TfidfVectorizer(tokenizer=_zh_tokenize, token_pattern=None, max_features=8000, ngram_range=(1, 2))
    else:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=8000)

    try:
        X = vec.fit_transform(texts + texts)
    except ValueError:
        return []

    scores = np.asarray(X.sum(0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    idx = scores.argsort()[::-1]
    kws = []
    for i in idx:
        k = vocab[i]
        k = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", "", k).strip()
        if 2 <= len(k) <= 10:
            kws.append(k)
        if len(kws) >= topk: break
    return kws


def clean_keywords(kws: list, max_len_each=8, max_k=5) -> list:
    seen, out = set(), []
    for k in kws:
        k = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", "", (k or "")).strip()
        if not k: continue
        if len(k) > max_len_each: k = k[:max_len_each]
        if 2 <= len(k) and k not in seen:
            seen.add(k);
            out.append(k)
        if len(out) >= max_k: break
    return out


def normalize_title(t: str) -> str:
    t = (t or "").strip().replace("  ", " ")
    t = re.sub(r"[，,]\s*", "，", t).strip("。!！?？~、 ")
    t = t.replace("“", "「").replace("”", "」")
    return t


def enforce_title_len(title: str, key_terms: list, max_len=20) -> str:
    t = (title or "").strip()
    if not t or len(t) <= max_len: return t
    parts = [p for p in re.split(r"[，,、:：;；\-—\s]+", t) if p]
    if parts:
        scored = []
        for p in parts:
            cov = sum(1 for k in key_terms if k and k in p)
            scored.append((cov, -len(p), p))
        scored.sort(reverse=True)
        acc, cur = [], 0
        for _, __, p in scored:
            add = (1 if acc else 0) + len(p)
            if cur + add <= max_len:
                acc.append(p);
                cur += add
        if acc:
            t2 = "、".join(acc)
            if len(t2) <= max_len: return t2
    t = t[:max_len]
    return t.rstrip("，、:：;；-— ")


def title_score(t: str, key_terms: list, entities: list, max_len=20) -> float:
    if not t: return -1e9
    tt = normalize_title(t)
    if any(w in tt for w in SAFEWORDS): return -1e6
    L = len(tt)
    length_pen = -1.0 * max(0, L - max_len)
    cov = sum(1 for k in key_terms if k and k in tt)
    ent = 1 if any(e for e in entities if e and e in tt) else 0
    action = 1 if re.search(r"(演习|会见|访台|对峙|制裁|表决|通过|爆发|坠毁|集会|抗议|谈判|合作|签署|启动|通报|搜救|停火|冲突)", tt) else 0
    return 2.0 * cov + 1.2 * ent + 0.8 * action + length_pen


def pick_best_title(cands: list, key_terms: list, entities: list, max_len=20) -> str:
    seen, scored = set(), []
    for t in cands or []:
        t = normalize_title(t)
        if not t or t in seen: continue
        seen.add(t)
        if len(t) > 2 * max_len: continue
        scored.append((title_score(t, key_terms, entities, max_len), t))
    if not scored: return ""
    scored.sort(reverse=True)
    return scored[0][1]


def ollama_generate_json(prompt: str, timeout=240) -> dict:
    url = settings.OLLAMA_HOST.rstrip("/") + "/api/generate"
    try:
        resp = requests.post(url, json={
            "model": settings.HOTSPOT_OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.2, "top_p": 0.95}
        }, stream=True, timeout=timeout)
        resp.raise_for_status()
        buf = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line: continue
            try:
                obj = json.loads(line);
                buf += obj.get("response", "")
                if obj.get("done"): break
            except Exception:
                continue
        m = re.search(r"\{[\s\S]*\}", buf)
        if not m: return {}
        return json.loads(m.group(0))
    except Exception:
        return {}


LLM_PROMPT_TITLE = """你是“新闻事件命名器”。根据下列代表性标题、关键词与时间范围，生成5-8个中文候选标题（中性、信息密度高、≤20字、无句号/感叹号），只输出JSON：
{{"titles":[...], "keywords_suggested":[...]}}
代表性标题：
{titles}
关键词（高→低）：{key_terms}
实体：{entities}
时间范围：{time_range}
"""


def make_event_title_llm(top_titles, key_terms, entities, time_range, fallback_func, max_len=20):
    prompt = LLM_PROMPT_TITLE.format(
        titles="\n".join("- " + t for t in top_titles[:10]),
        key_terms=", ".join(key_terms[:12]),
        entities=", ".join(entities[:10]),
        time_range=time_range or ""
    )
    titles, kws_llm = [], []
    try:
        obj = ollama_generate_json(prompt)
        titles = obj.get("titles") or []
        kws_llm = obj.get("keywords_suggested") or []
    except Exception:
        pass
    best = pick_best_title(titles, key_terms, entities, max_len=max_len) if titles else ""
    if not best:
        best = make_event_title_fallback(key_terms, top_titles)
        kws_llm = []
    return best, kws_llm


def make_event_title_fallback(kws: list, top_titles: list) -> str:
    kws = clean_keywords(kws, max_len_each=8, max_k=5)
    if kws:
        if len(kws) >= 3: return f"{kws[0]}、{kws[1]}与{kws[2]}"
        if len(kws) == 2: return f"{kws[0]}与{kws[1]}"
        return kws[0]
    if not top_titles: return "热点事件"
    s1, s2 = min(top_titles, key=len), max(top_titles, key=len)
    prefix = ""
    for i, ch in enumerate(s1):
        if s2[i:i + 1] != ch: prefix = s1[:i]; break
    else:
        prefix = s1
    prefix = prefix.strip(" -，。、《》")
    if len(prefix) < 4: prefix = (top_titles[0][:16] if top_titles else "热点事件")
    return prefix


# ------------------ 事实库与校验 ------------------
def build_fact_bank(news, keywords, max_sentences=120):
    kwset = set(keywords or [])
    sents, refs, dates = [], [], set()
    for d in news:
        url = (d.get("url") or "").strip()
        if url: refs.append(url)
        try:
            dates.add(to_datestr(parse_date_any(d.get("published_at"))))
        except Exception:
            pass

        chunks = []
        t = (d.get("title") or "").strip()
        c = (d.get("content") or "").strip()
        if t: chunks.append(t)
        if c:
            chunks.extend([x.strip() for x in SENT_SPLIT_RE.split(c) if x and 8 <= len(x) <= 120])

        for s in chunks:
            hit = sum(1 for k in kwset if k and k in s)
            has_num = 1 if re.search(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}", s) or re.search(r"\d+", s) else 0
            score = 2 * hit + has_num
            if score > 0:
                sents.append((score, s))

    sents.sort(key=lambda x: (-x[0], -len(x[1])))
    seen = set();
    picked = []
    for _, s in sents:
        ss = re.sub(r"\s+", "", s)
        if ss in seen: continue
        seen.add(ss)
        picked.append(s)
        if len(picked) >= max_sentences: break

    fact_lines = [f"- {s}" for s in picked]
    fact_bank_text = "\n".join(fact_lines[:max_sentences])
    references = sorted(set([u for u in refs if u]))
    allowed_dates = sorted(set(dates))
    return fact_bank_text, references, allowed_dates


def filter_progression_by_dates(prog_list, allowed_dates, max_items=8):
    out = []
    allowset = set(allowed_dates or [])
    for item in prog_list or []:
        d = (item.get("date") or "").strip()
        txt = f"{d} {item.get('what', '')}"
        ds = set(DATE_RE.findall(txt))
        if not ds or ds.issubset(allowset):
            out.append(item)
        if len(out) >= max_items: break
    return out


# ------------------ 报告 Prompt ------------------
PROMPT_REPORT_CLASSIC = """你是“新闻专题报告撰稿人”。仅基于【事实库】改写，禁止添加事实库之外的具体时间、数字、人名、机构名。
如素材不足请写“（素材不足）”，不要自行补充。只输出合法 JSON。
【事实库】
%(fact_bank)s

{
  "overall_summary": "150-250字整体摘要（中性，含关键结论+直接证据）",
  "background": "500-800字背景/前史/争点（说明与本次时间范围关系）",
  "progression": [
    {"date":"YYYY-MM-DD","title":"<=20字","what":"40-80字","evidence":["要点1","要点2"]}
  ],
  "impacts": {
    "politics": ["3-6条（<=40字/条）"],
    "diplomacy": ["..."],
    "military": ["..."],
    "economy": ["..."],
    "public_opinion": ["..."]
  },
  "detection_status": {
    "first_article_date": "%(first_article_date)s",
    "event_created_at": "%(event_created_at)s",
    "detection_latency_days": %(detection_latency_days)d,
    "cluster_size": %(cluster_size)d,
    "unique_sources": %(unique_sources)d,
    "dup_title_ratio": %(dup_title_ratio).3f,
    "method": "HDBSCAN/Agglomerative",
    "params": {"min_cluster_size": %(min_cluster_size)d}
  },
  "key_targets": [],
  "evolution": {"phases": [], "explanation":"（如启用时填写）"},
  "forecast": {"horizon_days": %(horizon)d, "volume_forecast": [], "scenarios": [], "caveats":""},
  "references": []
}
硬性规则：中性、无感叹号；progression ≥3；仅输出合法 JSON。
【事件】%(event_title)s
【时间范围】%(time_range)s
【关键词】%(keywords)s
【代表性标题（≤12条）】
%(top_titles)s
【统计】总量=%(news_count)d；来源数=%(source_cnt)d；日均=%(daily_avg).2f；峰值日=%(peak_date)s(%(peak_count)d)
"""

PROMPT_REPORT_WIKI = """你是“百科体专题撰稿人”。仅基于【事实库】改写，禁止添加事实库之外的具体时间、数字、人名、机构名。
如素材不足请写“（素材不足）”，不要自行补充。只输出合法 JSON。
【事实库】
%(fact_bank)s

{
  "lead": "120-200字导语/序言（中性客观、信息密集）",
  "background_and_precedents": {
    "congress_taiwan_relations": "200-350字；国会与台湾关系要点（含关键法案/访问先例）",
    "pelosi_taiwan_relations": "150-300字；裴洛西与台湾的历史关联"
  },
  "pretrip_and_preparations": "250-400字；邀访磋商与行前消息（时间线要素+来源要点）",
  "personnel": ["访问团与主要接待官员（若可归纳；每条≤30字）"],
  "itinerary": {
    "before_arrival": [{"date":"YYYY-MM-DD","event":"<=20字","detail":"40-80字"}],
    "arrival_night":  [{"date":"YYYY-MM-DD","event":"<=20字","detail":"40-80字"}],
    "next_day":       [{"date":"YYYY-MM-DD","event":"<=20字","detail":"40-80字"}],
    "departure":      [{"date":"YYYY-MM-DD","event":"<=20字","detail":"40-80字"}],
    "after":          [{"date":"YYYY-MM-DD","event":"<=20字","detail":"40-80字"}]
  },
  "related_military_actions": ["3-6条；相关军事行动与时间点（≤40字/条）"],
  "reactions": {"PRC": [], "Taiwan": [], "US": [], "Others": []},
  "related_events_and_impacts": {"cyberattacks": [], "sanctions": [], "trade": [], "follow_on_visits": []},
  "detection_status": {
    "first_article_date": "%(first_article_date)s",
    "event_created_at": "%(event_created_at)s",
    "detection_latency_days": %(detection_latency_days)d,
    "cluster_size": %(cluster_size)d,
    "unique_sources": %(unique_sources)d,
    "dup_title_ratio": %(dup_title_ratio).3f,
    "method": "HDBSCAN/Agglomerative",
    "params": {"min_cluster_size": %(min_cluster_size)d}
  },
  "key_targets": [],
  "evolution": {"phases": [], "explanation": ""},
  "forecast": {"horizon_days": 0, "volume_forecast": [], "scenarios": [], "caveats": ""},
  "references": []
}
要求：避免夸张和立场性措辞；尽量基于代表性标题中的明确事实。
【事件】%(event_title)s
【时间范围】%(time_range)s
【关键词】%(keywords)s
【代表性标题（≤12条）】
%(top_titles)s
【统计】总量=%(news_count)d；来源数=%(source_cnt)d；日均=%(daily_avg).2f；峰值日=%(peak_date)s(%(peak_count)d)
"""


# ------------------ 主类 ------------------
class HotEventsTwoReportsStrict:
    def __init__(self, source_collections=None):
        self.client = MongoClient(settings.MONGO_URI)
        self.db = self.client[settings.MONGO_DB_NAME]

        # 优化：允许传入所有源集合名称，若不传则从配置读取
        self.source_collections = source_collections if source_collections else settings.COL_SRC_LIST
        self.col_out = self.db[settings.KNOWLEDGE_COLLECTION_EVENT]

        # 移除本地目录创建
        # os.makedirs(settings.HOTSPOT_OUT_DIR, exist_ok=True)

        if settings.HTTP_PROXY:
            os.environ["HTTP_PROXY"] = settings.HTTP_PROXY
            os.environ["HTTPS_PROXY"] = settings.HTTP_PROXY

        print(f"[Hotspot] Loading Embedding Model: {settings.HOTSPOT_EMB_MODEL}")
        self.embedder = SentenceTransformer(settings.HOTSPOT_EMB_MODEL)

    # --- 优化核心：多源加载 (全量扫描模式) ---
    def load_recent_docs(self, days_window=None, limit_per_col=None):
        """
        加载文档用于聚类。
        修改说明：移除时间窗口限制，进行全量扫描。
        """
        # 如果未指定，给予一个较大的默认值，确保能覆盖现有数据量
        if limit_per_col is None:
            # 尊重 settings 中的配置，如果没配则默认 5000
            limit_per_col = getattr(settings, 'HOTSPOT_BATCH_LIMIT', 5000)

            # 打印一下实际使用的限制
        print(f"[Hotspot] 数据加载限制: 每源 {limit_per_col} 条")

        proj = {"title": 1, "content": 1, "url": 1, "published_at": 1, "scraped_at": 1, "source": 1}


        all_docs = []

        for col_name in self.source_collections:
            try:
                col = self.db[col_name]
                # 【全量扫描】query为空，匹配所有数据
                query = {}
                # 按发布时间倒序取最新的 limit 条
                docs = list(col.find(query, proj).sort("published_at", -1).limit(limit_per_col))

                for d in docs:
                    d["_id_str"] = str(d["_id"])
                    pub = d.get("published_at") or parse_date_from_text(d.get("content", "")) or d.get("scraped_at")
                    d["published_at"] = parse_date_any(pub)

                    if not d.get("source"):
                        if "cctv" in col_name.lower():
                            d["source"] = "CCTV"
                        elif "toutiao" in col_name.lower():
                            d["source"] = "头条"
                        elif "weibo" in col_name.lower():
                            d["source"] = "微博"
                        elif "xinhua" in col_name.lower():
                            d["source"] = "新华网"
                        elif "cnn" in col_name.lower():
                            d["source"] = "CNN"
                        else:
                            d["source"] = derive_source_from_url(d.get("url", ""))

                    all_docs.append(d)
            except Exception as e:
                print(f"[Hotspot] Warning: Load from {col_name} failed: {e}")

        print(f"[Hotspot] 融合了 {len(self.source_collections)} 个源, 共加载 {len(all_docs)} 条新闻 (模式: 全量/无时间窗口)")
        return all_docs

    def build_events(self, docs):
        if not docs: return []
        emb = embed_docs(self.embedder, docs)
        labels = cluster_embeddings(emb)
        today = dt.datetime.utcnow().date()
        events = []

        max_title_len = settings.HOTSPOT_MAX_TITLE_LEN
        min_cluster_size = settings.HOTSPOT_MIN_CLUSTER_SIZE

        for lb in sorted(set(labels)):
            if lb == -1: continue
            idx = np.where(labels == lb)[0].tolist()
            if len(idx) < min_cluster_size: continue
            group = [docs[i] for i in idx]
            titles = [g.get("title", "") for g in group]
            bodies = [concat_for_embed(g.get("title", ""), g.get("content", "")) for g in group]

            # 【优化点】传入 self.embedder，避免重复加载模型
            kws_raw = extract_keywords(titles + bodies, model=self.embedder, topk=5)

            kws = clean_keywords(kws_raw, max_len_each=8, max_k=5)

            dates = [g.get("published_at", dt.datetime.utcnow()).date() for g in group]
            ev_time = choose_event_time(dates)
            time_range_str = ev_time["value"] if ev_time["type"] == "point" else " 至 ".join(ev_time["value"]) if \
            ev_time["value"] else ""

            entities = kws[:3]
            title_llm, kws_llm = make_event_title_llm(
                top_titles=titles[:8], key_terms=kws, entities=entities, time_range=time_range_str,
                fallback_func=make_event_title_fallback, max_len=max_title_len
            )
            final_kws = clean_keywords(kws_llm or kws, max_len_each=8, max_k=5)
            final_title = enforce_title_len(title_llm or make_event_title_fallback(final_kws, titles[:8]), final_kws,
                                            max_len=max_title_len)

            news_ids = [g["_id_str"] for g in group]
            sources = [g.get("source", "") for g in group]
            latest_date = max(dates)
            ncount = len(group)
            rec = recency_score(latest_date, today)
            div = diversity_ratio(sources, ncount)
            score = round(settings.HOTSPOT_ALPHA * math.log1p(
                ncount) + settings.HOTSPOT_BETA * rec + settings.HOTSPOT_GAMMA * div, 4)
            eid = stable_event_id(news_ids)

            events.append({
                "event_id": eid,
                "event_title": final_title,
                "keywords": final_kws,
                "news_count": ncount,
                "score": score,
                "news_ids": news_ids,
                "search_keywords": " ".join(final_kws),
                "event_time": ev_time,
                "created_at": dt.datetime.utcnow(),
                "updated_at": dt.datetime.utcnow(),
            })
        return events

    def upsert_event(self, ev):
        existed = self.col_out.find_one({"event_id": ev["event_id"]}, {"_id": 1})
        if existed:
            self.col_out.update_one({"event_id": ev["event_id"]}, {"$set": ev})
        else:
            self.col_out.insert_one(ev)

    def _collect_news_stats(self, ev):
        """
        【修改版】只收集统计信息，不绘图，不生成文件路径
        """
        news = []
        ids_to_find = set(ev["news_ids"])

        for col_name in self.source_collections:
            if not ids_to_find: break
            q_ids = []
            for x in ids_to_find:
                try:
                    q_ids.append(ObjectId(x))
                except:
                    pass

            col = self.db[col_name]
            found = list(col.find({"_id": {"$in": q_ids}},
                                  {"title": 1, "content": 1, "published_at": 1, "scraped_at": 1, "source": 1,
                                   "url": 1}))
            for d in found:
                ids_to_find.remove(str(d["_id"]))
                pub = d.get("published_at") or parse_date_from_text(d.get("content", "")) or d.get("scraped_at")
                d["published_at"] = parse_date_any(pub)
                if not d.get("source"): d["source"] = derive_source_from_url(d.get("url", ""))
                news.append(d)

        news.sort(key=lambda x: x["published_at"])

        dates = [to_datestr(n["published_at"]) for n in news]
        sources = [n.get("source", "") for n in news]
        titles = [(n.get("title") or "").strip() for n in news]
        norm_titles = [re.sub(r"\s+", " ", t) for t in titles if t]
        dup_ratio = 1.0 - len(set(norm_titles)) / len(norm_titles) if norm_titles else 0.0

        # --- 【修改点】移除所有绘图逻辑 ---
        # 仅保留数据计算用于报告生成
        dates_dt = pd.to_datetime(pd.Series(dates)).dt.date
        daily_cnt = pd.Series(1, index=dates_dt).groupby(level=0).sum().sort_index()

        news_count = len(news)
        source_cnt = len(set([s for s in sources if s]))
        daily_avg = news_count / max(1, len(daily_cnt)) if len(daily_cnt) > 0 else 0.0
        peak_date, peak_count = ("—", 0)
        if len(daily_cnt) > 0:
            peak_date = to_datestr(daily_cnt.idxmax());
            peak_count = int(daily_cnt.max())

        et = ev.get("event_time", {})
        if et.get("type") == "point":
            time_range = et.get("value")
        elif et.get("type") == "range" and et.get("value"):
            time_range = f"{et['value'][0]} 至 {et['value'][1]}"
        else:
            time_range = "未知"

        first_article_date = to_datestr(news[0]["published_at"]) if news else "未知"
        event_created_at = to_datestr(parse_date_any(ev.get("created_at"))) if ev.get("created_at") else "未知"
        try:
            _d1 = parse_date_any(ev.get("created_at")).date()
            _d0 = parse_date_any(news[0]["published_at"]).date() if news else _d1
            detection_latency_days = max(0, (_d1 - _d0).days)
        except Exception:
            detection_latency_days = 0

        titles_clean = []
        for d in news[-200:]:
            t = (d.get("title") or "").strip()
            if 6 <= len(t) <= 80:
                titles_clean.append(f"{to_datestr(d['published_at'])}｜{d.get('source', '')}｜{t}")
        top_titles = "\n".join("- " + x for x in titles_clean[-12:])
        keywords_line = ", ".join(ev.get("keywords", [])[:12])

        horizon = 7

        base_fmt = dict(
            event_title=ev.get("event_title", "热点事件"),
            time_range=time_range,
            keywords=keywords_line or "（暂无）",
            top_titles=top_titles or "（代表性标题不足）",
            news_count=news_count,
            source_cnt=source_cnt,
            daily_avg=daily_avg,
            peak_date=peak_date, peak_count=peak_count,
            first_article_date=first_article_date,
            event_created_at=event_created_at,
            detection_latency_days=detection_latency_days,
            cluster_size=news_count,
            unique_sources=source_cnt,
            dup_title_ratio=dup_ratio,
            min_cluster_size=settings.HOTSPOT_MIN_CLUSTER_SIZE,
            horizon=horizon
        )

        # 统计对象保留，供后续可能使用
        stats_obj = {
            "daily_counts": {to_datestr(k): int(v) for k, v in daily_cnt.items()},
            "news_count": news_count,
            "source_count": source_cnt,
            "dup_title_ratio": dup_ratio,
            "peak_date": peak_date, "peak_count": peak_count,
        }
        return news, base_fmt, stats_obj

    # === [新增] JSON 转 Markdown 辅助方法 ===
    def _json_to_md_classic(self, title, data):
        md = []
        md.append(f"# {title}")
        if data.get("overall_summary"):
            md.append(f"## 摘要\n{data['overall_summary']}")
        if data.get("background"):
            md.append(f"## 背景与前史\n{data['background']}")
        if data.get("progression"):
            md.append("## 事件进展")
            for item in data["progression"]:
                md.append(f"- **{item.get('date', '')}**: {item.get('what', '')}")
        if data.get("impacts"):
            md.append("## 影响评估")
            for k, v in data["impacts"].items():
                if v:
                    lines = "\n".join([f"  - {x}" for x in v])
                    md.append(f"- **{k}**:\n{lines}")
        return "\n\n".join(md)

    def _json_to_md_wiki(self, title, data):
        md = []
        md.append(f"# {title}")
        if data.get("lead"):
            md.append(f"## 导语\n{data['lead']}")
        if data.get("background_and_precedents"):
            md.append("## 背景")
            for k, v in data["background_and_precedents"].items():
                md.append(f"### {k}\n{v}")
        if data.get("itinerary"):
            md.append("## 时间线")
            for stage, events in data["itinerary"].items():
                md.append(f"### {stage}")
                for e in events:
                    md.append(f"- **{e.get('date')}**: {e.get('event')} ({e.get('detail')})")
        return "\n\n".join(md)

    def _gen_one_style(self, ev, base_fmt, stats_obj, references, allowed_dates, style: str):
        prompt = (PROMPT_REPORT_WIKI % base_fmt) if style == "wiki" else (PROMPT_REPORT_CLASSIC % base_fmt)
        try:
            sections = ollama_generate_json(prompt)
        except Exception:
            # 回退逻辑 (简化版)
            sections = {}

            # 过滤开关
        if settings.HOTSPOT_DISABLE_KEY_TARGETS and "key_targets" in sections: sections["key_targets"] = []
        if settings.HOTSPOT_DISABLE_EVOLUTION and "evolution" in sections: sections["evolution"] = {"phases": [],
                                                                                                    "explanation": ""}
        if settings.HOTSPOT_DISABLE_FORECAST and "forecast" in sections: sections["forecast"] = {"horizon_days": 0,
                                                                                                 "volume_forecast": [],
                                                                                                 "scenarios": [],
                                                                                                 "caveats": ""}

        if "progression" in sections:
            sections["progression"] = filter_progression_by_dates(sections.get("progression") or [], allowed_dates)

        try:
            if isinstance(sections, dict):
                if "references" in sections and isinstance(sections["references"], list):
                    sections["references"] = sorted(set(sections["references"] + references))
                else:
                    sections["references"] = references
        except Exception:
            pass

        # === [核心修改] 生成全文 Markdown，不写入本地文件 ===
        if style == 'wiki':
            content_md = self._json_to_md_wiki(ev.get("event_title"), sections)
        else:
            content_md = self._json_to_md_classic(ev.get("event_title"), sections)

        # 返回结构：包含 section JSON 和 content_md 字符串
        return {
            "sections": sections,
            "content_md": content_md,  # <--- 将用于存入数据库
            "stats": stats_obj
        }

    def build_report_for_event(self, ev):
        # 1. 收集数据 (不绘图)
        news, base_fmt, stats_obj = self._collect_news_stats(ev)

        # 2. 构建事实库
        fact_bank_text, references, allowed_dates = build_fact_bank(news, ev.get("keywords", []))
        base_fmt = dict(base_fmt, fact_bank=fact_bank_text)

        # 3. 生成报告 (不写文件)
        classic = self._gen_one_style(ev, base_fmt, stats_obj, references, allowed_dates, style="classic")
        wiki = self._gen_one_style(ev, base_fmt, stats_obj, references, allowed_dates, style="wiki")

        # 4. 更新数据库
        self.col_out.update_one(
            {"event_id": ev["event_id"]},
            {"$set": {
                "report": {"classic": classic, "wiki": wiki},
                "updated_at": dt.datetime.utcnow()
            }}
        )

    def run(self):
        docs = self.load_recent_docs()
        if not docs:
            print("No docs in window; relax filter or check fields.")
            return
        events = self.build_events(docs)
        if not events:
            print("No clusters formed.")
            return
        for ev in events:
            self.upsert_event(ev)
        print(f"[Batch] 生成/更新事件：{len(events)}")

        for ev in events:
            try:
                self.build_report_for_event(ev)
            except Exception as e:
                print(f"[WARN] 报告失败 {ev['event_id']}: {e}")
        print("[Done] 报告已更新至数据库。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noop", action="store_true", help="只测试连接")
    args = parser.parse_args()

    app = HotEventsTwoReportsStrict()
    if args.noop: return
    app.run()


# 优化后的入口函数
def run_on_collection(collection_name=None, source_collections=None):
    """
    collection_name: 废弃，保留兼容
    source_collections: 指定要扫描的源集合列表
    """
    target_sources = source_collections if source_collections else settings.COL_SRC_LIST
    print(f"🔥 开始热点事件识别 (Sources: {target_sources})")
    app = HotEventsTwoReportsStrict(source_collections=target_sources)
    app.run()
    print("🔥 热点事件识别完成")


if __name__ == "__main__":
    main()