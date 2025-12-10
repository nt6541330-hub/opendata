# open_source_data/tools/text_annotation/api.py
import re
import json
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from common.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ========= 极速版模型配置 =========
MODEL_NAME = "qwen3:0.6b"
fast_chain = None

try:
    logger.info(f"正在初始化极速文本标注模型: {MODEL_NAME} ...")

    fast_llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,  # 0.0 消除随机性，防止瞎编
        num_predict=1024,
        num_ctx=2048,
        num_thread=4,  # 确保开启多线程
        keep_alive="5m",
        request_timeout=15,
    )

    # ========= Prompt (防过拟合优化版) =========
    # 策略变更：
    # 1. 移除具体示例值（如 "A", "Run"），只给结构模板。
    # 2. Key 使用简短英文单词 (sub, type, res) 而非单字母，增强语义理解。
    fast_prompt = PromptTemplate.from_template(
        """
        Extract Information from text to JSON. 

        Keys Definition:
        1. "events": list of {{ "sub"(Subject), "type"(Event Type), "res"(Result), "loc"(Location), "time"(Time) }}
        2. "relations": list of {{ "sub"(Subject), "type"(Relation Type), "obj"(Object) }}
        3. "entities": list of {{ "text"(Entity Name), "type"(Entity Type) }}

        Constraint:
        - Output strictly valid JSON.
        - No markdown formatting.
        - If a field is missing, use empty string "".
        - If a list is empty, use [].

        Input Text: {text}
        JSON Output:
        """
    )

    fast_chain = fast_prompt | fast_llm
    logger.info(f"✅ 文本标注模型 ({MODEL_NAME}) 初始化完成 (均衡模式)")

except Exception as e:
    logger.error(f"❌ 文本标注模型初始化失败: {e}")


# ========= 映射还原逻辑 (适配新 Key) =========
def expand_keys(data):
    # 1. 还原事件
    events = []
    # 兼容模型可能输出 "events" 或 "E" 的情况，优先匹配 prompt 里的全称
    raw_events = data.get("events") or data.get("E") or []
    for item in raw_events:
        # 过滤掉无效数据（比如全空的）
        if not any(item.values()): continue
        events.append({
            "主体": item.get("sub", "") or item.get("s", ""),
            "事件类型": item.get("type", "") or item.get("t", ""),
            "事件结果": item.get("res", "") or item.get("r", ""),
            "地点": item.get("loc", "") or item.get("l", ""),
            "时间": item.get("time", "") or item.get("tm", "")
        })

    # 2. 还原关系
    relations = []
    raw_relations = data.get("relations") or data.get("R") or []
    for item in raw_relations:
        if not any(item.values()): continue
        relations.append({
            "主体": item.get("sub", "") or item.get("s", ""),
            "关系类型": item.get("type", "") or item.get("t", ""),
            "客体": item.get("obj", "") or item.get("o", "")
        })

    # 3. 还原实体
    entities = []
    raw_entities = data.get("entities") or data.get("K") or []
    for item in raw_entities:
        if not any(item.values()): continue
        entities.append({
            "文本": item.get("text", "") or item.get("txt", ""),
            "类型": item.get("type", "") or item.get("ty", "")
        })

    return {"事件": events, "关系": relations, "实体": entities}


# ========= 核心抽取函数 (增加重试与清洗) =========
async def extract_short_text(text: str):
    if not fast_chain:
        return 0.0, {"事件": [], "关系": [], "实体": [], "error": "模型未就绪"}

    start_time = time.time()
    try:
        res = await fast_chain.ainvoke({"text": text})
    except Exception as e:
        logger.error(f"模型调用异常: {e}")
        return 0.0, {"事件": [], "关系": [], "实体": [], "error": str(e)}

    elapsed = time.time() - start_time

    content = res.content if hasattr(res, "content") else str(res)
    content = content.strip()

    # 提取 JSON
    json_str = ""
    # 优先找最外层大括号
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        json_str = match.group()
    else:
        json_str = content

    parsed = {"events": [], "relations": [], "entities": []}

    if json_str:
        try:
            # 常见错误修复
            json_str = json_str.replace("，", ",").replace("：", ":")
            # 修复末尾可能的逗号 (e.g., "events": [...], } )
            json_str = re.sub(r",\s*\}", "}", json_str)
            json_str = re.sub(r",\s*\]", "]", json_str)

            parsed = json.loads(json_str)
        except Exception as e:
            logger.warning(f"JSON解析失败: {e} | 前50字符: {content[:50]}...")

    # 还原中文 Key
    final_data = expand_keys(parsed)

    return elapsed, final_data


# ========= 请求/响应模型 =========
class TextLabelRequest(BaseModel):
    text: str


class TextLabelResponse(BaseModel):
    code: int
    data: dict
    message: str
    elapsed_time: str


# ========= 路由接口 =========
@router.post("/textLabel", response_model=TextLabelResponse, summary="文本多事件抽取(Speed)")
async def text_label(request: TextLabelRequest):
    elapsed, record = await extract_short_text(request.text)

    # 质量检查：如果完全为空，可能是模型没理解
    is_empty = not (record["事件"] or record["关系"] or record["实体"])

    # 简单的后处理：去重
    # (这里可以加简单的 Python 逻辑去重，因为小模型容易重复)

    return {
        "code": 200,
        "data": record,
        "message": "抽取成功" if not is_empty else "结果为空",
        "elapsed_time": f"{elapsed:.2f}s"
    }


# ========= 预热函数 =========
async def warmup():
    if not fast_chain:
        return
    logger.info(f"🔥 [TextLabel-Speed] 预热中...")
    try:
        await fast_chain.ainvoke({"text": "Apple is a company."})
        logger.info("✅ [TextLabel-Speed] 预热完成")
    except Exception as e:
        logger.warning(f"⚠️ 预热失败: {e}")