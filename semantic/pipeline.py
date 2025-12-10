import json
import re
import time
import asyncio
import traceback
from pymongo import MongoClient, UpdateOne
from bson import ObjectId

# === 导入配置 ===
from config.settings import settings
from common.utils import get_logger

# === 导入功能子模块 ===
from semantic.conversion import conversion
from semantic.Anaphora_Resolution import Disambiguation
from semantic.Text_Extraction import event
from semantic.Hot_topic import hotspot
from semantic.Into_mongodb import mogongdb
from semantic.Time_Standard import event_time
from semantic.Abstract import abstract
from semantic.Images import images
# 导入 Nebula 导入模块
from semantic.Into_nebula import nebula_import

logger = get_logger(__name__)


class SemanticPipeline:
    def __init__(self):
        self.client = MongoClient(settings.MONGO_URI)
        self.db = self.client[settings.MONGO_DB_NAME]

        # 初始化集合对象
        self.interim_col = self.db[settings.COLL_INTERIM]  # interim (存放本批次待处理增量数据)
        self.detail_col = self.db[settings.INTERIM_COLLECTION]  # toutiao_news_event
        self.new_detail_col = self.db[settings.COL_EVOLUTION]  # evolution_event
        self.event_node_col = self.db[settings.EVENT_NODE_COLLECTION]  # extract_element_event

        # 源数据集合字典
        self.col_src_dict = {name: self.db[name] for name in settings.COL_SRC_LIST}

        self.running = False  # 控制循环标志
        self.last_run_time = time.time()  # 上次运行时间

        # === 触发阈值配置 (优先从 settings 读取，否则使用默认值) ===
        self.TRIGGER_COUNT = getattr(settings, 'TRIGGER_DOC_COUNT', 50)  # 积攒多少条触发
        self.TRIGGER_WAIT = getattr(settings, 'TRIGGER_MAX_WAIT_SECONDS', 1800)  # 最长等待秒数 (30分钟)

    def get_max_event_ids(self):
        """获取各类型事件的最大编号（从 extract_element_event 集合统计）"""
        max_ids = {}
        all_events = self.event_node_col.aggregate([
            {"$match": {"event_id": {"$exists": True}}},
            {"$group": {"_id": None, "ids": {"$addToSet": "$event_id"}}}
        ])

        prefix_pattern = re.compile(r"^(Th_(ECON|MIL|POL))-(\d+)$")

        for doc in all_events:
            for eid in doc.get("ids", []):
                if not eid: continue
                match = prefix_pattern.match(eid)
                if not match: continue
                prefix, _, num = match.groups()
                try:
                    num = int(num)
                except ValueError:
                    continue
                if prefix not in max_ids or num > max_ids[prefix]:
                    max_ids[prefix] = num
        return max_ids

    def reassign_event_ids(self, docs, max_ids):
        """为新文档分配新的事件ID"""
        prefix_pattern = re.compile(r"^(Th_(ECON|MIL|POL))-(\d+)$")
        id_mapping = {}

        for item in docs:
            events = item.get("structured_data", {}).get("events", [])
            relations = item.get("structured_data", {}).get("event_relations", [])

            for event_item in events:
                eid = event_item.get("event_id")
                if not eid: continue
                match = prefix_pattern.match(eid)
                if not match: continue

                prefix, _, _ = match.groups()
                # 自增 ID
                max_ids[prefix] = max_ids.get(prefix, 0) + 1
                new_id = f"{prefix}-{max_ids[prefix]:05d}"

                id_mapping[eid] = new_id
                event_item["event_id"] = new_id

            # 更新关系中的 ID
            new_relations = []
            for r in relations:
                parts = r.strip().split()
                if len(parts) != 3: continue
                src, tgt, rtype = parts
                src = id_mapping.get(src, src)
                tgt = id_mapping.get(tgt, tgt)
                if src != tgt:
                    new_relations.append(f"{src}  {tgt}  {rtype}")
            item["structured_data"]["event_relations"] = new_relations

        return docs, id_mapping

    def has_valid_time(self, event_item):
        """检查事件是否包含有效时间"""
        if not isinstance(event_item, dict): return False
        time_pos = event_item.get("time_position") or event_item.get("time_position_moment") or event_item.get(
            "time_position_period")
        if not time_pos: return False
        if isinstance(time_pos, str) and time_pos.strip(): return True
        if isinstance(time_pos, dict) and any(v for v in time_pos.values()): return True
        return False

    def clean_structured_data(self, structured_data):
        """递归清理空字段"""

        def _clean(data):
            if isinstance(data, dict):
                return {k: _clean(v) for k, v in data.items()
                        if v not in ["", None, [], {}] and k not in ["trigger_word", "role", "emotion"]}
            elif isinstance(data, list):
                return [_clean(v) for v in data if v not in ["", None, [], {}]]
            else:
                return data

        if not isinstance(structured_data, dict): return {}
        return _clean(structured_data)

    def to_object_id(self, id_val):
        if isinstance(id_val, ObjectId): return id_val
        try:
            return ObjectId(str(id_val))
        except:
            return ObjectId()

    # --- 核心逻辑优化：触发检查 ---
    def check_trigger_condition(self):
        """
        检查是否满足触发条件：
        1. 积压总数 >= TRIGGER_COUNT
        2. 等待时间 >= TRIGGER_WAIT 且有数据
        """
        total_new = 0
        for col_name in settings.COL_SRC_LIST:
            # 统计 status='0' (未处理)
            cnt = self.col_src_dict[col_name].count_documents({"status": "0"})
            total_new += cnt

        elapsed = time.time() - self.last_run_time
        should_run = False
        reason = ""

        if total_new >= self.TRIGGER_COUNT:
            should_run = True
            reason = f"数量阈值触发 (积压 {total_new} 条)"
        elif total_new > 0 and elapsed >= self.TRIGGER_WAIT:
            should_run = True
            reason = f"时间阈值触发 (等待 {int(elapsed)}s, 积压 {total_new} 条)"

        return should_run, reason, total_new

    # --- 核心执行逻辑 ---
    def run_once(self, force=False):
        """执行一次完整的 Pipeline (支持积攒触发 + 增量处理)"""

        # 1. 检查触发条件
        if not force:
            should_run, reason, total_new = self.check_trigger_condition()
            if not should_run:
                # logger.debug(f"[Semantic] 未满足触发条件 (积压: {total_new})")
                return f"Skipped: Not enough data ({total_new})"
            logger.info(f"🚀 [Semantic] 触发执行: {reason}")
        else:
            logger.info("🚀 [Semantic] 强制触发执行...")

        self.last_run_time = time.time()

        # 2. 数据搬运：源集合(status=0) -> Interim，并标记源 status=1
        self.interim_col.delete_many({})  # 清空 interim，准备接收本批次增量

        moved_ids_map = {}  # {col_name: [ids...]}
        total_moved = 0

        for col_name in settings.COL_SRC_LIST:
            # 获取该集合的一批新数据
            # 限制一次处理量，防止单次过多
            batch_limit = getattr(settings, 'SEMANTIC_BATCH_SIZE', 200)
            docs = list(self.col_src_dict[col_name].find({"status": "0"}).limit(batch_limit))

            if docs:
                self.interim_col.insert_many(docs)
                ids = [d["_id"] for d in docs]
                moved_ids_map[col_name] = ids
                total_moved += len(docs)

        if total_moved == 0 and not force:
            return "No new data moved"

        # 3. 标记源数据为 "1" (处理中/已处理)，防止重复搬运
        for col_name, ids in moved_ids_map.items():
            if ids:
                self.col_src_dict[col_name].update_many(
                    {"_id": {"$in": ids}},
                    {"$set": {"status": "1"}}
                )

        logger.info(f"📥 [Semantic] 本次增量处理数据: {total_moved} 条")

        try:
            # 4. 调用各个子模块

            # [Step 1] 热点事件识别
            # 优化：不只看 interim，而是扫描所有源集合的最近 N 天数据，保证热点连贯性
            logger.info("🔥 [Step 1] 热点事件识别 (扫描全量源上下文)...")
            # 传入源集合列表，hotspot 模块会去遍历这些集合
            hotspot.run_on_collection(source_collections=settings.COL_SRC_LIST)

            # [Step 2] 增量信息抽取 (仅针对 interim 中的新数据)
            logger.info("🧠 [Step 2] 增量信息抽取...")
            event.run_on_collection(settings.COLL_INTERIM)

            # [Step 3] 指代消解 (仅针对 interim)
            logger.info("🔗 [Step 3] 指代消解...")
            Disambiguation.main(collection_name=settings.COLL_INTERIM)

            # [Step 4] ID分配与分发 (将抽取结果入库)
            logger.info("🆔 [Step 4] ID 分配与格式转换...")
            processed_docs = list(self.interim_col.find({}))
            max_ids = self.get_max_event_ids()
            processed_docs, _ = self.reassign_event_ids(processed_docs, max_ids)

            detail_docs = []
            evo_docs = []

            for item in processed_docs:
                s_data = self.clean_structured_data(item.get("structured_data", {}))
                # 过滤无时间事件
                valid_events = [e for e in s_data.get("events", []) if self.has_valid_time(e)]
                if not valid_events: continue
                s_data["events"] = valid_events

                # 构造原始结构数据
                detail_docs.append({
                    "_id": self.to_object_id(item.get("_id")),
                    "source": item.get("source"),
                    "event_first_level": item.get("predicted_category"),
                    "event_second_level": item.get("predicted_subcategory"),
                    "structured_data": s_data
                })
                # 构造演化结构数据
                evo_docs.append(conversion.convert_document_simple(item))

            # 写入结果表
            if detail_docs:
                for d in detail_docs:
                    self.detail_col.replace_one({"_id": d["_id"]}, d, upsert=True)
            if evo_docs:
                for d in evo_docs:
                    self.new_detail_col.replace_one({"_id": d["_id"]}, d, upsert=True)

            logger.info(f"💾 [Step 5] 数据入库完成 (Detail: {len(detail_docs)}, Evo: {len(evo_docs)})")

            # [Step 6] 图谱构建 (从库中读取数据构建关联)
            logger.info("🕸️ [Step 6] 图谱构建 (Mogongdb)...")
            mogongdb.main()

            # [Step 7] 时间标准化
            logger.info("⏱️ [Step 7] 时间标准化...")
            # 迁移旧字段兼容
            self.db[settings.EVENT_NODE_COLLECTION].update_many(
                {"time_position_period": {"$exists": True}},
                [{"$set": {"time_position_moment": "$time_position_period", "time_position_period": "$$REMOVE"}}]
            )
            event_time.run_update(limit=0)

            # [Step 8] 生成摘要
            logger.info("📝 [Step 8] 生成摘要...")
            abstract.main()

            # [Step 9] 图片处理 (按需开启)
            logger.info("🖼️ [Step 9] 图片处理...")
            # images.main()

            # [Step 10] Nebula 入库
            logger.info("🌌 [Step 10] Nebula 图谱导入...")
            nebula_import.main()

            # [Step 11] 清理临时集合
            logger.info("🧹 [Step 11] 清空临时集合 interim / toutiao_news_event ...")
            self.interim_col.drop()
            self.detail_col.drop()

            logger.info("✅ [Semantic] 流程结束")
            return "Success"

        except Exception as e:
            logger.error(f"❌ [Semantic] 处理流程异常: {e}\n{traceback.format_exc()}")
            # 出错保留 status="1" 以便人工排查，或者可选择回滚为 "0"
            return f"Error: {str(e)}"

    async def run_loop(self, interval=10):
        """后台自动循环任务"""
        logger.info(f"🔄 [Semantic] 自动监控已启动 (检测频率: {interval}s)")
        self.running = True
        while self.running:
            try:
                # 在线程池中运行同步任务，避免阻塞
                await asyncio.to_thread(self.run_once, force=False)
            except Exception as e:
                logger.error(f"Loop Error: {e}")

            # 等待间隔 (使用较短间隔以便及时响应强制触发或达到阈值)
            for _ in range(interval):
                if not self.running: break
                await asyncio.sleep(1)

        logger.info("🛑 [Semantic] 自动监控已停止")

    def stop(self):
        self.running = False


# 全局单例
pipeline_instance = SemanticPipeline()