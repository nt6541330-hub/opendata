# semantic/Into_nebula/nebula_import.py
import time
from pymongo import MongoClient
from config.settings import settings
from common.utils import get_logger

try:
    from nebula3.gclient.net import ConnectionPool
    from nebula3.Config import Config

    NEBULA_AVAILABLE = True
except ImportError:
    NEBULA_AVAILABLE = False

logger = get_logger(__name__)


# ==========================
# 辅助函数
# ==========================
def escape_ngql_string(s: str) -> str:
    """对字符串进行简单转义，避免 nGQL 语法错误"""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return s


def is_event_id(id_str: str) -> bool:
    """判断是否为事件ID（根据前缀）"""
    if not id_str:
        return False
    return (
            id_str.startswith("Th_POL")
            or id_str.startswith("Th_ECON")
            or id_str.startswith("Th_MIL")
    )


def is_target_id(id_str: str) -> bool:
    """判断是否为目标ID（根据前缀）"""
    if not id_str:
        return False
    return (
            id_str.startswith("PORT-")
            or id_str.startswith("ORG-")
            or id_str.startswith("PER-")
            or id_str.startswith("AIR-")
            or id_str.startswith("WAR-")
    )


def get_event_tag_by_id(event_id: str):
    """根据 event_id 前缀选择事件 Tag"""
    if event_id.startswith("Th_ECON"):
        return "economic_event"
    if event_id.startswith("Th_MIL"):
        return "military_event"
    if event_id.startswith("Th_POL"):
        return "political_event"
    return None


def get_target_tag_by_id(target_id: str):
    """根据 target_id 前缀选择目标 Tag"""
    if target_id.startswith("PORT-"):
        return "port_target"
    if target_id.startswith("AIR-"):
        return "airport_target"
    if target_id.startswith("PER-"):
        return "person_target"
    if target_id.startswith("ORG-"):
        return "organization_target"
    if target_id.startswith("WAR-"):
        return "warship_target"
    return None


def safe_error_msg(resp):
    """安全获取 error_msg，避免编码问题"""
    try:
        return resp.error_msg()
    except Exception as e:
        return f"<error_msg decode failed: {e}>"


# ==========================
# 主逻辑
# ==========================
def main():
    if not NEBULA_AVAILABLE:
        logger.error("❌ 未安装 nebula3-python，无法执行图谱导入。")
        return

    logger.info("🌌 开始导入 NebulaGraph ...")

    # 1. 连接 Nebula
    config = Config()
    config.max_connection_pool_size = 5

    pool = ConnectionPool()
    ok = pool.init([(settings.NEBULA_IP, settings.NEBULA_PORT)], config)
    if not ok:
        logger.error(f"❌ 连接 Nebula 失败，请检查 IP/端口: {settings.NEBULA_IP}:{settings.NEBULA_PORT}")
        return

    try:
        session = pool.get_session(settings.NEBULA_USER, settings.NEBULA_PASSWORD)
    except Exception as e:
        logger.error(f"❌ Nebula 认证失败: {e}")
        pool.close()
        return

    try:
        # 2. 选择已有的 space
        resp = session.execute(f"USE {settings.NEBULA_SPACE_NAME};")
        if not resp.is_succeeded():
            logger.error(f"❌ USE SPACE {settings.NEBULA_SPACE_NAME} 失败: " + safe_error_msg(resp))
            return
        logger.info(f"✅ 已切换到 Space: {settings.NEBULA_SPACE_NAME}")

        # 3. 连接 MongoDB
        mongo_client = MongoClient(settings.MONGO_URI)
        db = mongo_client[settings.MONGO_DB_NAME]

        # 使用 settings 定义的集合名称
        coll_event = db[settings.EVENT_NODE_COLLECTION]
        coll_target = db[settings.TARGET_NODE_COLLECTION]
        coll_edges = db[settings.EDGES_COLLECTION]

        logger.info(
            f"✅ 已连接 MongoDB, 准备处理集合: {settings.EVENT_NODE_COLLECTION}, {settings.TARGET_NODE_COLLECTION}, {settings.EDGES_COLLECTION}")

        # =======================================================
        # 一、导入事件点（属性直接落在事件节点上）
        # =======================================================
        logger.info("🔄 开始导入事件点 ...")
        event_count = 0

        # 获取字段映射
        EVENT_TAG_FIELDS = settings.EVENT_TAG_FIELDS

        for doc in coll_event.find({}):
            event_id = doc.get("event_id")
            if not event_id:
                continue

            tag = get_event_tag_by_id(event_id)
            if tag is None:
                # logger.warning(f"⚠️ 未识别事件类型，跳过: {event_id}")
                continue

            fields = EVENT_TAG_FIELDS.get(tag, [])
            values = []
            for f in fields:
                v = doc.get(f, "")
                v = escape_ngql_string(v)
                values.append(f'"{v}"')

            props_str = ", ".join(fields)
            vals_str = ", ".join(values)

            nql = f'''
            INSERT VERTEX {tag}({props_str})
            VALUES "{event_id}":({vals_str});
            '''
            resp = session.execute(nql)
            if not resp.is_succeeded():
                logger.error(f"❌ 插入事件失败: {event_id} {safe_error_msg(resp)}")
                continue

            event_count += 1
            if event_count % 100 == 0:
                logger.info(f"  已导入事件 {event_count} 条")

        logger.info(f"✅ 事件导入完成，共 {event_count} 条")

        # =======================================================
        # 二、导入目标点（属性直接落在目标节点上）
        # =======================================================
        logger.info("🔄 开始导入目标点 ...")
        target_count = 0

        # 获取字段映射
        TARGET_TAG_FIELDS = settings.TARGET_TAG_FIELDS

        for doc in coll_target.find({}):
            target_id = doc.get("target_id")
            if not target_id:
                continue

            tag = get_target_tag_by_id(target_id)
            if tag is None:
                # logger.warning(f"⚠️ 未识别目标类型，跳过: {target_id}")
                continue

            fields = TARGET_TAG_FIELDS.get(tag, [])
            values = []

            # 1) 构造属性名列表（遇到 time 用反引号包起来）
            prop_name_tokens = []
            for f in fields:
                if f == "time":
                    prop_name_tokens.append("`time`")  # 避免与关键字冲突
                else:
                    prop_name_tokens.append(f)

            # 2) 构造属性值列表
            for f in fields:
                v = doc.get(f, "")
                v = escape_ngql_string(v)
                values.append(f'"{v}"')

            props_str = ", ".join(prop_name_tokens)
            vals_str = ", ".join(values)

            nql = f'''
            INSERT VERTEX {tag}({props_str})
            VALUES "{target_id}":({vals_str});
            '''

            resp = session.execute(nql)
            if not resp.is_succeeded():
                logger.error(f"❌ 插入目标失败: {target_id} {safe_error_msg(resp)}")
                continue

            target_count += 1
            if target_count % 100 == 0:
                logger.info(f"  已导入目标 {target_count} 条")

        logger.info(f"✅ 目标导入完成，共 {target_count} 条")

        # =======================================================
        # 三、导入关系边（事件-事件 / 事件-目标 / 目标-目标）
        # =======================================================
        logger.info("🔄 开始导入关系边 ...")
        edge_count = 0

        for doc in coll_edges.find({}):
            src = doc.get("src")
            dst = doc.get("dst")
            relation = escape_ngql_string(doc.get("relation", ""))

            if not src or not dst:
                continue

            src_is_event = is_event_id(src)
            dst_is_event = is_event_id(dst)
            src_is_target = is_target_id(src)
            dst_is_target = is_target_id(dst)

            nql = None

            # 1) 事件-事件
            if src_is_event and dst_is_event:
                nql = f'''
                INSERT EDGE event_event_rel(relation)
                VALUES "{src}" -> "{dst}":("{relation}");
                '''

            # 2) 事件-目标（统一方向：事件 -> 目标）
            elif (src_is_event and dst_is_target) or (src_is_target and dst_is_event):
                if src_is_event:
                    event_vid, target_vid = src, dst
                else:
                    event_vid, target_vid = dst, src

                nql = f'''
                INSERT EDGE event_target_rel(relation)
                VALUES "{event_vid}" -> "{target_vid}":("{relation}");
                '''

            # 3) 目标-目标
            elif src_is_target and dst_is_target:
                nql = f'''
                INSERT EDGE target_target_rel(relation)
                VALUES "{src}" -> "{dst}":("{relation}");
                '''

            if nql:
                resp = session.execute(nql)
                if not resp.is_succeeded():
                    # 关系插入失败通常可能是端点不存在，记录但不中断
                    # logger.warning(f"  插入边失败: {src}->{dst} {safe_error_msg(resp)}")
                    pass
                else:
                    edge_count += 1
                    if edge_count % 100 == 0:
                        logger.info(f"  已导入关系边 {edge_count} 条")
            else:
                # 两端都不是已知 ID 类型，跳过
                pass

        logger.info(f"✅ 关系边导入完成，共 {edge_count} 条")

    except Exception as e:
        logger.error(f"❌ Nebula 导入过程中发生异常: {e}")
    finally:
        session.release()
        pool.close()
        logger.info("🌌 Nebula 导入任务结束")


if __name__ == "__main__":
    main()