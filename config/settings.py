import os
from typing import Optional, List, Dict, Set
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ========== 基础信息 ==========
    PROJECT_NAME: str = "Open Source Data Platform"
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ========== MongoDB 配置 ==========
    MONGO_URI: str = "mongodb://root:123456@39.104.200.88:41004/?authSource=admin"
    MONGO_DB_NAME: str = "NEWS"

    # --- 集合名称 ---
    COLL_WEIBO: str = "source_weibo"
    COLL_CCTV: str = "source_cctv"
    COLL_TOUTIAO: str = "source_toutiao"
    COLL_XINHUA: str = "source_xinHua_net"
    COLL_CNN: str = "source_cnn"

    COLL_INTERIM: str = "interim"
    INTERIM_COLLECTION: str = "toutiao_news_event"
    COL_EVOLUTION: str = "evolution_event"
    EVENT_NODE_COLLECTION: str = "extract_element_event"
    TARGET_NODE_COLLECTION: str = "extract_element_target"
    KNOWLEDGE_COLLECTION_TARGET: str = "knowledge_target"
    EDGES_COLLECTION: str = "event_event_edges"
    KNOWLEDGE_COLLECTION_EVENT: str = "hotspot_events"

    # --- GridFS 存储桶 ---
    BUCKET_WEIBO: str = "fs_weibo"
    BUCKET_CCTV: str = "fs_cctv"
    BUCKET_TOUTIAO: str = "fs_toutiao"
    BUCKET_XINHUA: str = "fs_xinhua"
    BUCKET_CNN: str = "fs_cnn"
    BUCKET_IMAGES: str = "fs_images"

    # ========== 触发机制 (必须加类型注解) ==========
    TRIGGER_DOC_COUNT: int = 50  # 积攒多少条新闻才触发处理
    TRIGGER_MAX_WAIT_SECONDS: int = 1800  # 最长等待时间（秒）
    SEMANTIC_BATCH_SIZE: int = 100  # 语义抽取批处理大小

    # [新增] 热点更新累积阈值
    HOTSPOT_TRIGGER_THRESHOLD: int = 200

    # ========== 热点聚类配置 (必须加类型注解) ==========
    HOTSPOT_DAYS_WINDOW: int = 30
    HOTSPOT_BATCH_LIMIT: int = 5000
    HOTSPOT_MIN_CLUSTER_SIZE: int = 4
    HOTSPOT_OUT_DIR: str = "./reports_out"
    HOTSPOT_ALPHA: float = 0.6
    HOTSPOT_BETA: float = 0.3
    HOTSPOT_GAMMA: float = 0.1
    HOTSPOT_RECENCY_TAU_DAYS: int = 7
    HOTSPOT_DISABLE_KEY_TARGETS: bool = True
    HOTSPOT_DISABLE_EVOLUTION: bool = True
    HOTSPOT_DISABLE_FORECAST: bool = True
    HOTSPOT_MAX_TITLE_LEN: int = 20

    # ========== 爬虫/浏览器环境 ==========
    DRIVER_PATH: str = "/usr/local/bin/chromedriver"
    HTTP_PROXY: Optional[str] = None
    WEIBO_COOKIE_FILE: str = os.path.join(BASE_DIR, "weibo_cookies.json")
    TOUTIAO_COOKIE_FILE: str = os.path.join(BASE_DIR, "toutiao_cookies.json")
    STEALTH_JS_PATH: str = os.path.join(BASE_DIR, "stealth.min.js")
    CRAWL_LIMIT: int = 20

    # ========== 模型相关配置 (Ollama) ==========
    OLLAMA_HOST: str = "http://127.0.0.1:11434"

    # 爬取
    CRAWL_LLM_MODEL: str = "qwen3:32b"
    CRAWL_LLM_TEMPERATURE: float = 0.2

    # 摘要 (Abstract)
    ABSTRACT_OLLAMA_URL: str = "http://127.0.0.1:11434/api/generate"
    ABSTRACT_OLLAMA_MODEL: str = "qwen3:32b"
    MAX_PROMPT_INPUT_LEN: int = 4000
    MAX_ABSTRACT_LEN: int = 200
    HTTP_TIMEOUT: int = 90

    # 抽取 (Text Extraction)
    EXTRACTION_MODEL_NAME: str = "qwen3:32b"
    EXTRACTION_TEMPERATURE: float = 0.1
    EXTRACTION_NUM_CTX: int = 8192
    EXTRACTION_KEEP_ALIVE: int = -1

    # 指代消解 (Disambiguation)
    DISAMBIGUATION_EMBEDDING_MODEL: str = "nomic-embed-text:v1.5"

    # 热点 (Hotspot)
    HOTSPOT_OLLAMA_MODEL: str = "qwen3:32b"
    HOTSPOT_EMB_MODEL: str = "BAAI/bge-m3"

    # ========== 图片处理配置 (Images) ==========
    IMAGES_MODE: str = "convert"
    IMAGES_STANDARD_FORMAT: str = "JPEG"
    IMAGES_DELETE_OLD: bool = False
    IMAGES_DRY_RUN: bool = False
    IMAGES_LIMIT: int = 0
    IMAGES_NAME_FILTER: Optional[str] = None

    # Google Map API
    GOOGLE_MAP_API_KEY: Optional[str] = "68956a5e9d3bb604493964kqo78745c"
    GOOGLE_GEOCODE_URL: Optional[str] = "https://geocode.maps.co/search"

    # ========== NebulaGraph 配置 ==========
    NEBULA_IP: str = "39.104.200.88"
    NEBULA_PORT: int = 41003
    NEBULA_USER: str = "root"
    NEBULA_PASSWORD: str = "123456"
    NEBULA_SPACE_NAME: str = "event_target1"

    # ========== 关系白名单与 Tag 字段配置 ==========

    # --- 1. 关系白名单 ---
    ALLOWED_EVENT_EVENT_RELATIONS: Set[str] = {
        "event-link-link", "event-link-parallel", "event-link-includes",
        "event-link-overlap", "event-link-alternative",
        "event-causal-trigger", "event-causal-result", "event-causal-condition",
        "event-causal-suppress", "event-causal-successor", "event-causal-reason",
        "event-causal-dependency", "event-causal-constraint",
        "event-evolution-leadsTo", "event-evolution-cause", "event-evolution-promote",
        "event-evolution-escalate", "event-evolution-deescalate", "event-evolution-transfer",
        "event-combination-subEvent", "event-combination-stage",
        "event-combination-parallelTask",
    }

    ALLOWED_EVENT_TARGET_RELATIONS: Set[str] = {
        "event-target-facility-occursAtAirport", "event-target-facility-occursAtPort",
        "event-target-facility-affectsAirport", "event-target-facility-affectsPort",
        "event-target-organization-subjectOrganization",
        "event-target-organization-relatedOrganization",
        "event-target-organization-superiorCommandOrganization",
        "event-target-organization-affectedOrganization",
        "event-target-person-subjectPerson", "event-target-person-relatedPerson",
        "event-target-person-leaderOrSpokesperson",
        "event-target-person-affectedPerson",
    }

    REL_PERSON_ORG: Set[str] = {"person-personOrganization-affiliated"}
    REL_ORG_ORG: Set[str] = {
        "organization-organization-superior",
        "organization-organization-subordinate",
        "organization-organization-joint",
    }
    REL_ORG_PER: Set[str] = {"target-target-position-leader"}
    REL_ANY_TARGET_ANY_TARGET: Set[str] = {
        "target-target-operation-operatedBy",
        "target-target-operation-commandedBy",
    }

    # --- 2. 事件标签字段定义 ---
    ECON_EVENT_FIELDS: List[str] = [
        "event_name", "event_first_level", "event_second_level", "time_position_moment",
        "space_position_region", "relationship_characteristics_subject_person",
        "relationship_characteristics_subject_organization",
        "relationship_characteristics_related_party_person",
        "relationship_characteristics_related_party_organization",
        "attribute_event_type", "attribute_policy_tool", "attribute_economic_indicator",
        "attribute_port", "attribute_airport", "emotion_characteristics",
    ]

    MIL_EVENT_FIELDS: List[str] = [
        "event_name", "event_first_level", "event_second_level", "time_position_moment",
        "space_position_place_name", "relationship_characteristics_subject_person",
        "relationship_characteristics_subject_organization",
        "relationship_characteristics_related_party_person",
        "relationship_characteristics_related_party_organization",
        "attribute_event_type", "attribute_background", "attribute_military_means",
        "attribute_casualty", "attribute_strategic_goal", "attribute_command_level",
        "attribute_port", "attribute_airport", "emotion_characteristics",
    ]

    POL_EVENT_FIELDS: List[str] = [
        "event_name", "event_first_level", "event_second_level", "time_position_moment",
        "space_position_place_name", "relationship_characteristics_subject_person",
        "relationship_characteristics_subject_organization",
        "relationship_characteristics_related_party_person",
        "relationship_characteristics_related_party_organization",
        "attribute_event_type", "attribute_background", "attribute_key_action",
        "attribute_port", "attribute_airport", "emotion_characteristics",
    ]

    # --- 3. 目标标签字段定义 ---
    PERSON_FIELDS: List[str] = [
        "name", "alias", "location", "coordinate", "gender", "ethnicity_religion",
        "nationality", "family_members", "affiliated_organization", "rank_position",
        "main_responsibilities", "security_level", "source_url", "event_first_level",
        "event_second_level", "type",
    ]

    ORG_FIELDS: List[str] = [
        "name", "alias", "superior_organization", "subordinate_units", "functions",
        "location", "coordinate", "personnel_size", "leader", "creation_time_history",
        "international_cooperation", "source_url", "event_first_level",
        "event_second_level", "type",
    ]

    PORT_FIELDS: List[str] = [
        "name", "alias", "un_locode", "port_code", "location", "operator", "status",
        "coordinate", "waterway_type", "area_ha", "coastline_length_m", "berths_count",
        "berths_max_depth_m", "container_terminals_count", "bulk_terminals_count",
        "passenger_terminals_count", "storage_capacity_tons", "cranes_count",
        "rail_connection", "road_connection", "max_vessel_size_dwt",
        "annual_throughput_teu", "annual_throughput_tons", "annual_passenger_volume",
        "main_cargo_types", "shipping_lines", "international_status", "opened_date",
        "imo_number", "customs_availability", "pilotage_required",
        "environmental_certifications", "notes", "source_url", "event_first_level",
        "event_second_level", "type",
    ]

    AIRPORT_FIELDS: List[str] = [
        "name", "alias", "iata_code", "icao_code", "location", "coordinate",
        "location_elevation_m", "facilities_runways_count",
        "facilities_runways_lengths_m", "facilities_terminals",
        "facilities_parking_positions", "operation_opened_date", "operation_operator",
        "operation_major_airlines", "traffic_passengers_per_year",
        "traffic_cargo_tons_per_year", "source_url", "event_first_level",
        "event_second_level", "type",
    ]

    WARSHIP_FIELDS: List[str] = [
        "name", "alias", "formation_type", "command_unit", "home_port", "status",
        "mission_roles", "operational_area", "task_force_designation",
        "exercise_participation", "total_ships_count", "ship_types_distribution",
        "flagship_name", "command_ship_class", "support_vessels_count",
        "submarines_count", "aircraft_embarked", "missile_systems",
        "air_defense_capability", "asw_capability", "amphibious_support",
        "mine_warfare_capability", "commander_name", "chain_of_command",
        "communication_systems", "year_established", "recent_upgrades",
        "alliances_or_exercises", "notable_operations", "notes", "source_url",
        "event_first_level", "event_second_level", "type",
    ]

    # --- 4. 字段映射表 (作为属性) ---
    @property
    def EVENT_TAG_FIELDS(self) -> Dict[str, List[str]]:
        return {
            "economic_event": self.ECON_EVENT_FIELDS,
            "military_event": self.MIL_EVENT_FIELDS,
            "political_event": self.POL_EVENT_FIELDS,
        }

    @property
    def TARGET_TAG_FIELDS(self) -> Dict[str, List[str]]:
        return {
            "person_target": self.PERSON_FIELDS,
            "organization_target": self.ORG_FIELDS,
            "port_target": self.PORT_FIELDS,
            "airport_target": self.AIRPORT_FIELDS,
            "warship_target": self.WARSHIP_FIELDS,
        }

    # --- 动态属性 ---
    @property
    def IMAGES_TARGET_BUCKETS(self) -> List[str]:
        return [
            self.BUCKET_WEIBO,
            self.BUCKET_CCTV,
            self.BUCKET_TOUTIAO,
            self.BUCKET_XINHUA,
            self.BUCKET_CNN
        ]

    @property
    def COL_SRC_LIST(self) -> List[str]:
        return [
            self.COLL_WEIBO,
            self.COLL_CCTV,
            self.COLL_TOUTIAO,
            self.COLL_XINHUA,
            self.COLL_CNN
        ]

    class Config:
        env_file = ".env"


settings = Settings()