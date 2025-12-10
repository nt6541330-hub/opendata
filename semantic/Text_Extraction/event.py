# semantic/Text_Extraction/event.py
import json
import sys
import os
import re
import time
from datetime import datetime, timedelta
from bson import ObjectId
from bson import ObjectId as BsonObjectId
from pymongo import MongoClient

# 【修改点 1】引入 settings 和 utils
from config.settings import settings
from common.utils import geocode_address, default_serializer
from semantic.Anaphora_Resolution.entity_synonyms import synonym_map, banned_keywords, location_mapping, banned_patterns

# 【修改点 2】引入 LangChain 组件
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# 简单正则判断
RE_YEAR = re.compile(r'(19|20)\d{2}年')
RE_FULL_DATE = re.compile(r'(?P<month>\d{1,2})月(?P<day>\d{1,2})日')
RE_MONTH_ONLY = re.compile(r'^(?P<month>\d{1,2})月')
RE_YEAR_RANGE = re.compile(r'(?P<y1>(?:19|20)\d{2})年.*(至|-|到).*?(?P<y2>(?:19|20)\d{2})年')
FUZZY = ['昨天', '昨日', '今天', '今日', '上月', '去年', '今年']

# 仅保留事件类分类
PRESET_CATEGORIES = ["台海军事", "台海政治", "台海经济"]
ID_PATTERN = re.compile(r"^(Th_(ECON|MIL|POL))-(\d+)$")


# 【修改点 3】初始化模型函数
def get_extraction_model():
    """根据 settings 配置初始化 ChatOllama 模型"""
    return ChatOllama(
        base_url=settings.OLLAMA_HOST,
        model=settings.EXTRACTION_MODEL_NAME,
        temperature=settings.EXTRACTION_TEMPERATURE,
        num_ctx=settings.EXTRACTION_NUM_CTX,
        keep_alive=settings.EXTRACTION_KEEP_ALIVE
    )


def create_classification_chain(llm):
    """创建分类链"""
    schemas = [
        ResponseSchema(
            name="category",
            description=f"新闻分类，必须严格从以下3个选项中选择一个：{', '.join(PRESET_CATEGORIES)}"
        )
    ]
    parser = StructuredOutputParser.from_response_schemas(schemas)

    prompt = PromptTemplate.from_template(
        """你是一个专业的新闻分类机器人。请仔细阅读新闻内容，并从指定的3个事件分类中选择最恰当的一项。

分类说明：
- 台海军事：涉及军事行动、战斗、冲突、演习、军事设施活动、武器装备使用、军事演习、军事人员行为等具体军事相关活动
- 台海政治：涉及政治决策、外交活动、政府行为、政策变化、政治人物活动、政治组织、党派活动、两岸关系谈判、两岸和平统一行为、两岸和平统一言论等政治相关动态
- 台海经济：涉及经济活动、市场变化、贸易、投资、金融、商业行为、经济政策实施等经济相关活动

可选分类：{categories}

新闻内容：{content}

请严格按照以下格式返回结果：
{format_instructions}

/no_think
        """
    )

    return (
            prompt.partial(
                categories=", ".join(PRESET_CATEGORIES),
                format_instructions=parser.get_format_instructions()
            )
            | llm
            | parser
    )


def create_subclassification_chain(llm, category: str):
    """根据一级分类创建二级三级子类分类链"""

    SUBCATEGORIES = {
        "台海军事": [
            "演训与部署",
            "军事摩擦与冲突",
            "武器试验与部署",
            "对台军售与军援",
            "战略威慑与声明",
            "军事设施与基地动态"
        ],
        "台海政治": [
            "高层互动与外交动作",
            "政策发布与法案推进",
            "台湾岛内政治活动",
            "两岸官方互动变动",
            "舆情与社会动员",
            "领导人象征行为"
        ],
        "台海经济": [
            "经贸制裁与出口限制",
            "产业链风险与转移",
            "金融市场反应",
            "能源与资源安全",
            "外资与对外投资",
            "网络与基础设施攻击"
        ]
    }

    available_options = SUBCATEGORIES.get(category, [])

    schemas = [
        ResponseSchema(
            name="subcategory",
            description=f"请从以下选项中选择最符合的子类，格式为“二级”：{', '.join(available_options)}"
        )
    ]
    parser = StructuredOutputParser.from_response_schemas(schemas)

    prompt = PromptTemplate.from_template(
        """你是一个台海事件子类识别助手。根据给定的新闻内容，在下列事件子类选项中选择最符合的一个。

子类说明（格式为“二级”）：
{options}

新闻内容：{content}

请严格按照以下格式返回：
{format_instructions}

/no_think
        """
    )

    return (
            prompt.partial(
                options=", ".join(available_options),
                format_instructions=parser.get_format_instructions()
            )
            | llm
            | parser
    )


def create_political_extraction_chain_multi(llm):
    """支持多事件的台海政治结构化抽取"""
    event_schema = ResponseSchema(
        name="events",
        description="结构化事件数组，每条新闻最多提取5个政治事件，每个为独立结构体"
    )
    parser = StructuredOutputParser.from_response_schemas([event_schema])

    prompt = PromptTemplate.from_template(
        """你是专业的台海政治事件信息提取专家。请从以下新闻中提取"所有"可识别的独立的台海政治事件（最多5条），每个事件请按如下字段结构单独列出：
注意：
- 即使文本中仅存在一个事件，也请以数组形式输出；
- 若事件信息不完整，请尽可能补全；
- 若事件较模糊，也请合理推断其核心结构；
- 不要省略任何可疑似事件。

【事件名称(event_name)】
- event_name：简洁准确概括事件主干动作+关键参与方，建议10-30字（如："中美高级别贸易磋商会议"、"台立法院审议两岸法案"）。

【触发词（trigger_word）】
- trigger_word：该事件的核心关键词或关键短语（如："李大维"、"出任"、"外交部部长"），以列表形式返回；
- 必须覆盖事件中能代表核心内容的1~5个关键词，不可为空。

【时间定位（time_position）】
- moment：具体时刻（如 "2025年6月18日"、"2024年5月"、"2025年"）。
- period：时间范围（如 "2025年1月 至 2025年12月"、"2022年 至今"、"2020年5月 至 2022年8月"）。
⚠️ 注意：1.输出时只保留 moment 或 period 其中一个，按文本实际情况选择；
        2.如果文本只出现“月日”或“月”，则把2025年作为默认年份补齐时间；
        3.如果文本只出现“日”，向前查找相连的“月”或“月日”或“年月”叙述，将其与默认年份组合。
        4.时间中必须有年份，不能单独写“去年”“上月”“昨日”等模糊词；

【空间定位（space_position）】
- coordinate：经纬度位置（如有）；
- place_name：具体地点名称（如"北京"、"华盛顿"）；
- region：宏观区域（如"亚太地区"、"欧洲联盟"）；
⚠️ 只输出其一，按文本实际出现选择。

【关系特征（relationship_characteristics）】
- subject_person：主要人物，包括：
  - name：主体人物名称，指事件中直接参与、发起或主导行为的核心人物名称（必须是自然人姓名，如："王毅"、"南希·佩洛西"）；
  - role：主体人物在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：主体人物在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
- subject_organization：主体组织，包括：
  - name：主体组织名称，指事件中直接参与、发起或主导行为的核心组织名称；
  - role：主体组织在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：主体组织在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
- related_party_person：关系人物，包括：
  - name：关系人物名称，指与事件相关但不作为主体的人物名称（必须是自然人姓名，如："赖清德"等人名）；
  - role：关系人物在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：关系人物在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
- related_party_organization：关系组织，包括：
  - name：关系组织名称，指与事件相关但不作为主体的组织名称（如合作对象、对手、受影响方等）；
  - role：关系组织在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：关系组织在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
⚠️注意事项：
    主体与关系对象必须是事件中不同的实体，不得重复；
    人物名称字段仅限自然人姓名，组织、机构、国家地区等名称不得放入人物字段；
    组织字段可包含政府部门、公司企业、机构团体等名称；
    若文本中未明确提及可留空（""），但角色和情感可根据上下文合理推测。

【属性特征（attribute_characteristics）】
- event_type：台海政治类型（如"选举"、"外交访问"）；
- background：政治事件相关背景；
- key_action：关键行为或动作；
- port：该政治事件中出现的相关的的港口（如："厦门军港"）;
- airport：该政治事件中出现的相关的的机场（如："厦门高崎机场"）.
按实际出现填写。

【情绪特征（emotion_characteristics）】
- positive：若事件整体走向具有积极作用，例如促进合作、缓解紧张、推动和平稳定。
- negative：若事件整体走向具有消极作用，例如导致局势紧张、冲突升级、挑衅威胁或造成损失。
- neutral：若事件整体走向中性，不表现出明显的正面或负面趋势，属于常规或客观报道。
⚠️ 按实际出现填写,严格只输出其一，按文本实际出现选择最重要的一个输出。

【新闻内容】
{content}

请严格按照以下格式返回结果：
{format_instructions}

⚠️ 请严格只输出一个 JSON 对象，禁止输出<think>、解释、markdown、自然语言说明等。

请以如下 JSON 格式返回（最多5个）：
{{
  "events": [
    {{
      "event_name": "王毅会见美国国务卿讨论台海局势",
      "trigger_word": ["王毅", "美国国务卿", "会见", "台海局势", "反对干涉"],
      "time_position": {{"moment": "2025年7月8日"}},
      "space_position": {{"place_name": "北京"}},
      "relationship_characteristics": {{
        "subject_person": {{
          "name": "王毅",
          "role": "发言方",
          "emotion": "反对"
        }},
        "subject_organization": {{
          "name": "中国外交部",
          "role": "执行方",
          "emotion": "反对"
        }},
        "related_party_person": {{
          "name": "安东尼·布林肯",
          "role": "会谈方",
          "emotion": "合作"
        }},
        "related_party_organization": {{
          "name": "美国国务院",
          "role": "合作方",
          "emotion": "合作"
        }}
      }},
      "attribute_characteristics": {{
        "event_type": "外交访问",
        "background": "中美就台海议题展开高层对话",
        "key_action": "举行会晤并发表讲话",
        "port": "",
        "airport": ""
      }},
      "emotion_characteristics": {{"negative": "坚决反对外部干涉"}}
    }},
    ...
  ]
}}
/no_think
"""
    )
    return (
            prompt.partial(format_instructions=parser.get_format_instructions())
            | llm
            | parser
    )


def create_military_extraction_chain_multi(llm):
    """支持多事件的台海军事结构化抽取"""
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="events", description="结构化军事事件列表，每条为独立事件")
    ])

    prompt = PromptTemplate.from_template(
        """你是台海军事事件提取专家，请从以下新闻中提取"所有"可识别的独立的台海军事事件（最多5个）。每个事件按如下结构组织：
注意：
- 即使文本中仅存在一个事件，也请以数组形式输出；
- 若事件信息不完整，请尽可能补全；
- 若事件较模糊，也请合理推断其核心结构；
- 不要省略任何可疑似事件。

【事件名称(event_name)】
- event_name：简洁准确概括事件主干动作+关键参与方，建议10-30字。

【触发词（trigger_word）】
- trigger_word：该事件的核心关键词或关键短语（如："李大维"、"出任"、"外交部部长"），以列表形式返回；
- 必须覆盖事件中能代表核心内容的1~5个关键词，不可为空。

【时间定位（time_position）】
- moment：具体时刻（如 "2025年6月18日"、"2024年5月"、"2025年"）。
- period：时间范围（如 "2025年1月 至 2025年12月"、"2022年 至今"、"2020年5月 至 2022年8月"）。
⚠️ 注意：1.输出时只保留 moment 或 period 其中一个，按文本实际情况选择；
        2.如果文本只出现“月日”或“月”，则把2025年作为默认年份补齐时间；
        3.如果文本只出现“日”，向前查找相连的“月”或“月日”或“年月”叙述，将其与默认年份组合。
        4.时间中必须有年份，不能单独写“去年”“上月”“昨日”等模糊词；

【空间定位（space_position）】
- coordinate：经纬度位置；
- place_name：具体地名（如"巴以边境"、"南海"）；
- region：更大范围地理区域（如"中东"、"东欧战区"）；
⚠️ 只输出其一，按实际出现选择。

【关系特征（relationship_characteristics）】
- subject_person：主要人物，包括：
  - name：主体人物名称，指事件中直接参与、发起或主导行为的核心人物名称（必须是自然人姓名，如："王毅"、"南希·佩洛西"）；
  - role：主体人物在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：主体人物在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
- subject_organization：主体组织，包括：
  - name：主体组织名称，指事件中直接参与、发起或主导行为的核心组织名称；
  - role：主体组织在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：主体组织在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
- related_party_person：关系人物，包括：
  - name：关系人物名称，指与事件相关但不作为主体的人物名称（必须是自然人姓名，如："赖清德"等人名）；
  - role：关系人物在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：关系人物在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
- related_party_organization：关系组织，包括：
  - name：关系组织名称，指与事件相关但不作为主体的组织名称（如合作对象、对手、受影响方等）；
  - role：关系组织在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：关系组织在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
⚠️注意事项：
    主体与关系对象必须是事件中不同的实体，不得重复；
    人物名称字段仅限自然人姓名，组织、机构、国家地区等名称不得放入人物字段；
    组织字段可包含政府部门、公司企业、机构团体等名称；
    若文本中未明确提及可留空（""），但角色和情感可根据上下文合理推测。

【属性特征（attribute_characteristics）】
- event_type：台海军事类型（如"军事打击"、"边境冲突"）；
- background：背景事件或起因；
- military_means：所使用的军事手段或装备（如"空袭"、"导弹"、"电子战"）；
- casualty：人员伤亡或损失描述；
- strategic_goal：战略目标（如"打击反政府武装"、"控制海域"）；
- command_level：涉及的指挥层级或人物（如"旅级指挥官"、"国防部长"）。
- port：该军事事件中出现的相关的的港口（如："厦门军港"）
- airport：该军事事件中出现的相关的的机场（如："厦门高崎机场"）

【情绪特征（emotion_characteristics）】
- positive：正面影响或鼓舞表达；
- negative：负面情绪、谴责、伤亡等；
- neutral：中性客观描述。
⚠️ 按实际出现填写,严格只输出其一，按文本实际出现选择最重要的一个输出。

【新闻内容】
{content}

请严格按照以下格式返回结果：
{format_instructions}

⚠️ 请严格只输出一个 JSON 对象，禁止输出<think>、解释、markdown、自然语言说明等。

示例格式（请严格参照）：
{{
  "events": [
    {{
      "event_name": "东部战区对英国军舰过航台湾海峡进行警戒处置",
      "trigger_word": ["警戒处置", "英国军舰", "台湾海峡"],
      "time_position": {{"moment": "2025年6月18日"}},
      "space_position": {{"place_name": "台湾海峡"}},
      "relationship_characteristics": {{
        "subject_person": {{
            "name": "刘润科",
            "role": "发言方",
            "emotion": "谴责"
        }},
        "subject_organization": {{
            "name": "东部战区",
            "role": "执行方",
            "emotion": "强硬"
        }},
        "related_party_person": {{
            "name": "",
            "role": "",
            "emotion": ""
        }},
        "related_party_organization": {{
            "name": "英国海军",
            "role": "过航方",
            "emotion": "挑衅"
        }}
      }},
      "attribute_characteristics": {{
        "event_type": "军事警戒",
        "background": "英国斯佩号近岸巡逻舰过航台湾海峡并公开炒作",
        "military_means": "兵力跟监警戒",
        "casualty": "",
        "strategic_goal": "反制威胁挑衅，维护台海和平稳定",
        "command_level": "东部战区海军新闻发言人刘润科大校",
        "port": "",
        "airport": ""
      }},
      "emotion_characteristics": {{"negative": "英方行为被指蓄意滋扰搅局，破坏台海和平稳定"}}
    }},
    ...
  ]
}}
/no_think
"""
    )
    return (
            prompt.partial(format_instructions=parser.get_format_instructions())
            | llm
            | parser
    )


def create_economic_extraction_chain_multi(llm):
    """支持多事件的台海经济结构化抽取"""
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="events", description="结构化经济事件列表，每条为独立事件对象")
    ])

    prompt = PromptTemplate.from_template(
        """你是台海经济新闻事件分析专家，请从以下新闻中提取"所有"可识别的独立的经济类事件（最多5条），每条事件请提取以下字段：

注意：
- 即使文本中仅存在一个事件，也请以数组形式输出；
- 若事件信息不完整，请尽可能补全；
- 若事件较模糊，也请合理推断其核心结构；
- 不要省略任何可疑似事件。

【事件名称(event_name)】
- event_name：准确概括经济事件核心，保持10-30字（如："美联储宣布加息决定"、"中欧投资协定谈判"）。

【触发词（trigger_word）】
- trigger_word：该事件的核心关键词或关键短语（如："李大维"、"出任"、"外交部部长"），以列表形式返回；
- 必须覆盖事件中能代表核心内容的1~5个关键词，不可为空。

【时间定位（time_position）】
- moment：具体时刻（如 "2025年6月18日"、"2024年5月"、"2025年"）。
- period：时间范围（如 "2025年1月 至 2025年12月"、"2022年 至今"、"2020年5月 至 2022年8月"）。
⚠️ 注意：1.输出时只保留 moment 或 period 其中一个，按文本实际情况选择；
        2.如果文本只出现“月日”或“月”，则把2025年作为默认年份补齐时间；
        3.如果文本只出现“日”，向前查找相连的“月”或“月日”或“年月”叙述，将其与默认年份组合。
        4.时间中必须有年份，不能单独写“去年”“上月”“昨日”等模糊词；

【空间定位（space_position）】
- coordinate：经纬度位置（如有）；
- place_name：具体地点名称（如"上海"、"纽约"）；
- region：宏观区域（如"亚太地区"、"欧洲市场"）；
⚠️ 只输出其一，按文本实际出现选择。

【关系特征（relationship_characteristics）】
- subject_person：主要人物，包括：
  - name：主体人物名称，指事件中直接参与、发起或主导行为的核心人物名称（必须是自然人姓名，如："王毅"、"南希·佩洛西"）；
  - role：主体人物在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：主体人物在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
- subject_organization：主体组织，包括：
  - name：主体组织名称，指事件中直接参与、发起或主导行为的核心组织名称；
  - role：主体组织在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：主体组织在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
- related_party_person：关系人物，包括：
  - name：关系人物名称，指与事件相关但不作为主体的人物名称（必须是自然人姓名，如："赖清德"等人名）；
  - role：关系人物在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：关系人物在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
- related_party_organization：关系组织，包括：
  - name：关系组织名称，指与事件相关但不作为主体的组织名称（如合作对象、对手、受影响方等）；
  - role：关系组织在事件中的角色（如“发言方”“合作对象”“执行方”“受害方”等）；
  - emotion：关系组织在事件中的情感标记（如“合作”“对等”“谴责”“支持”“反对”等，文中未提及可进行推测）；
⚠️注意事项：
    主体与关系对象必须是事件中不同的实体，不得重复；
    人物名称字段仅限自然人姓名，组织、机构、国家地区等名称不得放入人物字段；
    组织字段可包含政府部门、公司企业、机构团体等名称；
    若文本中未明确提及可留空（""），但角色和情感可根据上下文合理推测。

【属性特征（attribute_characteristics）】
- event_type：台海经济类型（如"贸易协定签署"、"货币政策调整"）；
- policy_tool：政策工具（如"利率变动"、"关税调整"）；
- economic_indicator：经济指标（如"GDP增长率"、"失业率"）；
- port：该经济事件中出现的相关的的港口：
- airport：该经济事件中出现的相关的的机场（如："厦门高崎机场"）.
按实际出现填写。

【情绪特征（emotion_characteristics）】
- positive：正面影响或积极表达；
- negative：负面影响或批评表达；
- neutral：中立客观描述；
⚠️ 按实际出现填写,严格只输出其一，按文本实际出现选择最重要的一个输出。

【新闻内容】
{content}

请严格按照以下格式返回结果：
{format_instructions}

⚠️ 请严格只输出一个 JSON 对象，禁止输出<think>、解释、markdown、自然语言说明等。

示例格式（请严格参照）：

{{
"events": [
{{
  "event_name": "中美贸易协定达成",
  "trigger_word": ["关键词1", "关键词2", "关键词3"],
  "time_position": {{"moment": "2025年7月8日"}},
  "space_position": {{"region": "亚太地区"}},
  "relationship_characteristics": {{
    "subject_person": {{
      "name": "王美花",
      "role": "签署方",
      "emotion": "支持"
    }},
    "subject_organization": {{
      "name": "台湾经济部",
      "role": "执行方",
      "emotion": "合作"
    }},
    "related_party_person": {{
      "name": "戴琪",
      "role": "签署方",
      "emotion": "支持"
    }},
    "related_party_organization": {{
      "name": "美国贸易代表办公室",
      "role": "参与方",
      "emotion": "合作"
    }}
  }},
  "attribute_characteristics": {{
    "event_type": "贸易协定签署",
    "policy_tool": "关税调整",
    "economic_indicator": "进出口总额增长",
    "port": "",
    "airport": ""
  }},
  "emotion_characteristics": {{"positive": "双方均表示合作意愿强烈"}},
}},
...
]
}}
/no_think
"""
    )

    return (
            prompt.partial(format_instructions=parser.get_format_instructions())
            | llm
            | parser
    )


def create_event_relation_chain(llm):
    """创建事件关系识别链（最终整合版，满足全部 1~4 条要求）"""
    RELATIONS = """
【一、事件 ↔ 事件（event-link 系列，非因果）】
1. event-link                        —— 普通关联（存在一般性关联但非因果/包含）
2. event-link-link                   —— 间接关联（通过第三方、间接因素产生联系，但无直接关系）
3. event-link-parallel               —— 并行发生，无相互影响
4. event-link-includes               —— 包含关系（A 包含 B）
5. event-link-overlap                —— 重叠 / 重复（同一事件或内容高度重合）
6. event-link-alternative            —— 替代关系（互为候选方案、替代方案或备选行动）

【二、事件 ↔ 事件（event-causal 系列，因果）】
7.  event-causal                     —— 存在因果，但不区分方向
8.  event-causal-trigger             —— A 触发 B（直接诱发）
9.  event-causal-result              —— B 是 A 的结果（由 A 导致）
10. event-causal-condition           —— A 是 B 的条件/前提
11. event-causal-suppress            —— A 抑制/阻止 使事件 B 无法发生或延迟发生

【三、事件 ↔ 事件（event-evolution 系列，演化）】
12. event-evolution                  —— 属于同一演化链条，显示发展趋势
13. event-evolution-leadsTo          —— A 导致 B 进一步发展（偏中性）
14. event-evolution-cause            —— A 引起 B（偏触发）
15. event-evolution-promote          —— A 促进/加速 B
16. event-evolution-escalate         —— B 是 A 的升级
17. event-evolution-deescalate       —— B 是 A 的缓和
18. event-evolution-transfer         —— 事件方向/地点/对象转移

【四、事件 ↔ 事件（event-combination 系列，组合）】
19. event-combination                —— 多事件构成事件群
20. event-combination-subEvent       —— B 是 A 的子事件
21. event-combination-stage          —— 同一事件不同阶段
22. event-combination-parallelTask   —— 平行任务，属于同一行动的子任务

【五、事件 ↔ 实体（event-target 系列）】

——设施（来源：事件的 airport / port 字段）——
23. event-target-facility                    —— 事件涉及某设施但具体类型不明确
24. event-target-facility-occursAtAirport    —— 事件发生在机场
25. event-target-facility-occursAtPort       —— 事件发生在港口
26. event-target-facility-affectsAirport     —— 事件影响机场
27. event-target-facility-affectsPort        —— 事件影响港口

——组织（来源：subject_organization / related_party_organization）——
28. event-target-organization                        —— 事件针对某组织
29. event-target-organization-subjectOrganization    —— 该组织是事件的主体方
30. event-target-organization-relatedOrganization    —— 该组织是事件的相关方
31. event-target-organization-superiorCommandOrganization —— 该组织是负责指挥、监管该事件的上级单位
32. event-target-organization-affectedOrganization   —— 事件影响该组织，使其受到损害或影响

——人物（来源：subject_person / related_party_person）——
33. event-target-person                      —— 事件指向某自然人
34. event-target-person-subjectPerson        —— 该人物是事件主体执行者
35. event-target-person-relatedPerson        —— 该人物与事件相关
36. event-target-person-leaderOrSpokesperson —— 领导人、发言人、官方代表
37. event-target-person-affectedPerson       —— 事件影响到该人物
"""

    prompt = PromptTemplate.from_template(f"""
你是一个专业的事件关系识别模型。现给你一组结构化事件描述，请严格识别事件之间或事件与实体之间是否存在以下关系列表中的一种关系：

{RELATIONS}

==============================
【事件描述无歧义原则】
事件描述是从结构化字段拼接而来，不包含隐含推断。你必须仅依据事件提供的内容判断关系，不得添加外推、常识补全、主观推测、现实背景等信息。
==============================
【实体来源规则】
- 人物实体必须来自 subject_person.name / related_party_person.name
- 组织实体必须来自 subject_organization.name / related_party_organization.name
- 设施实体必须来自 airport / port
不得输出任何事件描述之外的实体，不得生成臆想的新实体。
==============================
【实体名称合法性约束】
你必须过滤掉以下情况的实体，不允许作为 event-target 的对象：
- 人物：不得是泛化称呼（如“网友”“政要”“一名男子”等），必须为自然人姓名；
- 组织：必须是具体组织全称，不得是泛称（如“政府”“部门”“媒体”“机构”）；
- 不得输出头衔形式，如“美国总统”“××部长”“国务院”，只能保留最终人名或组织名；
- 不得输出泛化、模糊、不完整、类别错误、统称式实体。
若不满足这些条件，不得输出该实体的 event-target 关系。
==============================
【输出格式（必须遵守）】
1. 事件↔事件关系格式（前 22 类关系）：
   事件ID<两个空格>事件ID<两个空格>关系类型
2. 事件↔实体关系格式（设施/组织/人物）：
   事件ID<两个空格>实体名称<两个空格>关系类型
3. 多个关系请分行输出，每行一个三元组。
4. 禁止输出任何解释、评论、推理过程、说明文字，不得输出标题、附加文本、标点符号。
5. 若无法确定关系，则不要输出该关系。
6. 若完全无关系，则输出空。
==============================

【示例输出】 
"Th_POL-00001  Th_POL-00002  event-link-link"
"Th_POL-00003  佩罗西  event-target-person"

以下是抽取出的事件：
{{events_text}}

请严格按格式输出三元组列表，不要解释或说明。
/no_think
""")
    return prompt | llm | StrOutputParser()


def filter_entity_name(name: str) -> str:
    """过滤或修正实体名称，返回过滤后结果，空串表示剔除该实体"""
    if not name:
        return ""
    name = name.strip()
    # 如果在映射表中，同义替代处理
    if name in synonym_map:
        return synonym_map[name]

    # 黑名单关键词列表（忽略大小写）
    for kw in banned_keywords:
        if kw in name:
            return ""
    # 黑名单模式（正则）
    for pattern in banned_patterns:
        if re.fullmatch(pattern, name):
            return ""

    # 地理位置映射
    if name in location_mapping:
        return location_mapping[name]

    # 大使馆 / 领事馆 / 代表处 属于组织
    embassy_keywords = ["大使馆", "领事馆", "代表处", "驻"]
    if any(k in name for k in embassy_keywords):
        return name  # 组织，不删除
    # 如果是人名 + 大使（明确到人）
    if "大使" in name and len(name) > 3:
        # 这里 len>3 是为了排除单独"大使"两个字
        return name
    # 单独的“大使”或含“使”但不具体 → 删除
    if name in ["大使", "特使"] or ("使" in name and len(name) <= 3):
        return ""

    # 其他过滤规则补充
    return name


def resolve_entity_coreferences_with_qwen3(name: str, model):
    """使用 Qwen3 模型进行人物/组织名称的规范化"""
    prompt = f"""
    /no_think
    你是一个名称标准化工具。只返回最终名称，不输出任何思维链内容或解释。
    名称：{name}

    规范化要求：
    - 禁止输出 <think> 内容或任何推理过程。
    - 禁止输出多段文字。
    - 如果是人物，返回中文名字；如果是组织，返回组织中文名称；如果是地名或泛化组织，返回空字符串。
    - 不要附加任何说明或背景信息，仅返回最终名称。

     """.strip()
    try:
        res = model.invoke(prompt)

        # 取文本内容
        text = res.content if hasattr(res, "content") else str(res)

        # ① 删除 <think>…</think>
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)

        # ② 删除代码块、markdown、json等
        text = re.sub(r"```.*?```", "", text, flags=re.S)

        # ③ 只取最后一个中文段
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return ""

        text = lines[-1]  # 最后一段通常是最终答案

        # ④ 去掉英文、数字、符号
        text = re.sub(r"[A-Za-z0-9<>/{}[\]()\"']", "", text).strip()

        # ⑤ 应用同义词映射
        if text in synonym_map:
            text = synonym_map[text]

        # ⑥ 最后过滤（黑名单等）
        text = filter_entity_name(text)

        return text
    except Exception:
        return ""


def resolve_location_with_qwen3(location: str, model):
    """使用 Qwen3 模型进行地理位置信息的清洗"""
    prompt = f"""
    /no_think
    你是一个地理位置名称标准化专家。请规范化以下地理位置信息，只返回最终地理位置名称，不输出任何思维链内容或解释。
    地理位置：{location}

    规范化要求：
    - 如果名称包含“台湾”，请返回“台湾省”。
    - 对于其他地理位置名称，请返回标准化的中文地名。
    - 返回的名称应为中文，不含英文描述或其他无关信息。
    - 禁止输出思维链、解释、<think> 标签或多段内容。
    - 如果识别失败返回空字符串。

    """.strip()

    try:
        res = model.invoke(prompt)
        text = res.content if hasattr(res, "content") else str(res)

        # 清除 think
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)

        # 清理多余
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return ""

        text = lines[-1]

        # 使用 location_mapping
        if text in location_mapping:
            text = location_mapping[text]

        return text.strip()
    except:
        return ""


def clean_event_entities(relationship_characteristics: dict, model) -> dict:
    """
    只处理 relationship_characteristics 中各实体的 name 字段。
    - 使用 Qwen3 消解人物/组织名称
    - strip_title_prefix、synonym_map、filter_entity_name
    - 名称为空 → 删除整个实体
    - 返回清洗后的 relationship_characteristics（可能为空 dict）
    """

    def is_empty(v):
        return v is None or (isinstance(v, str) and v.strip() == "")

    def clean_name_output(name: str) -> str:
        """模型输出后的最终清洗：去掉 think、非法字符、映射等"""

        if not name:
            return ""
        # 去除 <think> 污染
        name = re.sub(r"<think>.*?</think>", "", name, flags=re.S)
        # 去除开头非中文（如 5中国外交部、birth日本内阁）
        name = re.sub(r"^[^一-龥]*", "", name).strip()
        # strip 职务
        name = strip_title_prefix(name)
        # 同义词映射
        if name in synonym_map:
            name = synonym_map[name]
        # 黑名单过滤
        name = filter_entity_name(name)
        return name

    # 如果为 None 或空，直接返回空 dict
    if not isinstance(relationship_characteristics, dict):
        return {}

    cleaned = {}
    entity_fields = [
        "subject_person",
        "subject_organization",
        "related_party_person",
        "related_party_organization",
    ]

    for field in entity_fields:
        entity = relationship_characteristics.get(field)
        if not entity or not isinstance(entity, dict):
            continue
        raw_name = entity.get("name", "").strip()
        # 清理 think 污染
        raw_name = re.sub(r"<think>.*?</think>", "", raw_name, flags=re.S).strip()
        # 名称为空 → 删除整个实体
        if is_empty(raw_name):
            continue
        # Qwen 消解
        resolved = resolve_entity_coreferences_with_qwen3(raw_name, model)
        # 最终清理
        final_name = clean_name_output(resolved)
        # 名称为空 → 删除实体
        if is_empty(final_name):
            continue
        # 保留实体 + 更新 name
        entity["name"] = final_name
        cleaned[field] = entity

    return cleaned


def clean_event_locations(space_position: dict, model) -> dict:
    """
    清洗 space_position 中的 place_name（只处理 place_name，不处理 region）。
    规则：
    - 若 space_position 不是 dict 或 place_name 为空，返回 {}（视为无效）
    - 若存在 region 字段且非空，不做任何处理，返回原 space_position
    - 对 place_name 做严格的“是否为地名”判断：若模型返回空或判定为非地名，则保留原 place_name（不修改）
    - 对外国城市标准化时尽量带上国家名；对中国境内地名可不带国家名
    - 清除 <think> 污染；不写 coordinate
    """
    # 常见国家中文名称集合（用于判断与补全）
    COUNTRY_NAMES = {
        "中国", "美国", "日本", "韩国", "朝鲜", "俄罗斯", "英国", "法国",
        "德国", "加拿大", "澳大利亚", "印度", "巴基斯坦", "印度尼西亚",
        "越南", "菲律宾", "泰国", "新加坡", "马来西亚", "意大利", "荷兰",
        "西班牙", "以色列", "沙特", "阿联酋"
    }

    # 非地名词黑名单（这些词往往说明不是具体地名，例如“日本国会”“日本内阁”）
    NON_PLACE_KEYWORDS = ["国会", "议会", "内阁", "政府", "首相府", "外交部", "内阁府", "议员"]

    def is_empty(v):
        return v is None or (isinstance(v, str) and v.strip() == "")

    # 输入校验
    if not isinstance(space_position, dict):
        return {}

    # 如果 region 存在且非空 → 不处理任何地名
    region_val = space_position.get("region", "")
    if isinstance(region_val, str) and region_val.strip():
        return space_position

    raw_place = str(space_position.get("place_name", "") or "").strip()

    # 无 place_name → 删除该字段（返回空 dict）
    if is_empty(raw_place):
        return {}

    # 清除 <think> 等噪声
    raw_place = re.sub(r"<think>.*?</think>", "", raw_place, flags=re.S).strip()

    # 如果包含明显的非地名关键词，保守策略：**不改变**，返回原始 place_name（避免误修改）
    for kw in NON_PLACE_KEYWORDS:
        if kw in raw_place:
            return {"place_name": raw_place}

    # 调用模型进行地名规范化（模型应被设计为对非地名返回空）
    try:
        normalized = resolve_location_with_qwen3(raw_place, model)
    except Exception:
        normalized = None

    # 如果模型返回空或 None，保守策略：不覆盖原始 place_name
    if not normalized:
        return {"place_name": raw_place}

    normalized = normalized.strip()

    # 若模型把非地名转成了类似国家/机构名，或返回仍包含非地名关键词，则视为不可用 -> 保留原值
    for kw in NON_PLACE_KEYWORDS:
        if kw in normalized:
            return {"place_name": raw_place}

    # 应用 location_mapping（配置表中的别名映射）
    if normalized in location_mapping:
        normalized = location_mapping[normalized].strip()

    if is_empty(normalized):
        return {"place_name": raw_place}

    # 判断是否包含国家名（用于是否需要补国家名）
    has_country = any(cn in normalized for cn in COUNTRY_NAMES)

    # 判断原始文本是否明显包含国家提示（例如 "日本国会" 包含 '日本'）用于补全国家信息
    detected_country_from_raw = None
    for cn in COUNTRY_NAMES:
        if cn in raw_place:
            detected_country_from_raw = cn
            break

    # 如果标准化结果看起来像国内地名（含市/省/县/区等后缀），且国家是中国，则无需加国家
    domestic_suffixes = ["省", "市", "县", "区", "自治州", "自治县"]
    looks_like_domestic = any(normalized.endswith(suf) for suf in domestic_suffixes)

    # 如果标准化结果不含国家名且原始或 mapping 提示为国外城市，则尽量把国家名补上
    if (not has_country) and detected_country_from_raw:
        # 如果检测到原文中写了国家（例如 '日本 台北' 或 '日本 东京'），将国家加入
        # 优先使用 "国家 + 空格 + 城市" 的形式（中文）
        normalized = f"{detected_country_from_raw} {normalized}"
        has_country = True
    # 进一步：如果标准化结果看起来像“单一城市名”（不带省/市/国），且未包含国家名，且原始文本中并未显示国内省市提示，则保守**尝试**推断是否为国外城市：
    if (not has_country) and (not looks_like_domestic):
        # 简单策略：若 raw_place 含有英文逗号后面的文本，则把其视为国家线索；否则保留 normalized（不附加国家）
        m = re.search(r"[,，]\s*([^\d]+)$", raw_place)
        if m:
            tail = m.group(1).strip()
            # 若尾部包含国家中文名或常见英文国家名（简短判断），则把它作为国家名
            for cn in COUNTRY_NAMES:
                if cn in tail:
                    normalized = f"{cn} {normalized}"
                    has_country = True
                    break
        # 否则不强行补国家，保留 normalized

    # 最终再做一次简单合法性检查：若 normalized 仍然包含非地名关键词则放弃
    for kw in NON_PLACE_KEYWORDS:
        if kw in normalized:
            return {"place_name": raw_place}

    # 返回清洗后的 place_name（国内不强制加国家，国外尽量带国家）
    return {"place_name": normalized}


def strip_title_prefix(name: str) -> str:
    """去掉人名前的职务/头衔前缀，例如 '法国总统马克龙' -> '马克龙'，组织如 '美国国务院' 则直接保留"""
    # 常见的人名前缀模式（可以继续扩展）
    title_patterns = [
        r".*总统",  # 法国总统马克龙 -> 马克龙
        r".*总理",
        r".*首相",
        r".*部长",
        r".*大使",
        r".*国务卿",
        r".*书记",
        r".*主席",
    ]
    for pat in title_patterns:
        m = re.match(pat + r"([\u4e00-\u9fa5A-Za-z·]+)$", name)
        if m:
            return m.group(1)
    return name


def clean_event_relations(relations: list):
    """过滤事件关系中的实体名（仅针对 TARGETS，去掉头衔前缀）"""
    cleaned_relations = []
    for rel in relations:
        parts = rel.split("  ")
        if len(parts) != 3:
            continue

        subj, obj, rel_type = parts
        if rel_type == "TARGETS":
            candidate = strip_title_prefix(obj.strip())
            filtered_name = filter_entity_name(candidate)
            if filtered_name:  # 过滤掉空或黑名单
                cleaned_relations.append(f"{subj}  {filtered_name}  TARGETS")
        else:
            cleaned_relations.append(rel)
    return cleaned_relations


def _get_reference_datetime_from_record(record: dict) -> datetime:
    """
    优先取 record['time']、['crawl_time']、['publish_date']（若为 str 尝试 isoparse），
    再 fallback _id(ObjectId).generation_time，最后 utcnow。
    """
    if not isinstance(record, dict):
        return datetime.utcnow()
    for k in ('time', 'crawl_time', 'publish_date', 'publish_time'):
        if k in record and record[k]:
            v = record[k]
            if isinstance(v, datetime):
                return v
            try:
                # 兼容 ISO 字符串
                return datetime.fromisoformat(v)
            except Exception:
                pass
    oid = record.get('_id')
    if isinstance(oid, BsonObjectId):
        try:
            return oid.generation_time
        except Exception:
            pass
    return datetime.utcnow()


def _fill_year_into_moment(moment_str: str, ref_year: int, ref_dt: datetime) -> str:
    """
    规则最小化：只在字符串里为月/日或月添加 `YYYY年` 前缀或替换模糊词。
    - 若已有 YYYY年，返回原样；
    - 处理模糊词（去年/上月/昨天等）转为带年（或带年月日）；
    - 处理 '9月22日' -> 'YYYY年9月22日'
    - 处理 '10月起' -> 'YYYY年10月起'
    - 其它不匹配的保留原样
    """
    s = moment_str.strip()
    # 若已有明确年份，直接返回
    if RE_YEAR.search(s):
        return s

    # 模糊词处理（按 ref_dt）
    for token in FUZZY:
        if token in s:
            if token in ('昨天', '昨日'):
                dt = (ref_dt - timedelta(days=1)).date()
                return f"{dt.year}年{dt.month}月{dt.day}日"
            if token in ('今天', '今日'):
                dt = ref_dt.date()
                return f"{dt.year}年{dt.month}月{dt.day}日"
            if token == '上月':
                # 上月保守处理为上月的“1日”形式：YYYY年M月（不指定日）
                year = ref_dt.year if ref_dt.month > 1 else (ref_dt.year - 1)
                month = ref_dt.month - 1 if ref_dt.month > 1 else 12
                return f"{year}年{month}月"
            if token == '去年':
                return f"{ref_dt.year - 1}年"
            if token == '今年':
                return f"{ref_dt.year}年"

    # 月日形式（如 9月22日）
    m = RE_FULL_DATE.search(s)
    if m:
        # 在原字符串前加年份（保留原月日形式）
        return f"{ref_year}年{s}"

    # 仅月形式（例如 "10月起", "10月"）
    m2 = RE_MONTH_ONLY.search(s)
    if m2:
        # 插入年份前缀
        return f"{ref_year}年{s}"

    # 其他形式保持不变
    return s


def _fill_year_into_period(period_str: str, ref_year: int, ref_dt: datetime) -> str:
    """
    对 period 字符串做最小改动：仅在两端缺少年份时按规则插入年份。
    例如：
    - "9月1日 至 9月30日" -> "YYYY年9月1日 至 YYYY年9月30日"
    - "2024年1月 至 2024年12月" -> 保留不变
    - "2024年1月 至 12月" -> 保留左侧年份并为右侧补同年 "2024年12月"
    """
    s = period_str.strip()
    # 若整段已经含有年份区间则返回原样
    if RE_YEAR_RANGE.search(s) or RE_YEAR.search(s):
        # 如果部分带年、部分不带年，需要补全右侧（例如 "2024年1月 至 12月"）
        # 简单策略：若左侧含年且右侧不含年，则把左侧年填到右侧
        parts = re.split(r'(至|-|到)', s)
        if len(parts) >= 3:
            left = parts[0].strip()
            sep = parts[1]
            right = parts[2].strip()
            if RE_YEAR.search(left) and not RE_YEAR.search(right):
                # extract year from left
                y_match = RE_YEAR.search(left)
                if y_match:
                    year_text = y_match.group(0)  # e.g. '2024年'
                    return f"{left} {sep} {year_text}{right}"
        return s

    # 若两端都不含年，则为每端补上 ref_year（粗略处理）
    parts = re.split(r'(至|-|到)', s)
    if len(parts) >= 3:
        left = parts[0].strip()
        sep = parts[1]
        right = parts[2].strip()
        left_filled = left
        right_filled = right
        # 为左/右端分别补年（若存在月日/仅月）
        if RE_YEAR.search(left) is None:
            if RE_FULL_DATE.search(left) or RE_MONTH_ONLY.search(left):
                left_filled = f"{ref_year}年{left}"
        if RE_YEAR.search(right) is None:
            if RE_FULL_DATE.search(right) or RE_MONTH_ONLY.search(right):
                right_filled = f"{ref_year}年{right}"
        return f"{left_filled} {sep} {right_filled}"
    # 其它不处理
    return s


def normalize_event_time(events, record=None):
    """
    原名保持：对事件列表做“年份补全”——仅修改 time_position 内的字符串（moment 或 period），不改变格式结构。
    record: 原始文章 record，用于获取参考时间（record['time'] / ['crawl_time'] / _id）
    """
    if not isinstance(events, list):
        return
    # 获取参考 datetime
    ref_dt = _get_reference_datetime_from_record(record) if record is not None else datetime.utcnow()
    ref_year = ref_dt.year

    for ev in events:
        tp = ev.get("time_position", {})
        if not isinstance(tp, dict):
            continue

        # 处理 moment（只在 moment 字符串缺年时补年）
        if "moment" in tp and tp.get("moment"):
            orig = str(tp["moment"]).strip()
            new_m = _fill_year_into_moment(orig, ref_year, ref_dt)
            tp["moment"] = new_m  # 仅替换字符串，不添加新字段

        # 处理 period（区间）
        if "period" in tp and tp.get("period"):
            origp = str(tp["period"]).strip()
            new_p = _fill_year_into_period(origp, ref_year, ref_dt)
            tp["period"] = new_p

        # 不更改其他字段或增加标记（保持格式不变）
        ev["time_position"] = tp


def clear_screen():
    """清空终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')


def display_progress(current, total, category_counts, current_record_info=None):
    """显示实时进度和统计信息"""
    clear_screen()

    print("=" * 60)
    print("🧠 新闻分类与结构化提取实时监控")
    print("=" * 60)

    # 显示总体进度
    progress_percent = (current / total) * 100 if total > 0 else 0
    progress_bar = "█" * int(progress_percent // 2) + "░" * (50 - int(progress_percent // 2))
    print(f"\n📊 总体进度: [{progress_bar}] {current}/{total} ({progress_percent:.1f}%)")

    # 显示当前处理的记录信息
    if current_record_info:
        print(f"\n 当前处理:")
        print(f"   ID: {current_record_info.get('id', 'N/A')}")
        print(f"   分类: {current_record_info.get('category', '处理中...')}")
        print(f"   状态: {current_record_info.get('status', '处理中...')}")

    # 显示分类统计
    print(f"\n 分类统计:")
    total_processed = sum(category_counts.values())
    for category, count in category_counts.items():
        if count > 0:
            percentage = (count / total_processed) * 100 if total_processed > 0 else 0
            print(f"   {category}: {count} 条 ({percentage:.1f}%)")

    print(f"\n💾 event_result.json (实时保存)")
    print("-" * 60)


def normalize_relation_ids(relations: list) -> list:
    """
    把关系三元组里的事件 ID 统一规范成 5 位数字：
    Th_POL-7 / Th_POL-007 / Th_POL-0007 → Th_POL-00007
    其他非 ID 的对象（比如“中国时报”）保持不变
    """
    normalized = []
    for rel in relations:
        parts = rel.strip().split()
        if len(parts) != 3:
            continue
        src, tgt, rtype = parts

        # 处理第一个字段（一定是事件 ID）
        m1 = ID_PATTERN.match(src)
        if m1:
            prefix, _, num = m1.groups()
            src = f"{prefix}-{int(num):05d}"

        # 第二个字段：可能是事件 ID，也可能是实体名
        m2 = ID_PATTERN.match(tgt)
        if m2:
            prefix, _, num = m2.groups()
            tgt = f"{prefix}-{int(num):05d}"

        normalized.append(f"{src}  {tgt}  {rtype}")
    return normalized


def extract_event_relations(llm, events: list) -> list:
    """
    输入事件列表，调用事件关系识别链，返回事件之间的关系三元组列表。
    输出格式如：["Th_POL-00001  Th_POL-00002  TRIGGERS", "Th_POL-00003  佩洛西  TARGETS"]
    """
    if not events or len(events) < 2:
        return []

    try:
        def build_event_description(ev: dict) -> str:
            """根据新拼接逻辑生成事件描述"""
            parts = []
            # 事件名称
            if ev.get("event_name"):
                parts.append(ev["event_name"])
            # 触发词
            if ev.get("trigger_word"):
                tw = [w for w in ev["trigger_word"] if w]
                if tw:
                    parts.append("触发词: " + "、".join(tw))
            # 时间
            time_info = ev.get("time_position", {}).get("moment")
            if time_info:
                parts.append(f"时间: {time_info}")
            # 地点
            space = ev.get("space_position", {})
            place_name = space.get("place_name")
            region = space.get("region")

            if place_name:
                parts.append(f"地点：{place_name}")
            elif region:
                parts.append(f"地点：{region}")

            # 实体：人物 / 组织（只取 name，不要 role / emotion）
            rc = ev.get("relationship_characteristics", {})
            role_texts = []
            if rc.get("subject_person", {}).get("name"):
                role_texts.append(rc["subject_person"]["name"])
            if rc.get("related_party_person", {}).get("name"):
                role_texts.append(rc["related_party_person"]["name"])
            if rc.get("subject_organization", {}).get("name"):
                role_texts.append(rc["subject_organization"]["name"])
            if rc.get("related_party_organization", {}).get("name"):
                role_texts.append(rc["related_party_organization"]["name"])
            if role_texts:
                parts.append("涉及人物/组织: " + ", ".join(role_texts))
            # 类型特有字段
            attr = ev.get("attribute_characteristics", {})
            # 政治事件
            if attr.get("key_action"):
                parts.append(f"关键动作：{attr['key_action']}")
            # 军事事件
            if attr.get("military_means"):
                parts.append(f"军事手段：{attr['military_means']}")
            # 经济事件
            if attr.get("policy_tool"):
                parts.append(f"政策工具：{attr['policy_tool']}")
            # 事件类型
            ev_type = ev.get("attribute_characteristics", {}).get("event_type")
            if ev_type:
                parts.append(f"类型: {ev_type}")
            # 设施（airport/port）用于 facility target
            airport = ev.get("attribute_characteristics", {}).get("airport")
            if airport:
                parts.append(f"机场：{airport}")
            port = ev.get("attribute_characteristics", {}).get("port")
            if port:
                parts.append(f"港口：{port}")
            return f"{ev['event_id']}: " + "；".join(parts)

        # 拼接所有事件描述
        events_text = "\n".join(
            build_event_description(ev) for ev in events if ev.get("event_id")
        )
        # 调用 LLM 关系识别链
        relation_chain = create_event_relation_chain(llm)
        relations_raw = relation_chain.invoke({"events_text": events_text})
        # 解析输出
        if isinstance(relations_raw, str):
            relations = [
                line.strip()
                for line in relations_raw.strip().splitlines()
                if line.strip()
                   and not line.strip().startswith("<")
                   and len(line.strip().split()) == 3
            ]
        else:
            relations = []
        # 把关系里的事件 ID 统一成 5 位
        relations = normalize_relation_ids(relations)
        return relations
    except Exception as e:
        print(f"⚠️ 事件关系识别失败: {e}")
        return []


# 解析失败重复解析
def invoke_with_retry(chain, inputs, max_retries=3, delay_seconds=1):
    for attempt in range(max_retries):
        output = chain.invoke(inputs)
        try:
            if isinstance(output, dict):
                return output
            else:
                return json.loads(output)
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试解析 JSON 失败: {e}")
            if attempt < max_retries - 1:
                print("🔁 重试中...\n")
                time.sleep(delay_seconds)
            else:
                raise RuntimeError(f"达到最大重试次数({max_retries})，仍无法解析JSON")


def run_on_collection(collection_name=None):
    """
    从指定集合读取数据，导出 JSON，调用 main 抽取，结果写回集合
    collection_name: 默认为 None，将使用 settings.COLL_INTERIM
    """
    # 【修改点 4】动态集合选择
    target_col_name = collection_name if collection_name else settings.COLL_INTERIM

    # 连接 MongoDB
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    collection = db[target_col_name]

    docs = list(collection.find({}))
    if not docs:
        print(f"⚠️ 集合 {target_col_name} 没有数据可抽取事件")
        return

    print(f"📝 开始结构化信息抽取，共 {len(docs)} 条 (Collection: {target_col_name})")

    # 1️⃣ 导出集合为 interim.json（自动处理 ObjectId、datetime）
    interim_json_path = "interim.json"
    with open(interim_json_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2, default=default_serializer)

    # 2️⃣ 调用 main() 函数
    main(input_file=interim_json_path, output_file="event_result.json")

    # 3️⃣ 读取抽取结果 event_result.json
    with open("event_result.json", "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # 4️⃣ 按要求写回集合
    for item in all_results:
        _id_str = item.get("_id")
        if not _id_str:
            continue
        try:
            _id = ObjectId(_id_str)
        except Exception:
            _id = _id_str  # 如果不是 ObjectId 就直接用字符串
        # 更新/插入记录
        collection.update_one(
            {"_id": _id},
            {"$set": {
                "source": item.get("source"),
                "predicted_category": item.get("predicted_category"),
                "predicted_subcategory": item.get("predicted_subcategory"),
                "structured_data": item.get("structured_data")
            }},
            upsert=True
        )

    print("📝 信息抽取完成并已写回 MongoDB")


def main(records_to_process=None, input_file="interim.json", output_file="event_result.json"):
    """主函数：两阶段处理 - 分类 + 结构化提取"""
    # ========== 新增逻辑 ==========
    if records_to_process is not None:
        # 外部传入了数据（数据库已经查好）
        data = records_to_process
    else:
        # 原始逻辑：从文件中加载
        json_file_path = input_file
        output_file = output_file
        # 加载数据
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)

            if isinstance(input_data, list):
                records_to_process = input_data
            elif isinstance(input_data, dict) and "RECORDS" in input_data and isinstance(input_data["RECORDS"], list):
                records_to_process = input_data["RECORDS"]
            else:
                print("错误: JSON 文件格式错误，应为列表或包含 'RECORDS' 的字典。")
                sys.exit(1)

        except FileNotFoundError:
            print(f"错误: 数据文件 '{json_file_path}' 未找到。")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"错误: 文件 '{json_file_path}' 不是有效的JSON格式。")
            sys.exit(1)

        if not isinstance(records_to_process, list):
            print("错误: JSON文件中未找到'RECORDS'键或其内容不是列表。")
            sys.exit(1)

    # 【修改点 5】获取模型实例
    model = get_extraction_model()

    # 创建处理链
    classification_chain = create_classification_chain(model)
    political_chain = create_political_extraction_chain_multi(model)
    military_chain = create_military_extraction_chain_multi(model)
    economic_chain = create_economic_extraction_chain_multi(model)
    # 初始化
    total_records = len(records_to_process)
    all_results = []
    category_counts = {"台海军事": 0, "台海政治": 0, "台海经济": 0, "处理失败": 0}
    # 事件编号初始化记录
    event_id_counters = {
        "台海经济": 1,
        "台海军事": 1,
        "台海政治": 1,
    }
    category_prefix_map = {
        "台海经济": "Th_ECON",
        "台海军事": "Th_MIL",
        "台海政治": "Th_POL",
    }
    # 🌐 地理编码缓存，避免重复请求。全局缓存
    visited_addresses = {}

    def save_results():
        """保存当前所有结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 显示初始界面
    display_progress(0, total_records, category_counts)
    print("\n⏳ 准备开始处理...")

    try:
        for i, record in enumerate(records_to_process):
            current_record_info = {
                'id': record.get('_id', f'Record-{i + 1}'),
                'category': '分类中...',
                'status': '处理中...'
            }

            # 显示当前处理状态
            display_progress(i, total_records, category_counts, current_record_info)

            # 提取内容
            content = record.get("content", "").strip() or record.get("title", "").strip()

            if not content:
                record['predicted_category'] = '内容为空'
                record['structured_data'] = {}
                category_counts["处理失败"] += 1
                current_record_info.update({
                    'category': '内容为空',
                    'status': '❌ 跳过'
                })
            else:
                try:
                    # 第一阶段：分类
                    current_record_info.update({
                        'category': '一级分类中...',
                        'status': '🔍 一级分类'
                    })
                    display_progress(i, total_records, category_counts, current_record_info)

                    classification_result = classification_chain.invoke({"content": content})
                    predicted_category = classification_result.get("category", "解析失败")

                    if predicted_category not in PRESET_CATEGORIES:
                        record['predicted_category'] = '一级分类无效'
                        record['structured_data'] = {}
                        category_counts["处理失败"] += 1
                        current_record_info.update({
                            'category': '一级分类无效',
                            'status': '❌ 无效分类'
                        })
                    else:
                        record['predicted_category'] = predicted_category
                        category_counts[predicted_category] += 1

                        # 第二步分类（Level 2-3 子类）
                        current_record_info.update({
                            'category': predicted_category,
                            'status': '🔍 子类识别中'
                        })
                        display_progress(i, total_records, category_counts, current_record_info)

                        try:
                            sub_chain = create_subclassification_chain(model, predicted_category)
                            sub_result = sub_chain.invoke({"content": content})
                            predicted_subcategory = sub_result.get("subcategory", "子类识别失败")
                        except Exception as e:
                            print(f"[子类识别异常] {str(e)}")
                            predicted_subcategory = "子类识别失败"

                        record['predicted_subcategory'] = predicted_subcategory

                        # 第三阶段：结构化提取
                        current_record_info.update({
                            'category': predicted_category,
                            'status': ' 结构化提取'
                        })
                        display_progress(i, total_records, category_counts, current_record_info)

                        if predicted_category == "台海政治":
                            extraction_result = invoke_with_retry(political_chain, {"content": content})
                        elif predicted_category == "台海军事":
                            extraction_result = invoke_with_retry(military_chain, {"content": content})
                        elif predicted_category == "台海经济":
                            extraction_result = invoke_with_retry(economic_chain, {"content": content})
                        else:
                            extraction_result = {"events": []}
                        # 加入 event_id 编号
                        events = extraction_result.get("events", [])
                        clean_event_entities(events, model)  # 清洗人名和组织
                        clean_event_locations(events, model)  # 清洗地理位置
                        normalize_event_time(events, record=record)  # 以 record 中的 time/crawl_time/_id 为参考年
                        # 编号 + 地理编码
                        prefix = category_prefix_map.get(predicted_category, "Th_UNK")
                        counter = event_id_counters.get(predicted_category, 1)

                        for ev in events:
                            ev["event_id"] = f"{prefix}-{counter:05d}"
                            counter += 1
                            # 地理编码：从 place_name 或 region 获取地址
                            sp = ev.get("space_position", {})
                            addr = sp.get("place_name") or sp.get("region")
                            if addr:
                                if addr in visited_addresses:
                                    sp["coordinate"] = visited_addresses[addr]
                                else:
                                    coord = geocode_address(addr)
                                    if coord:
                                        sp["coordinate"] = coord
                                        visited_addresses[addr] = coord  # 加入缓存
                                        time.sleep(1)
                                    else:
                                        print(f"⚠️ 地址匹配失败：{addr}")
                                ev["space_position"] = sp
                        event_id_counters[predicted_category] = counter  # 更新计数器
                        relations = extract_event_relations(model, events)  # 事件关系抽取
                        relations = clean_event_relations(relations)  # 实体name清洗
                        # 写入 structured_data
                        record['structured_data'] = {
                            "events": events,
                            "event_relations": relations
                        }

                        current_record_info.update({
                            'status': '✅ 完成'
                        })

                except Exception as e:
                    record['predicted_category'] = '处理失败'
                    record['structured_data'] = {}
                    category_counts["处理失败"] += 1
                    current_record_info.update({
                        'category': '处理失败',
                        'status': f'❌ 错误: {str(e)[:30]}...'
                    })

            all_results.append(record)
            save_results()  # 立即保存
            # 显示完成状态
            display_progress(i + 1, total_records, category_counts, current_record_info)

    except KeyboardInterrupt:
        display_progress(len(all_results), total_records, category_counts)
        print("\n\n⚠️  检测到中断操作！已处理的结果已保存到文件中...")

    finally:
        # 最终保存和显示
        save_results()
        display_progress(len(all_results), total_records, category_counts)

        if all_results:
            print(f"\n\n✅ 处理完成！")
            print(f" 总计 {len(all_results)} 条结果已保存到 '{output_file}'")

            # 显示成功处理的示例
            successful_results = [r for r in all_results if
                                  r.get('structured_data') and r.get('predicted_category') in PRESET_CATEGORIES]
            if successful_results:
                print(f"\n 成功提取 {len(successful_results)} 条结构化数据")
            print(f"\n 完整结果请查看: {output_file}")
        else:
            print("\n❌ 未处理任何记录。")
        return all_results


if __name__ == "__main__":
    main()