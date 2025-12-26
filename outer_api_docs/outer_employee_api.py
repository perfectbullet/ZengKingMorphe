# 数字员工外部API文档实体对象定义文件，实体对象定义的python版本是python 3.7


from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


@dataclass
class SysDigitalLanguage:
    """SysDigitalLanguage"""
    """语言"""
    language: Optional[str] = None
    """备注"""
    remark: Optional[str] = None


@dataclass
class SysDigitalTone:
    """SysDigitalTone"""
    """性别: 1-男 2-女, 0-未知"""
    gender: Optional[int] = None
    """备注"""
    remark: Optional[str] = None
    """音色"""
    tone: Optional[str] = None


class TypeEnum(Enum):
    """数字员工类型: AVATAR("头像"), HALF("半身"), FULL("全身")"""
    AVATAR = "AVATAR"
    FULL = "FULL"
    HALF = "HALF"


@dataclass
class TeamDigitalEmployee:
    """数字员工信息
    
    TeamDigitalEmployee
    """
    """创建时间"""
    create_time: Optional[datetime] = None
    """创建人ID/雇佣人ID"""
    create_user_id: Optional[int] = None
    digital_code: Optional[str] = None
    """数字员工性别: 1-男 2-女, 0-未知"""
    gender: Optional[int] = None
    """团队数字员工ID"""
    id: Optional[int] = None
    intro: Optional[str] = None
    """语言"""
    language: Optional[str] = None
    languages: Optional[List[SysDigitalLanguage]] = None
    model_image: Optional[str] = None
    """员工姓名"""
    name: Optional[str] = None
    """值班状态: 0-休息中，1-值班中"""
    onduty_status: Optional[int] = None
    portrait: Optional[str] = None
    """职位"""
    position: Optional[str] = None
    """团队ID"""
    team_id: Optional[int] = None
    """音色"""
    tone: Optional[str] = None
    tones: Optional[List[SysDigitalTone]] = None
    """数字员工类型: AVATAR("头像"), HALF("半身"), FULL("全身")"""
    type: Optional[TypeEnum] = None
    """更新时间"""
    update_time: Optional[datetime] = None


@dataclass
class DigitalDatasetFAQDO:
    """DigitalDatasetFaqDO，数字员工-FAQ问答信息实体对象"""
    answers: Optional[List[str]] = None
    """创建时间"""
    create_time: Optional[datetime] = None
    """创建者用户ID"""
    create_user_id: Optional[int] = None
    """生效结束时间"""
    end_time: Optional[datetime] = None
    """主键"""
    id: Optional[int] = None
    """是否澄清：0=不澄清，1=澄清"""
    is_clear: Optional[int] = None
    """是否启用：0=不启用，1=启用"""
    is_enable: Optional[int] = None
    """标准问题"""
    question_name: Optional[str] = None
    similar_questions: Optional[List[str]] = None
    """生效开始时间"""
    start_time: Optional[datetime] = None
    """团队id"""
    team_id: Optional[int] = None
    """更新时间"""
    update_time: Optional[datetime] = None


@dataclass
class DigitalDatasetVideoDO:
    """DigitalDatasetVideoDO，数字员工-视频资源信息实体对象"""
    """创建时间"""
    create_time: Optional[datetime] = None
    """创建者用户ID"""
    create_user_id: Optional[int] = None
    """生效结束时间"""
    end_time: Optional[datetime] = None
    """删除标识：-1=已删除，1=正常"""
    flag: Optional[int] = None
    """主键"""
    id: Optional[int] = None
    """是否生效：0=不生效，1=生效"""
    is_enable: Optional[int] = None
    """是否知识增强：0=不增强，1=增强"""
    is_enhance: Optional[int] = None
    """是否发布到RAG上：0=未发布，1=已发布"""
    is_publish: Optional[int] = None
    """RAG系统任务ID"""
    rag_task_id: Optional[str] = None
    """RAG视频ID"""
    rag_video_id: Optional[str] = None
    """系统资源ID"""
    resource_id: Optional[int] = None
    """生效开始时间"""
    start_time: Optional[datetime] = None
    """状态：0=未学习，1=学习中，2=学习成功，3=学习失败，4=知识增强中"""
    status: Optional[int] = None
    """团队id"""
    team_id: Optional[int] = None
    """视频名称"""
    video_name: Optional[str] = None


@dataclass
class DigitalDatasetDO:
    """DigitalDatasetDO，数字员工-知识库信息实体对象"""
    """创建时间"""
    create_time: Optional[datetime] = None
    """创建者用户ID"""
    create_user_id: Optional[int] = None
    """删除标识：-1=已删除，1=正常"""
    flag: Optional[int] = None
    """主键"""
    id: Optional[int] = None
    """是否生效：0=不生效，1=生效"""
    is_enable: Optional[int] = None
    """是否发布到RAG上：0=未发布，1=已发布"""
    is_publish: Optional[int] = None
    """知识库名称"""
    name: Optional[str] = None
    """RAG系统知识库ID"""
    rag_dataset_id: Optional[str] = None
    """团队id"""
    team_id: Optional[int] = None


@dataclass
class TeamDigitalEmployeeKnowledge:
    """知识库设置
    
    TeamDigitalEmployeeKnowledge
    """
    faqs: Optional[List[DigitalDatasetFAQDO]] = None
    medias: Optional[List[DigitalDatasetVideoDO]] = None
    question_banks: Optional[Dict[str, Any]] = None
    rag_datasets: Optional[List[DigitalDatasetDO]] = None


@dataclass
class DigitalMajorBankWordVo:
    """DigitalMajorBankWordVo"""
    """相似词，多个词用逗号分割"""
    similar_words: Optional[str] = None
    """更新时间"""
    update_time: Optional[datetime] = None
    """词条"""
    word: Optional[str] = None


@dataclass
class DigitalMajorBankVo:
    """DigitalMajorBankVo"""
    """词库名称"""
    bank_name: Optional[str] = None
    """词库ID"""
    id: Optional[int] = None
    """更新时间"""
    update_time: Optional[datetime] = None
    """包含的词条列表"""
    words: Optional[List[DigitalMajorBankWordVo]] = None


@dataclass
class TeamDigitalEmployeeWord:
    """专业词库设置
    
    TeamDigitalEmployeeWord
    """
    """是否同义词重写"""
    is_synonym_rewrite: Optional[bool] = None
    major_banks: Optional[List[DigitalMajorBankVo]] = None


@dataclass
class TeamDigitalEmployeePlugin:
    """TeamDigitalEmployeePlugin"""
    plugin_code: Optional[str] = None
    plugin_icon: Optional[str] = None
    """插件ID"""
    plugin_id: Optional[int] = None
    plugin_intro: Optional[str] = None
    plugin_name: Optional[str] = None
    """插件参数"""
    plugin_params: Optional[Dict[str, Any]] = None


@dataclass
class TeamDigitalEmployeePrologue:
    """开场白设置
    
    TeamDigitalEmployeePrologue
    """
    faqs: Optional[List[DigitalDatasetFAQDO]] = None
    """是否开启开场热门问题"""
    is_opening_questions: Optional[bool] = None
    """自定义开场问题列表"""
    my_questions: Optional[List[str]] = None
    """开场白内容"""
    prologue: Optional[str] = None
    """热门问题类型: 1-自动推荐 2-FAQ 3-自定义"""
    question_type: Optional[int] = None


@dataclass
class TeamDigitalEmployeeRole:
    """人设设置
    
    TeamDigitalEmployeeRole
    """
    """人设"""
    persona: Optional[str] = None
    style: Optional[str] = None
    style_desc: Optional[str] = None


@dataclass
class ChatRuleVo:
    """对话规则
    
    ChatRuleVo
    """
    """FAQ相似度阈值"""
    faq_sim_threshold: Optional[float] = None
    """FAQ最多推荐数量"""
    faq_top_k: Optional[int] = None
    """是否支持多模态"""
    is_multimodal: Optional[bool] = None


@dataclass
class DigitalSensitiveBankVo:
    """DigitalSensitiveBankVo"""
    """词库ID"""
    bank_name: Optional[str] = None
    """词库ID"""
    id: Optional[int] = None
    """是否生效：0=不生效，1=生效"""
    is_enable: Optional[int] = None
    """更新时间"""
    update_time: Optional[datetime] = None
    """包含的敏感词条列表"""
    words: Optional[List[str]] = None


@dataclass
class SafeRuleVo:
    """安全规则
    
    SafeRuleVo
    """
    """是否拒绝回答"""
    is_reject_answer: Optional[bool] = None
    """拒绝回答时的回复内容"""
    reject_answer: Optional[str] = None
    """敏感词库ID列表"""
    sensitive_bank_ids: Optional[List[int]] = None
    """敏感词库列表"""
    sensitive_banks: Optional[List[DigitalSensitiveBankVo]] = None


@dataclass
class LLMReplyVo:
    """模型闲聊回复，对应类型：notMatchReplyType=1
    
    LLMReplyVo
    """
    """是否启用自定义提示词"""
    is_my_prompt: Optional[bool] = None
    """是否显示标识"""
    is_show_sign: Optional[bool] = None
    """是否联网搜索"""
    is_web_search: Optional[bool] = None
    """自定义提示词内容，提示词中可以包含变量，变量格式：{{var_name}}，可以包含设置中的其他变量,如：{{style:
    对应角色设定中的风格}},{{persona：对应角色设定中的人设}}；也可以包含系统内置变量，如：{{current_time:系统当前时间戳}}, {{query:用户输入}}
    """
    my_prompt: Optional[str] = None


@dataclass
class UnusualRuleVo:
    """异常规则
    
    UnusualRuleVo
    """
    """系统异常时的回复内容"""
    excepiton_reply: Optional[str] = None
    """固定回复列表，对应类型：notMatchReplyType=0"""
    fixed_replys: Optional[List[str]] = None
    """模型闲聊回复，对应类型：notMatchReplyType=1"""
    llm_reply: Optional[LLMReplyVo] = None
    """未匹配时的回复类型：0：固定话术，1：模型闲聊回复"""
    not_match_reply_type: Optional[int] = None


@dataclass
class TeamDigitalEmployeeRule:
    """规则设置
    
    TeamDigitalEmployeeRule
    """
    """对话规则"""
    chat_rule: Optional[ChatRuleVo] = None
    id: Optional[int] = None
    """安全规则"""
    safe_rule: Optional[SafeRuleVo] = None
    """异常规则"""
    unusual_rule: Optional[UnusualRuleVo] = None


@dataclass
class DigitalEmployeeSettingVo:
    """数字员工设置
    
    DigitalEmployeeSettingVo
    """
    """知识库设置"""
    knowledge: Optional[TeamDigitalEmployeeKnowledge] = None
    """专业词库设置"""
    major_word: Optional[TeamDigitalEmployeeWord] = None
    """插件设置"""
    plugins: Optional[List[TeamDigitalEmployeePlugin]] = None
    """开场白设置"""
    prologue: Optional[TeamDigitalEmployeePrologue] = None
    """人设设置"""
    role: Optional[TeamDigitalEmployeeRole] = None
    """规则设置"""
    rule: Optional[TeamDigitalEmployeeRule] = None


@dataclass
class DigitalEmployeeAPIVo:
    """数据
    
    DigitalEmployeeApiVo
    """
    """数字员工信息"""
    employee: Optional[TeamDigitalEmployee] = None
    """数字员工设置"""
    setting: Optional[DigitalEmployeeSettingVo] = None


@dataclass
class Response:
    """ResultDigitalEmployeeApiVo"""
    """数据"""
    data: Optional[DigitalEmployeeAPIVo] = None
    error: Optional[str] = None
    """消息"""
    message: Optional[str] = None
    """状态码，200表示成功"""
    status: Optional[int] = None
    success: Optional[bool] = None
