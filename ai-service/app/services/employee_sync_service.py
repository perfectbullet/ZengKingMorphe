"""
员工数据同步服务
封装从 Java API 获取和同步员工数据的逻辑
"""
import httpx
import os

from typing import Optional, Dict, Any, List
from app.core.logging import get_logger
from app.models.database import DigitalEmployeeConfigModel, DigitalEmployeeConfigSettingModel

logger = get_logger(__name__)


class EmployeeSyncService:
    """员工数据同步服务"""

    def __init__(self):
        self.api_url = os.getenv(
            "EXTERNAL_EMPLOYEE_API_URL",
            "http://192.168.9.39/edu-api/avatar/api/digitalEmployee/get"
        )

    async def fetch_and_sync(self, employee_id: str, db) -> Optional[str]:
        """
        从 Java API 获取员工数据并同步到 MongoDB

        Args:
            employee_id: 员工ID，可以是字符串 "29" 或数字 29
            db: MongoDB 数据库实例

        Returns:
            同步成功的 employee_id（字符串格式），失败返回 None

        类型转换：
            - API 调用: str → int (Java API 要求数字参数)
            - MongoDB: int → str (统一使用字符串存储)
        """
        # 步骤1: 从 Java API 获取数据
        api_data = await self._fetch_from_api(employee_id)
        if not api_data:
            return None

        # 步骤2: 同步到 MongoDB
        return await self._sync_to_mongodb(employee_id, api_data, db)

    async def _fetch_from_api(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """从 Java API 获取员工数据

        注意：Java API 要求 employeeId 参数为数字类型
        """
        try:
            # 转换为数字用于 API 调用
            employee_id_int = int(employee_id)

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    self.api_url,
                    params={"employeeId": employee_id_int}
                )
                response.raise_for_status()

                data = response.json()
                if data.get("status") != 200 or not data.get("success"):
                    logger.error(
                        f"API 返回错误: employee_id={employee_id}, "
                        f"status={data.get('status')}, message={data.get('message')}"
                    )
                    return None

                logger.info(f"成功获取员工数据: employee_id={employee_id}")
                return data.get("data")

        except ValueError:
            logger.error(f"无效的 employee_id 格式: {employee_id}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"HTTP 请求失败: employee_id={employee_id}, error={e}")
            return None
        except Exception as e:
            logger.exception(f"获取员工数据失败: employee_id={employee_id}")
            return None

    async def _sync_to_mongodb(self, employee_id: str, api_data: Dict[str, Any], db) -> Optional[str]:
        """同步数据到 MongoDB

        注意：保存到 MongoDB 时 employee_id 必须是字符串
        """
        try:
            # 确保 employee_id 是字符串
            employee_id = str(employee_id)

            employee_data = api_data.get("employee", {})
            setting_data = api_data.get("setting", {})

            # 构建文档（内部会转为字符串）
            employee_doc = self._build_employee_doc(employee_id, employee_data)
            setting_doc = self._build_setting_doc(employee_id, setting_data)

            # 使用字符串 employee_id 进行查询和更新
            employee_id_str = str(employee_data.get("id", employee_id))

            await db.digital_employee_configs.update_one(
                {"employee_id": employee_id_str},
                {"$set": employee_doc.model_dump()},
                upsert=True
            )

            await db.digital_employee_settings.update_one(
                {"employee_id": employee_id_str},
                {"$set": setting_doc.model_dump()},
                upsert=True
            )

            logger.info(
                f"员工数据同步成功: employee_id={employee_id_str}, "
                f"name={employee_data.get('name')}, "
                f"kb_count={len(setting_doc.knowledge_kb_ids)}"
            )
            return employee_id_str

        except Exception as e:
            logger.exception(f"同步员工数据到 MongoDB 失败: employee_id={employee_id}")
            return None

    def _build_employee_doc(self, employee_id: str, employee_data: Dict[str, Any]) -> DigitalEmployeeConfigModel:
        """构建员工配置文档

        注意：保存到 MongoDB 时 employee_id 必须是字符串
        使用 or 模式确保 None 值也能正确处理
        """
        # API 返回的 id 转为字符串用于 MongoDB 存储
        employee_id_str = str(employee_data.get("id") or employee_id)
        return DigitalEmployeeConfigModel(
            employee_id=employee_id_str,
            team_id=employee_data.get("teamId") or 0,
            name=employee_data.get("name") or "",
            position=employee_data.get("position") or "",
            employee_type=employee_data.get("type") or "FULL",
            tone=employee_data.get("tone") or "elegant",
            language=employee_data.get("language") or "mandarin",
            gender=employee_data.get("gender") or 0,
            intro=employee_data.get("intro") or "",
            portrait=employee_data.get("portrait") or "",
            model_image=employee_data.get("modelImage") or "",
            digital_code=employee_data.get("digitalCode") or "",
            onduty_status=employee_data.get("ondutyStatus") or 0,
            create_time=employee_data.get("createTime") or "",
            update_time=employee_data.get("updateTime") or "",
        )

    def _build_setting_doc(self, employee_id: str, setting_data: Dict[str, Any]) -> DigitalEmployeeConfigSettingModel:
        """构建员工设置文档"""
        knowledge = setting_data.get("knowledge") or {}
        prologue = setting_data.get("prologue") or {}
        rule = setting_data.get("rule") or {}
        chat_rule = rule.get("chatRule") or {}
        unusual_rule = rule.get("unusualRule") or {}
        # 处理 llmReply 可能为 None 的情况
        llm_reply = unusual_rule.get("llmReply") or {}
        safe_rule = rule.get("safeRule") or {}
        role = setting_data.get("role") or {}

        # 从 ragDatasets 提取 kb_ids
        kb_ids = []
        for dataset in knowledge.get("ragDatasets") or []:
            if dataset.get("isEnable") == 1:
                rag_id = dataset.get("ragDatasetId")
                if rag_id:
                    kb_ids.append(rag_id)

        return DigitalEmployeeConfigSettingModel(
            employee_id=employee_id,
            update_time=setting_data.get("updateTime") or "",
            knowledge_kb_ids=kb_ids,
            prologue_prologue=prologue.get("prologue") or "",
            prologue_is_opening_questions=prologue.get("isOpeningQuestions") or False,
            prologue_question_type=prologue.get("questionType") or 1,
            prologue_hot_questions=prologue.get("myQuestions") or [],
            chat_rule_is_multimodal=chat_rule.get("isMultiModal") or False,
            chat_rule_fixed_answer=chat_rule.get("fixedAnswer") or "",
            chat_rule_faq_sim_threshold=chat_rule.get("faqSimThreshold") or 0.0,
            chat_rule_faq_top_k=chat_rule.get("faqTopK") or 1,
            unusual_rule_exception_reply=unusual_rule.get("excepitonReply") or "",
            unusual_rule_not_match_reply_type=unusual_rule.get("notMatchReplyType"),
            unusual_rule_fixed_replys=unusual_rule.get("fixedReplys") or [],
            unusual_rule_is_web_search=llm_reply.get("isWebSearch") or False,
            unusual_rule_is_show_sign=llm_reply.get("isShowSign") or False,
            unusual_rule_is_my_prompt=llm_reply.get("isMyPrompt") or False,
            unusual_rule_my_prompt=llm_reply.get("myPrompt") or "",
            safe_rule_is_reject_answer=safe_rule.get("isRejectAnswer") or False,
            safe_rule_reject_answer=safe_rule.get("rejectAnswer") or "",
            role_persona=role.get("persona") or "",
            role_style=role.get("style") or "",
            role_style_desc=role.get("styleDesc") or "",
            plugins=self._extract_plugins(setting_data.get("plugins") or []),
            major_bank_ids=self._extract_major_banks(setting_data.get("majorWord") or {}),
        )

    def _extract_plugins(self, plugins_data: list) -> List[Dict[str, Any]]:
        """提取插件数据"""
        result = []
        for p in plugins_data:
            plugin = {
                "plugin_id": p.get("pluginId"),
                "plugin_name": p.get("pluginName"),
                "plugin_code": p.get("pluginCode"),
                "plugin_intro": p.get("pluginIntro"),
                "plugin_icon": p.get("pluginIcon"),
                "plugin_params": p.get("pluginParams"),
            }
            result.append(plugin)
        return result

    def _extract_major_banks(self, major_word_data: dict) -> List[str]:
        """提取专业词库 ID"""
        banks = major_word_data.get("majorBanks", [])
        return [str(b.get("id")) for b in banks if b.get("id") is not None]
