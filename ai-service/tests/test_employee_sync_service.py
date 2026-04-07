"""
员工同步服务单元测试
使用 employee_id=29 和 employee_id=43 作为测试数据
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.employee_sync_service import EmployeeSyncService


# 测试数据 - 模拟 get.json 中 employee_id=29 的响应
MOCK_API_RESPONSE_29 = {
    "status": 200,
    "message": "ok",
    "success": True,
    "data": {
        "employee": {
            "id": 29,
            "teamId": 4,
            "name": "陈晓燕",
            "position": "校园助教",
            "type": "FULL",
            "tone": "elegant",
            "language": "mandarin",
            "gender": 2,
            "intro": "班级互动",
            "portrait": "/edu-api/fileserver/default/image/2026/2/4/47499d83.png",
            "modelImage": "/edu-api/fileserver/default/image/2026/2/5/a80f66bc.png",
            "digitalCode": "yunzhihui",
            "ondutyStatus": 1,
            "createTime": "2026-01-20 16:59:33",
            "updateTime": "2026-01-20 16:59:33",
        },
        "setting": {
            "knowledge": {
                "ragDatasets": [
                    {
                        "id": 55,
                        "ragDatasetId": "kb_9abcbe4aa557",
                        "name": "数学学科-zj",
                        "isEnable": 1
                    }
                ]
            },
            "prologue": {
                "prologue": "",
                "isOpeningQuestions": True,
                "questionType": 1,
                "myQuestions": []
            },
            "rule": {
                "id": 16,
                "chatRule": {
                    "isMultiModal": True,
                    "fixedAnswer": "无法识别图片内容，请用文字描述问题",
                    "faqSimThreshold": 0.5,
                    "faqTopK": 1
                },
                "unusualRule": {
                    "excepitonReply": "系统忙不过来了，请稍后重试",
                    "notMatchReplyType": 1,
                    "fixedReplys": [],
                    "llmReply": {
                        "isWebSearch": False,
                        "isShowSign": False,
                        "isMyPrompt": False,
                        "myPrompt": "自定义提示词"
                    }
                },
                "safeRule": {
                    "isRejectAnswer": True,
                    "rejectAnswer": "无法回答你的问题，请换个问题试试"
                }
            },
            "role": {
                "persona": "你是一个资深客服，懂得使用话术拉进与客户距离",
                "styleId": 1,
                "style": "通用",
                "styleTeamId": 0,
                "styleDesc": "不刻意运笑，而是通过独特的视角和逻辑重构世界"
            },
            "plugins": [],
            "majorWord": {
                "majorBanks": []
            }
        }
    }
}


# 测试数据 - 模拟 employee_id=43 的响应
MOCK_API_RESPONSE_43 = {
    "status": 200,
    "message": "ok",
    "success": True,
    "data": {
        "employee": {
            "id": 43,
            "teamId": 4,
            "name": "测试员工43",
            "position": "测试职位",
            "type": "HALF",
            "tone": "professional",
            "language": "mandarin",
            "gender": 1,
            "intro": "测试简介",
            "portrait": None,
            "modelImage": None,
            "digitalCode": "test43",
            "ondutyStatus": 1,
            "createTime": "2026-01-20 16:59:33",
            "updateTime": "2026-01-20 16:59:33",
        },
        "setting": {
            "knowledge": {
                "ragDatasets": [
                    {
                        "id": 56,
                        "ragDatasetId": "kb_test123",
                        "name": "测试知识库",
                        "isEnable": 1
                    }
                ]
            },
            "prologue": {
                "prologue": "你好，我是测试员工",
                "isOpeningQuestions": False,
                "questionType": 2,
                "myQuestions": ["问题1", "问题2"]
            },
            "rule": {
                "id": 17,
                "chatRule": {
                    "isMultiModal": False,
                    "fixedAnswer": None,
                    "faqSimThreshold": 0.6,
                    "faqTopK": 3
                },
                "unusualRule": {
                    "excepitonReply": "系统异常",
                    "notMatchReplyType": 0,
                    "fixedReplys": ["固定回复1", "固定回复2"],
                    "llmReply": None
                },
                "safeRule": {
                    "isRejectAnswer": False,
                    "rejectAnswer": None
                }
            },
            "role": {
                "persona": "你是一个专业的测试助手",
                "styleId": 2,
                "style": "专业",
                "styleTeamId": 0,
                "styleDesc": "专业风格"
            },
            "plugins": [
                {
                    "pluginId": 1,
                    "pluginName": "测试插件",
                    "pluginCode": "test_plugin",
                    "pluginIntro": "这是一个测试插件",
                    "pluginIcon": "icon.png",
                    "pluginParams": "{}"
                }
            ],
            "majorWord": {
                "majorBanks": [
                    {"id": 101},
                    {"id": 102}
                ]
            }
        }
    }
}


@pytest.fixture
def service():
    """创建同步服务实例"""
    return EmployeeSyncService()


@pytest.fixture
def mock_db():
    """创建模拟数据库"""
    db = Mock()

    # 模拟 digital_employee_configs 集合
    db.digital_employee_configs = Mock()
    db.digital_employee_configs.update_one = AsyncMock(return_value=Mock(upserted_id=1, modified_count=1))

    # 模拟 digital_employee_settings 集合
    db.digital_employee_settings = Mock()
    db.digital_employee_settings.update_one = AsyncMock(return_value=Mock(upserted_id=1, modified_count=1))

    return db


@pytest.mark.asyncio
async def test_sync_employee_29(service, mock_db):
    """测试同步 employee_id=29"""
    with patch("httpx.AsyncClient.get") as mock_get:
        # 设置 mock 响应
        mock_response = Mock()
        mock_response.json.return_value = MOCK_API_RESPONSE_29
        mock_response.raise_for_status = Mock()
        mock_get.return_value.__aenter__.return_value = mock_response

        # 执行同步
        result = await service.fetch_and_sync("29", mock_db)

        # 验证返回的是字符串
        assert result == "29"

        # 验证 API 调用时使用的是数字参数
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]["params"]["employeeId"] == 29  # 应该是数字，不是字符串

        # 验证 MongoDB 更新被调用
        assert mock_db.digital_employee_configs.update_one.called
        assert mock_db.digital_employee_settings.update_one.called

        # 验证 update_one 的参数
        config_call = mock_db.digital_employee_configs.update_one.call_args
        assert config_call[0][0]["employee_id"] == "29"  # MongoDB 查询使用字符串

        setting_call = mock_db.digital_employee_settings.update_one.call_args
        assert setting_call[0][0]["employee_id"] == "29"

        # 验证员工数据
        employee_doc = config_call[0][1]["$set"]
        assert employee_doc["name"] == "陈晓燕"
        assert employee_doc["position"] == "校园助教"
        assert employee_doc["employee_type"] == "FULL"
        assert employee_doc["gender"] == 2

        # 验证设置数据
        setting_doc = setting_call[0][1]["$set"]
        assert "kb_9abcbe4aa557" in setting_doc["knowledge_kb_ids"]
        assert setting_doc["chat_rule_is_multimodal"] is True
        assert setting_doc["safe_rule_is_reject_answer"] is True
        assert setting_doc["role_persona"] == "你是一个资深客服，懂得使用话术拉进与客户距离"


@pytest.mark.asyncio
async def test_sync_employee_43(service, mock_db):
    """测试同步 employee_id=43"""
    with patch("httpx.AsyncClient.get") as mock_get:
        # 设置 mock 响应
        mock_response = Mock()
        mock_response.json.return_value = MOCK_API_RESPONSE_43
        mock_response.raise_for_status = Mock()
        mock_get.return_value.__aenter__.return_value = mock_response

        # 执行同步
        result = await service.fetch_and_sync("43", mock_db)

        # 验证返回的是字符串
        assert result == "43"

        # 验证 API 调用时使用的是数字参数
        call_args = mock_get.call_args
        assert call_args[1]["params"]["employeeId"] == 43

        # 验证员工数据
        config_call = mock_db.digital_employee_configs.update_one.call_args
        employee_doc = config_call[0][1]["$set"]
        assert employee_doc["name"] == "测试员工43"
        assert employee_doc["employee_type"] == "HALF"
        assert employee_doc["gender"] == 1

        # 验证设置数据
        setting_call = mock_db.digital_employee_settings.update_one.call_args
        setting_doc = setting_call[0][1]["$set"]
        assert "kb_test123" in setting_doc["knowledge_kb_ids"]
        assert setting_doc["prologue_hot_questions"] == ["问题1", "问题2"]
        assert setting_doc["chat_rule_is_multimodal"] is False
        assert setting_doc["unusual_rule_not_match_reply_type"] == 0
        assert setting_doc["major_bank_ids"] == ["101", "102"]
        assert len(setting_doc["plugins"]) == 1
        assert setting_doc["plugins"][0]["plugin_name"] == "测试插件"


@pytest.mark.asyncio
async def test_fetch_api_failure(service, mock_db):
    """测试 API 调用失败的情况"""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = Exception("Network error")

        result = await service.fetch_and_sync("29", mock_db)

        assert result is None

        # 验证没有调用 MongoDB 更新
        assert not mock_db.digital_employee_configs.update_one.called
        assert not mock_db.digital_employee_settings.update_one.called


@pytest.mark.asyncio
async def test_api_returns_error(service, mock_db):
    """测试 API 返回错误状态"""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": 400,
            "message": "Bad Request",
            "success": False
        }
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await service.fetch_and_sync("29", mock_db)

        assert result is None


@pytest.mark.asyncio
async def test_extract_plugins(service):
    """测试插件数据提取"""
    plugins_data = [
        {
            "pluginId": 1,
            "pluginName": "测试插件1",
            "pluginCode": "code1",
            "pluginIntro": "简介1",
            "pluginIcon": "icon1.png",
            "pluginParams": "params1"
        },
        {
            "pluginId": 2,
            "pluginName": "测试插件2",
        }
    ]

    result = service._extract_plugins(plugins_data)

    assert len(result) == 2
    assert result[0]["plugin_id"] == 1
    assert result[0]["plugin_name"] == "测试插件1"
    assert result[0]["plugin_code"] == "code1"
    assert result[1]["plugin_id"] == 2
    assert result[1]["plugin_name"] == "测试插件2"


@pytest.mark.asyncio
async def test_extract_major_banks(service):
    """测试专业词库 ID 提取"""
    major_word_data = {
        "majorBanks": [
            {"id": 101},
            {"id": 102},
            {"id": None},
            {}
        ]
    }

    result = service._extract_major_banks(major_word_data)

    assert result == ["101", "102"]
    assert len(result) == 2  # None 和空 id 应该被过滤


@pytest.mark.asyncio
async def test_employee_id_type_conversion(service, mock_db):
    """测试 employee_id 类型转换"""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = MOCK_API_RESPONSE_29
        mock_response.raise_for_status = Mock()
        mock_get.return_value.__aenter__.return_value = mock_response

        # 使用数字类型调用
        result = await service.fetch_and_sync(29, mock_db)

        assert result == "29"  # 返回的应该是字符串

        # 验证 API 调用参数是数字
        call_args = mock_get.call_args
        assert call_args[1]["params"]["employeeId"] == 29
        assert isinstance(call_args[1]["params"]["employeeId"], int)
