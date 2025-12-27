"""
生成 hutao 测试数据文件的脚本

运行方式：
cd D:\zenking_work\metahuman_work\ZengKingMorphe
D:/zenking_work/metahuman_work/ZengKingMorphe/.venv/Scripts/python.exe ai-service/scripts/generate_hutao_testdata.py
"""
import json
from pathlib import Path

# 读取首饰雕蜡工艺 FAQs
faq_source_file = Path(__file__).parent.parent.parent / "outer_api_docs" / "首饰雕蜡工艺-faqs.json"
print(f"📖 Reading FAQ source file: {faq_source_file}")

with open(faq_source_file, "r", encoding="utf-8") as f:
    source_faqs = json.load(f)

print(f"✅ Loaded {len(source_faqs)} FAQs from source file")

# 转换为标准 FAQ 格式
faqs = []
for idx, item in enumerate(source_faqs, start=1):
    faq = {
        "id": f"faq_hutao_{idx:03d}",
        "teamId": 1001,
        "questionName": item["questionName"],
        "startTime": None,
        "endTime": None,
        "isEnable": 1,
        "isClear": 0,
        "createUserId": 1,
        "updateTime": "2025-12-27T10:00:00.000Z",
        "createTime": "2025-12-20T08:00:00.000Z",
        "similarQuestions": [],
        "answers": [item["answer"]]
    }
    faqs.append(faq)

print(f"🔄 Converted {len(faqs)} FAQs to standard format")

# 构建完整的员工数据
employee_data = {
    "status": 0,
    "message": "success",
    "data": {
        "employee": {
            "id": "hutao_001",
            "teamId": 1001,
            "name": "胡桃",
            "position": "课程助教",
            "type": "AVATAR",
            "tone": "温和亲切",
            "language": "zh-CN",
            "createUserId": 1,
            "ondutyStatus": 1,
            "updateTime": "2025-12-27T10:00:00.000Z",
            "createTime": "2025-12-20T08:00:00.000Z",
            "gender": 1,
            "languages": [
                {
                    "language": "zh-CN",
                    "remark": "中文"
                }
            ],
            "tones": [
                {
                    "gender": 1,
                    "tone": "温和亲切",
                    "remark": "适合教学场景"
                }
            ],
            "intro": "我是胡桃，一名专注于首饰雕蜡工艺教学的课程助教。我会帮助你理解雕蜡工艺的各个知识点，解答学习过程中遇到的问题。无论是工具使用、技法细节，还是铸造流程，我都会耐心为你讲解。",
            "portrait": "https://example.com/avatars/hutao.jpg",
            "modelImage": "https://example.com/models/hutao_3d.glb",
            "digitalCode": "HUTAO_EDU_001"
        },
        "setting": {
            "knowledge": {
                "ragDatasets": [
                    {
                        "id": "kb_jewelry_wax_001",
                        "teamId": 1001,
                        "ragDatasetId": "kb_jewelry_wax_001",
                        "name": "首饰雕蜡工艺知识库",
                        "isEnable": 1,
                        "isPublish": 1,
                        "createUserId": 1,
                        "flag": 1,
                        "createTime": "2025-12-20T08:00:00.000Z"
                    }
                ],
                "faqs": faqs,
                "medias": [],
                "questionBanks": {}
            },
            "prologue": {
                "prologue": "你好！我是胡桃，你的首饰雕蜡工艺课程助教。很高兴能帮助你学习雕蜡技法。",
                "isOpeningQuestions": True,
                "questionType": 0,
                "faqs": [],
                "myQuestions": [
                    "雕蜡工艺中常用的测量工具有哪些？",
                    "失蜡铸造的基本步骤是什么？",
                    "锉刀有哪些类型？",
                    "素圈戒指的雕蜡方法有哪几种？",
                    "什么是包镶和爪镶？"
                ]
            },
            "rule": {
                "id": "rule_hutao_001",
                "chatRule": {
                    "isMultimodal": False,
                    "faqSimThreshold": 0.85,
                    "faqTopK": 3
                },
                "unusualRule": {
                    "excepitonReply": "抱歉，我在理解你的问题时遇到了一些困难。",
                    "notMatchReplyType": 1,
                    "fixedReplys": [
                        "抱歉，我目前只能回答首饰雕蜡工艺相关的问题哦。"
                    ],
                    "llmReply": {
                        "isWebSearch": True,
                        "isShowSign": False,
                        "isMyPrompt": True,
                        "myPrompt": "你是一名首饰雕蜡工艺的专业课程助教，专注于教学和答疑。回答要专业、准确、易懂，适当使用教学案例。"
                    }
                },
                "safeRule": {
                    "isRejectAnswer": True,
                    "rejectAnswer": "抱歉，这个问题不在我的回答范围内。",
                    "sensitiveBankIds": [],
                    "sensitiveBanks": []
                }
            },
            "role": {
                "persona": "温和、耐心、专业的教学助手",
                "style": "亲切友好",
                "styleDesc": "用通俗易懂的语言讲解专业知识，适当使用比喻和例子帮助理解"
            },
            "plugins": [],
            "majorWord": {
                "isSynonymRewrite": False,
                "majorBanks": []
            }
        }
    },
    "success": True,
    "error": None
}

# 保存到文件
output_file = Path(__file__).parent.parent.parent / "outer_api_docs" / "按员工id返回的数据-hutao.json"
print(f"💾 Writing to output file: {output_file}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(employee_data, f, ensure_ascii=False, indent=2)

print(f"✅ 成功生成测试数据文件: {output_file}")
print(f"📊 包含 {len(faqs)} 条 FAQ 数据")
print(f"👤 员工信息: {employee_data['data']['employee']['name']} ({employee_data['data']['employee']['position']})")
print(f"🎭 性别: {'男生' if employee_data['data']['employee']['gender'] == 1 else '女生'}")
print(f"📚 知识库: {employee_data['data']['setting']['knowledge']['ragDatasets'][0]['name']}")
print("\n🎉 测试数据生成完成！可以在 session.py 中使用 employee_id='hutao' 进行测试")
