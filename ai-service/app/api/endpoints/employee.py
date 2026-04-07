"""
    数字员工接口服务
"""
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from datetime import datetime

from app.models.database import DigitalEmployeeConfigModel, FAQModel, ThesaurusMajorModel, ThesaurusSensitiveModel
from app.models.schemas import CreateEmployeeRequest, UpdateEmployeeRequest, ResponseResult, \
    UpdateEmployeeSettingRequest, DatasetFaqRequest, ThesaurusRequest
from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/create")
async def create_employee(
    request: CreateEmployeeRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        创建数字员工配置

        \nArgs:
            \n- request: 数字员工请求参数对象
            \n- api_key: API key from auth
            \n- db: Database instance

        \nReturns:
            \n- ResponseResult
    """
    employee_id = request.employee_id
    try:
        logger.info(f"create_employee request: employee_id={employee_id}")

        employee = await db.digital_employee_configs.find_one({"employee_id": employee_id})
        if employee:
            return ResponseResult.error(status.HTTP_409_CONFLICT, "error",
                                        f"create_employee already exists: employee_id={employee_id}")

        insert_data = DigitalEmployeeConfigModel(
            employee_id=employee_id,
            external_employee_id=employee_id,
            team_id=request.team_id,
            name=request.name,
            position=request.position,
            employee_type=request.employee_type,
            tone=request.tone,
            language=request.language,
            gender=request.gender,
            intro=request.intro,
            portrait=request.portrait,
            model_image=request.model_image,
            digital_code=request.digital_code,
            onduty_status=request.onduty_status,
            create_time=request.create_time,
            update_time=request.update_time,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        result = await db.digital_employee_configs.insert_one(insert_data.model_dump())

        if result and result.inserted_id:
            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error",
                                        "create_employee failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"create_employee exception: employee_id={employee_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="create_employee error",
        )


@router.put("/update")
async def update_employee(
    request: UpdateEmployeeRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        更新数字员工配置

        \nArgs:
            \n- request: 数字员工请求参数对象
            \n- api_key: API key from auth
            \n- db: Database instance

        \nReturns:
            \n- ResponseResult
    """
    employee_id = request.employee_id
    try:
        logger.info(f"update_employee request: employee_id={employee_id}")

        employee = await db.digital_employee_configs.find_one({"external_employee_id": employee_id})
        if not employee:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"update_employee not found: employee_id={employee_id}")

        update_data = {
            "team_id": request.team_id,
            "name": request.name,
            "position": request.position,
            "employee_type": request.employee_type,
            "tone": request.tone,
            "language": request.language,
            "update_time": request.update_time,
            "updated_at": datetime.now()
        }
        result = await db.digital_employee_configs.update_one(
            {'employee_id': employee_id},
            {"$set": update_data}
        )

        if result and result.modified_count == 1:
            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error",
                                        "update_employee failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"update_employee exception: employee_id={employee_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="update_employee error",
        )


@router.put("/setting/update")
async def update_employee_setting(
    request: UpdateEmployeeSettingRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database)
):
    """
        更新数字员工对话设定

        \nArgs:
            \n- request: 数字员工对话设定请求参数对象
            \n- api_key: API key from auth
            \n- db: Database instance

        \nReturns:
            \n- ResponseResult
    """
    employee_id = request.employee_id
    try:
        logger.info(f"update_employee_setting request: employee_id={employee_id} request={request}")

        employee = await db.digital_employee_configs.find_one({"employee_id": employee_id})
        if not employee:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"update_employee_setting not found: employee_id={employee_id}")

        if request.update_type == "knowledge":  # 对话准备--知识库配置
            if request.knowledge:
                faq_ids = []
                if request.knowledge.faqs:
                    for faq in request.knowledge.faqs:
                        faq_ids.append(f"faq_{employee_id}_{faq.faq_id}")

                        # 将还未下载的FAQ添加到任务列表中
                        await _execute_add_faq(employee_id, faq)

                update_data = {
                    "update_time": request.update_time,
                    "knowledge": {
                        "kb_ids": request.knowledge.kb_ids,
                        "faq_ids": faq_ids,
                        "video_ids": request.knowledge.video_ids
                    }
                }
            else:
                update_data = {
                    "update_time": request.update_time,
                    "knowledge": {}
                }
        elif request.update_type == "prologue":  # 对话开始--开场白配置
            faq_ids = []
            if request.prologue.prologue_faqs:
                for faq in request.prologue.prologue_faqs:
                    faq_ids.append(f"faq_{employee_id}_{faq.faq_id}")

                    # 将还未下载的FAQ添加到任务列表中
                    await _execute_add_faq(employee_id, faq)

            update_data = {
                "update_time": request.update_time,
                "prologue": {
                    "prologue": request.prologue.prologue,
                    "is_opening_questions": request.prologue.is_opening_questions,
                    "prologue_question_type": request.prologue.prologue_question_type,
                    "prologue_faqs": faq_ids,
                    "hot_questions": request.prologue.hot_questions,
                }
            }
        elif request.update_type == "rule":  # 对话中--对话规则、异常或未匹配规则、安全规则配置
            sensitive_ids = []
            if request.safe_rule:
                for thesaurus in request.safe_rule.thesaurus_sensitive:
                    for thesaurus_word in thesaurus.thesaurus_words:
                        sensitive_ids.append(f"sensitive_{employee_id}_{thesaurus.thesaurus_id}_{thesaurus_word.word_id}")

                    # 将还未下载的关联敏感词库添加到任务列表中
                    await _execute_add_sensitive(employee_id, thesaurus)

            update_data = {
                "update_time": request.update_time,
                "chat_rule": {
                    "is_multimodal": request.chat_rule.is_multimodal,
                    "fixed_answer": request.chat_rule.fixed_answer,
                    "faq_sim_threshold": request.chat_rule.faq_sim_threshold,
                    "faq_top_k": request.chat_rule.faq_top_k
                },
                "unusual_rule": {
                    "exception_reply": request.unusual_rule.exception_reply,
                    "not_match_reply_type": request.unusual_rule.not_match_reply_type,
                    "fixed_replys": request.unusual_rule.fixed_replys,
                    "is_web_search": request.unusual_rule.is_web_search,
                    "is_show_sign": request.unusual_rule.is_show_sign,
                    "is_my_prompt": request.unusual_rule.is_my_prompt,
                    "my_prompt": request.unusual_rule.my_prompt
                },
                "safe_rule": {
                    "is_reject_answer": request.safe_rule.is_reject_answer,
                    "reject_answer": request.safe_rule.reject_answer,
                    "sensitive_ids": sensitive_ids
                }
            }
        elif request.update_type == "role":  # 角色--人设
            update_data = {
                "update_time": request.update_time,
                "role": {
                    "persona": request.role.persona,
                    "style": request.role.style,
                    "style_desc": request.role.style_desc
                }
            }
        elif request.update_type == "plugins":  # 高级设置--插件
            if request.plugins:
                plugins = []
                for plugin_tmp in request.plugins:
                    plugin = {
                        "plugin_id": plugin_tmp.plugin_id,
                        "plugin_name": plugin_tmp.plugin_name,
                        "plugin_code": plugin_tmp.plugin_code,
                        "plugin_intro": plugin_tmp.plugin_intro,
                        "plugin_icon": plugin_tmp.plugin_icon,
                        "plugin_params": plugin_tmp.plugin_params
                    }
                    plugins.append(plugin)
                update_data = {
                    "update_time": request.update_time,
                    "plugins": plugins
                }
            else:
                update_data = {
                    "update_time": request.update_time,
                    "plugins": []
                }
        elif request.update_type == "thesaurus_major":  # 高级设置--专业词库配置
            major_ids = []
            for thesaurus in request.thesaurus_major.thesaurus_major:
                for thesaurus_word in thesaurus.thesaurus_words:
                    major_ids.append(f"major_{employee_id}_{thesaurus.thesaurus_id}_{thesaurus_word.word_id}")

                # 将还未下载的关联专业词库添加到任务列表中
                await _execute_add_major(employee_id, thesaurus)

            update_data = {
                "update_time": request.update_time,
                "thesaurus_major": {
                    "is_synonym_rewrite": request.thesaurus_major.is_synonym_rewrite,
                    "major_ids": major_ids
                }
            }
        else:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"update_employee_setting update_type is empty: employee_id={employee_id}")

        result = await db.digital_employee_configs.update_one(
            {"employee_id": employee_id},
            {"$set": update_data}
        )

        if result and result.modified_count == 1:
            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error", "update_employee_setting failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"update_employee_setting exception: employee_id={employee_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="update_employee_setting error",
        )


@router.delete("/delete/{employee_id}")
async def delete_employee(
    employee_id: str,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
        删除数字员工配置。

        \nArgs:
            \n- employee_id: 数字员工id
            \n- api_key: API key from auth
            \n- db: Database instance

        \nReturns:
            \n- Deletion result
    """
    try:
        logger.info(f"delete_employee request: employee_id={employee_id}")

        employee = await db.digital_employee_configs.find_one({"employee_id": employee_id})
        if not employee:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"delete_employee not found: employee_id={employee_id}")

        # 删除数字员工关联的数据：敏感词库
        # if employee["safe_rule"] and employee["safe_rule"]["thesaurus_sensitive"]:
        #     for thesaurus_id in employee["safe_rule"]["thesaurus_sensitive"]:
        #         # 刪除ChromaDB记录和ElasticSearch记录
        #         await thesaurus_sensitive_processor.delete_thesaurus_vectorization_data(thesaurus_id)

        #         # 刪除敏感词库任务记录
        #         await db.document_tasks.delete_many({"kb_id": f"sensitive_{thesaurus_id}"})

        #         # 删除敏感词库记录
        #         await db.thesaurus_sensitive.delete_many({"thesaurus_id": thesaurus_id})

        # 删除数字员工关联的数据：专业词库
        # if employee["thesaurus_major"]:
        #     for thesaurus_id in employee["thesaurus_major"]:
        #         # 刪除ChromaDB记录和ElasticSearch记录
        #         await thesaurus_major_processor.delete_thesaurus_vectorization_data(thesaurus_id)

        #         # 刪除专业词库任务记录
        #         await db.document_tasks.delete_many({"kb_id": f"major_{thesaurus_id}"})

        #         # 删除专业词库记录
        #         await db.thesaurus_major.delete_many({"thesaurus_id": thesaurus_id})

        result = await db.digital_employee_configs.delete_one({"employee_id": employee_id})

        if result and result.deleted_count == 1:
            logger.info(f"delete_employee success: employee_id={employee_id}")
            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error",
                                        "delete_employee failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"delete_employee exception: employee_id={employee_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="delete_employee error",
        )


@router.get("/detail/{employee_id}")
async def get_employee(
    employee_id: str = Path(..., description="数字员工id"),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
    获取数字员工配置（合并核心信息和设置）。

    Args:
        - employee_id: 数字员工id
        - api_key: API key from auth
        - db: Database instance

    Returns:
        - Employee configuration (merged from configs and settings collections)
    """
    try:
        logger.info(f"get_employee request: employee_id={employee_id}")

        # 查询员工核心信息
        employee = await db.digital_employee_configs.find_one({"employee_id": employee_id})

        if not employee:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"get_employee not found employee_id={employee_id}")

        # 查询员工设置信息
        setting = await db.digital_employee_settings.find_one({"employee_id": employee_id})

        # 格式化并合并数据
        employee.pop("_id", None)
        employee["created_at"] = employee["created_at"].isoformat() + "Z"
        employee["updated_at"] = employee["updated_at"].isoformat() + "Z"
        if employee.get("synced_at"):
            employee["synced_at"] = employee["synced_at"].isoformat() + "Z"

        # 合并设置信息
        if setting:
            setting.pop("_id", None)
            if setting.get("updated_at"):
                setting["updated_at"] = setting["updated_at"].isoformat() + "Z"
            employee["setting"] = setting
        else:
            employee["setting"] = {}

        return ResponseResult.success(employee)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"get_employee exception: employee_id={employee_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="get_employee error",
        )


@router.get("/{employee_id}/faqs")
async def get_employee_faqs(
    employee_id: str = Path(..., description="数字员工id"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    enabled_only: bool = Query(False, description="Only return enabled FAQs"),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """获取员工的 FAQ 列表"""
    logger.info(f"get_employee_faqs request: employee_id={employee_id}")

    # 构建查询条件
    query = {"employee_id": employee_id}
    if enabled_only:
        query["is_enable"] = 1

    # 分页查询
    skip = (page - 1) * page_size
    faqs_cursor = db.faqs.find(query).skip(skip).limit(page_size)
    faqs = await faqs_cursor.to_list(length=page_size)

    # 统计总数
    total = await db.faqs.count_documents(query)
    enabled_count = await db.faqs.count_documents(
        {"employee_id": employee_id, "is_enable": 1}
    )

    # 清理数据
    for faq in faqs:
        faq.pop("_id", None)

    data = {
            "employee_id": employee_id,
            "total": total,
            "enabled_count": enabled_count,
            "page": page,
            "page_size": page_size,
            "faqs": faqs,
        }
    return ResponseResult.success(data)


@router.get("/list")
async def list_employees(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of employees to return"),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
    获取数字员工列表（前N个，包含核心信息和设置）。

    Args:
        - limit: 返回的最大数量（默认10，最大100）
        - api_key: API key from auth
        - db: Database instance

    Returns:
        - Employee list with total count
    """
    try:
        logger.info(f"list_employees request: limit={limit}")

        # 查询员工列表，按创建时间倒序
        cursor = db.digital_employee_configs.find().sort("created_at", -1).limit(limit)
        employees = await cursor.to_list(length=limit)

        # 统计总数
        total = await db.digital_employee_configs.count_documents({})

        # 获取所有员工ID，批量查询设置
        employee_ids = [emp["employee_id"] for emp in employees]
        settings_cursor = db.digital_employee_settings.find({"employee_id": {"$in": employee_ids}})
        settings_list = await settings_cursor.to_list(length=len(employee_ids))
        settings_dict = {s["employee_id"]: s for s in settings_list}

        # 格式化返回数据
        result = []
        for emp in employees:
            emp.pop("_id", None)
            emp["created_at"] = emp["created_at"].isoformat() + "Z"
            emp["updated_at"] = emp["updated_at"].isoformat() + "Z"
            if emp.get("synced_at"):
                emp["synced_at"] = emp["synced_at"].isoformat() + "Z"

            # 合并设置信息
            setting = settings_dict.get(emp["employee_id"])
            if setting:
                setting.pop("_id", None)
                if setting.get("updated_at"):
                    setting["updated_at"] = setting["updated_at"].isoformat() + "Z"
                emp["setting"] = setting
            else:
                emp["setting"] = {}

            result.append(emp)

        data = {
            "total": total,
            "count": len(result),
            "employees": result,
        }
        return ResponseResult.success(data)

    except Exception as e:
        logger.exception(f"list_employees exception: limit={limit} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="list_employees error",
        )


@router.post("/create/test-financial-analyst")
async def create_test_financial_analyst(
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
        创建测试用的金融分析师数字员工（基于期刊文件主题）。

        \nReturns:
            \n- Created employee configuration
    """
    try:
        logger.info("Create test financial analyst employee")

        employee_id = "financial_analyst"

        # Check if employee already exists
        existing = await db.digital_employee_configs.find_one(
            {"employee_id": employee_id}
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Employee {employee_id} already exists",
            )

        # Prepare employee document with test data based on journal files
        now = datetime.now()
        employee_doc = {
            "employee_id": 1,
            "external_employee_id": employee_id,
            "name": "金融分析师小智",
            "domain": "金融经济",
            "role": "金融研究分析师",
            "description": "专业的金融与经济研究数字员工，专注于数字金融、数字货币、普惠金融、经济转型等领域的研究与分析",
            "personality": {
                "tone": "professional",
                "style": "friendly",
                "language": "zh-CN",
                "formality": "moderate",
            },
            "capabilities": {
                "kb_ids": ["kb_digital_finance", "kb_monetary_policy", "kb_fintech"],
                "web_search_enabled": True,
                "max_context_turns": 10,
                "multimodal_enabled": False,
            },
            "greeting": "您好！我是金融分析师小智，专注于数字金融、货币政策与经济转型研究。我可以帮您解答关于数字人民币、普惠金融、金融监管、数字化转型等方面的问题。有什么可以帮到您的吗？",
            "hot_questions": [
                "什么是数字人民币？它有哪些特点？",
                "数字金融如何赋能共同富裕？",
                "普惠金融在乡村振兴中有什么作用？",
                "金融科技面临哪些监管挑战？",
                "数字化转型对企业有什么影响？",
                "如何防范二维码支付的洗钱风险？",
                "数字央行建设的背景和意义是什么？",
            ],
            "personalization": {
                "user_profiling_enabled": True,
                "personalized_recommendations": True,
                "adaptive_tone": True,
            },
            "created_at": now,
            "updated_at": now,
            "synced_at": None,
        }

        # Insert into database
        await db.digital_employee_configs.insert_one(employee_doc)

        # Prepare response data (create new dict to avoid ObjectId serialization issues)
        response_data = {
            "employee_id": 1,
            "name": "金融分析师小智",
            "domain": "金融经济",
            "role": "金融研究分析师",
            "description": "专业的金融与经济研究数字员工，专注于数字金融、数字货币、普惠金融、经济转型等领域的研究与分析",
            "personality": {
                "tone": "professional",
                "style": "friendly",
                "language": "zh-CN",
                "formality": "moderate",
            },
            "capabilities": {
                "kb_ids": ["kb_digital_finance", "kb_monetary_policy", "kb_fintech"],
                "web_search_enabled": True,
                "max_context_turns": 10,
                "multimodal_enabled": False,
            },
            "greeting": "您好！我是金融分析师小智，专注于数字金融、货币政策与经济转型研究。我可以帮您解答关于数字人民币、普惠金融、金融监管、数字化转型等方面的问题。有什么可以帮到您的吗？",
            "hot_questions": [
                "什么是数字人民币？它有哪些特点？",
                "数字金融如何赋能共同富裕？",
                "普惠金融在乡村振兴中有什么作用？",
                "金融科技面临哪些监管挑战？",
                "数字化转型对企业有什么影响？",
                "如何防范二维码支付的洗钱风险？",
                "数字央行建设的背景和意义是什么？",
            ],
            "personalization": {
                "user_profiling_enabled": True,
                "personalized_recommendations": True,
                "adaptive_tone": True,
            },
            "created_at": now.isoformat() + "Z",
            "updated_at": now.isoformat() + "Z",
            "synced_at": None,
        }

        return {"code": 200, "message": "success", "data": response_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Create test financial analyst error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create test financial analyst",
        )


async def _execute_add_faq(employee_id: str, faq: DatasetFaqRequest):
    db = await get_database()
    update_faq_id = f"faq_{employee_id}_{faq.faq_id}"

    # 不存在则插入新记录
    result = await db.faqs.find_one({"faq_id": update_faq_id})
    if not result:
        # Build combined_text for embedding (question + similar questions)
        combined_text = faq.question_name
        if faq.similar_questions:
            combined_text += " " + " ".join(faq.similar_questions)
        try:
            insert_data = FAQModel(
                faq_id=update_faq_id,
                employee_id=employee_id,
                external_faq_id=faq.faq_id,
                question_name=faq.question_name,
                start_time=faq.start_time,
                end_time=faq.end_time,
                is_enable=faq.is_enable,
                is_clear=faq.is_clear,
                similar_questions=faq.similar_questions,
                answers=faq.answers,
                update_time=faq.update_time,
                combined_text=combined_text,
                keywords=faq.similar_questions,  # Use similar questions as keywords
                vector_id=update_faq_id,  # Use faq_id as vector_id
                es_indexed=False,  # Will be set to True after ES indexing
            ).model_dump()
            # await db.faqs.insert_one(insert_data)

            # await task_processor.submit_faq_vectorization_task(faq_id=update_faq_id, kb_id=f"faq_{faq.faq_id}")
        except Exception as e:
            logger.error(f"_execute_add_faq failed: faq_id={update_faq_id} error={str(e)}", exc_info=True)


async def _execute_add_sensitive(employee_id: str, thesaurus: ThesaurusRequest):
    db = await get_database()
    thesaurus_id = thesaurus.thesaurus_id

    for thesaurus_word in thesaurus.thesaurus_words:
        update_thesaurus_id = f"sensitive_{employee_id}_{thesaurus_id}_{thesaurus_word.word_id}"
        try:
            result = await db.thesaurus_sensitive.find_one({"thesaurus_id": update_thesaurus_id})
            if not result:
                # Build combined_text for embedding (thesaurus_name + word_name)
                combined_text = thesaurus.thesaurus_name + " " + thesaurus_word.word_name

                update_data = ThesaurusSensitiveModel(
                    thesaurus_id=update_thesaurus_id,
                    employee_id=employee_id,
                    external_thesaurus_id=thesaurus_id,
                    external_word_id=thesaurus_word.word_id,
                    thesaurus_name=thesaurus.thesaurus_name,
                    is_enable=thesaurus.is_enable,
                    update_time=thesaurus.update_time,
                    word_name=thesaurus_word.word_name,
                    combined_text=combined_text,
                    keywords=[thesaurus_word.word_name],
                    vector_id=update_thesaurus_id,  # Will be set after vectorization
                    es_indexed=False,  # Will be set after ElasticSearch indexing
                ).model_dump()
                await db.thesaurus_sensitive.insert_one(update_data)

                # await task_processor.submit_thesaurus_sensitive_vectorization_task(thesaurus_id=update_thesaurus_id,
                #                                                                    kb_id=f"sensitive_{thesaurus_id}")
        except Exception as e:
            logger.error(f"_execute_add_sensitive failed: thesaurus_id={thesaurus_id} error={str(e)}", exc_info=True)


async def _execute_add_major(employee_id: str, thesaurus: ThesaurusRequest):
    db = await get_database()
    thesaurus_id = thesaurus.thesaurus_id

    for thesaurus_word in thesaurus.thesaurus_words:
        update_thesaurus_id = f"major_{employee_id}_{thesaurus_id}_{thesaurus_word.word_id}"
        try:
            result = await db.thesaurus_major.find_one({"thesaurus_id": update_thesaurus_id})
            if not result:
                # Build combined_text for embedding (thesaurus_name + word_name + similar_word_name)
                combined_text = thesaurus.thesaurus_name + " " + thesaurus_word.word_name
                if thesaurus_word.similar_words:
                    combined_text += " " + " ".join(thesaurus_word.similar_words)

                update_data = ThesaurusMajorModel(
                    thesaurus_id=update_thesaurus_id,
                    employee_id=employee_id,
                    external_thesaurus_id=thesaurus_id,
                    external_word_id=thesaurus_word.word_id,
                    thesaurus_name=thesaurus.thesaurus_name,
                    is_enable=thesaurus.is_enable,
                    update_time=thesaurus.update_time,
                    word_name=thesaurus_word.word_name,
                    similar_words=thesaurus_word.similar_words,
                    combined_text=combined_text,
                    keywords=thesaurus_word.similar_words,
                    vector_id=update_thesaurus_id,  # Will be set after vectorization
                    es_indexed=False,  # Will be set after ElasticSearch indexing
                ).model_dump()
                # await db.thesaurus_major.insert_one(update_data)

                # await task_processor.submit_thesaurus_major_vectorization_task(thesaurus_id=update_thesaurus_id,
                #                                                                kb_id=f"major_{thesaurus_id}")
        except Exception as e:
            logger.error(f"_execute_add_major failed: thesaurus_id={thesaurus_id} error={str(e)}", exc_info=True)
