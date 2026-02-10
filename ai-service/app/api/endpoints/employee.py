"""
    数字员工接口服务
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from datetime import datetime

from app.models.database import DigitalEmployeeConfigModel, DigitalEmployeeConfigSettingModel
from app.models.schemas import CreateEmployeeRequest, UpdateEmployeeRequest, ResponseResult, \
    UpdateEmployeeSettingRequest
from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger
from app.services.dataset_faq_service import faq_processor
from app.services.thesaurus_major_service import thesaurus_major_processor
from app.services.thesaurus_sensitive_service import thesaurus_sensitive_processor

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

        update_data = DigitalEmployeeConfigModel(
            team_id=request.team_id,
            name=request.name,
            position=request.position,
            employee_type=request.employee_type,
            tone=request.tone,
            language=request.language,
            update_time=request.update_time,
            updated_at=datetime.now()
        )
        result = await db.digital_employee_configs.update_one(
            {'external_employee_id': employee_id},
            {"$set": update_data.model_dump()}
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
        logger.info(f"update_employee_setting request: employee_id={employee_id}")

        employee = await db.digital_employee_configs.find_one({"employee_id": employee_id})
        if not employee:
            return ResponseResult.error(status.HTTP_404_NOT_FOUND, "error",
                                        f"update_employee_setting not found: employee_id={employee_id}")

        update_data = DigitalEmployeeConfigSettingModel(
            employee_id=employee_id,
            update_time=request.update_time,
            knowledge=request.knowledge,  # 对话准备--知识库配置
            prologue=request.prologue,  # 对话开始--开场白配置
            chat_rule=request.chat_rule,  # 对话中--对话规则
            unusual_rule=request.unusual_rule,  # 对话中--异常或未匹配规则
            safe_rule=request.safe_rule,  # 对话中--安全规则配置
            role=request.role,  # 角色--人设
            plugins=request.plugins,  # 高级设置--插件
            thesaurus_major=request.thesaurus_major,  # 高级设置--专业词库
            updated_at=datetime.now()
        )
        result = await db.digital_employee_configs.update_one(
                {"employee_id": employee_id},
                {"$set": update_data.model_dump()}
            )

        if result and result.modified_count == 1:
            return ResponseResult.success(None)
        else:
            return ResponseResult.error(status.HTTP_400_BAD_REQUEST, "error",
                                        "update_employee_setting failed")
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
    employee_id: int,
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

        # 删除数字员工关联的数据：知识库（文档、FAQ，视频）
        for kb_id in employee["knowledge"]["kb_ids"]:
            # 刪除ChromaDB记录和ElasticSearch记录
            # 待补充
            await db.documents.delete_many({"kb_id": kb_id})
            await db.knowledge_bases.delete_one({'kb_id': kb_id})

        for faq_id in employee["knowledge"]["faqs"]:
            # 刪除ChromaDB记录和ElasticSearch记录
            await faq_processor.delete_faq_vectorization_data(faq_id)

            # 刪除faq任务记录
            await db.document_tasks.delete_many({"doc_id": faq_id})

            # 删除faq记录
            await db.faqs.delete_many({"faq_id": faq_id})

        for doc_id in employee["knowledge"]["video_ids"]:
            # 刪除ChromaDB记录和ElasticSearch记录
            # 待补充
            await db.documents.delete_one({'doc_id': doc_id})

        # 删除数字员工关联的数据：敏感词库
        for thesaurus_id in employee["safe_rule"]["thesaurus_sensitive"]:
            # 刪除ChromaDB记录和ElasticSearch记录
            await thesaurus_sensitive_processor.delete_thesaurus_vectorization_data(thesaurus_id)

            # 刪除敏感词库任务记录
            await db.document_tasks.delete_many({"kb_id": f"sensitive_{thesaurus_id}"})

            # 删除敏感词库记录
            await db.thesaurus_sensitive.delete_many({"thesaurus_id": thesaurus_id})

        # 删除数字员工关联的数据：专业词库
        for thesaurus_id in employee["thesaurus_major"]:
            # 刪除ChromaDB记录和ElasticSearch记录
            await thesaurus_major_processor.delete_thesaurus_vectorization_data(thesaurus_id)

            # 刪除专业词库任务记录
            await db.document_tasks.delete_many({"kb_id": f"major_{thesaurus_id}"})

            # 删除专业词库记录
            await db.thesaurus_major.delete_many({"thesaurus_id": thesaurus_id})

        result = await db.digital_employee_configs.delete_one({"employee_id": employee_id})

        if result and result.deleted_count == 1:
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


@router.get("/{employee_id}/faqs")
async def get_employee_faqs(
    employee_id: int = Path(..., description="数字员工id"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    enabled_only: bool = Query(False, description="Only return enabled FAQs"),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """获取员工的 FAQ 列表"""

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

    return {
        "code": 200,
        "message": "success",
        "data": {
            "employee_id": employee_id,
            "total": total,
            "enabled_count": enabled_count,
            "page": page,
            "page_size": page_size,
            "faqs": faqs,
        },
    }


@router.get("/list")
async def list_employees(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of employees to return"),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
        获取数字员工列表（前N个）。

        \nArgs:
            \n- limit: 返回的最大数量（默认10，最大100）
            \n- api_key: API key from auth
            \n- db: Database instance

        \nReturns:
            \n- Employee list with total count
    """
    try:
        logger.info(f"List employees request: limit={limit}")

        # 查询员工列表，按创建时间倒序
        cursor = db.digital_employee_configs.find().sort("created_at", -1).limit(limit)
        employees = await cursor.to_list(length=limit)

        # 统计总数
        total = await db.digital_employee_configs.count_documents({})

        # 格式化返回数据
        result = []
        for emp in employees:
            emp.pop("_id", None)
            emp["created_at"] = emp["created_at"].isoformat() + "Z"
            emp["updated_at"] = emp["updated_at"].isoformat() + "Z"
            if emp.get("synced_at"):
                emp["synced_at"] = emp["synced_at"].isoformat() + "Z"
            result.append(emp)

        return {
            "code": 200,
            "message": "success",
            "data": {
                "total": total,
                "count": len(result),
                "employees": result,
            }
        }

    except Exception as e:
        logger.exception(f"List employees error: limit={limit} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list employees",
        )


@router.get("/detail/{employee_id}")
async def get_employee(
    employee_id: int = Path(..., description="数字员工id"),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
        获取数字员工配置。

        \nArgs:
            \n- employee_id: 数字员工id
            \n- api_key: API key from auth
            \n- db: Database instance

        \nReturns:
            \n- Employee configuration
    """
    try:
        logger.info(f"Get employee request: employee_id={employee_id}")

        employee = await db.digital_employee_configs.find_one({"employee_id": employee_id})

        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Employee not found"
            )

        # Convert MongoDB document to dict
        employee.pop("_id", None)
        employee["created_at"] = employee["created_at"].isoformat() + "Z"
        employee["updated_at"] = employee["updated_at"].isoformat() + "Z"
        if employee.get("synced_at"):
            employee["synced_at"] = employee["synced_at"].isoformat() + "Z"

        return {"code": 200, "message": "success", "data": employee}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Get employee error: employee_id={employee_id} error={str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get employee",
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
