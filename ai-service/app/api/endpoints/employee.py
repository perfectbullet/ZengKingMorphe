"""
Digital Employee management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, Body
from datetime import datetime

from app.models.schemas import CreateEmployeeRequest, UpdateEmployeeRequest
from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


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
        logger.info("List employees request", limit=limit)

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
        logger.exception(f"List employees error: limit={limit}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list employees",
        )


@router.get("/detail/{employee_id}")
async def get_employee(
    employee_id: str = Path(..., description="Employee ID"),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
    获取数字员工配置。

    \nArgs:
        \n- employee_id: Employee ID
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
        logger.exception(f"Get employee error: employee_id={employee_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get employee",
        )


@router.post("/create")
async def create_employee(
    request: CreateEmployeeRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
    创建数字员工配置。

    \nArgs:
        \n- request: CreateEmployeeRequest with employee configuration
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- Created employee configuration
    """
    try:
        logger.info("Create employee request", employee_id=request.employee_id)

        # Check if employee already exists
        existing = await db.digital_employee_configs.find_one(
            {"employee_id": request.employee_id}
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Employee {request.employee_id} already exists",
            )

        # Prepare employee document
        now = datetime.utcnow()
        employee_doc = {
            "employee_id": request.employee_id,
            "name": request.name,
            "domain": request.domain,
            "role": request.role,
            "description": request.description,
            "personality": request.personality.model_dump(),
            "capabilities": request.capabilities.model_dump(),
            "greeting": request.greeting,
            "hot_questions": request.hot_questions,
            "personalization": request.personalization.model_dump(),
            "created_at": now,
            "updated_at": now,
            "synced_at": None,
        }

        # Insert into database
        await db.digital_employee_configs.insert_one(employee_doc)

        # Return created employee
        employee_doc["created_at"] = employee_doc["created_at"].isoformat() + "Z"
        employee_doc["updated_at"] = employee_doc["updated_at"].isoformat() + "Z"

        return {"code": 200, "message": "success", "data": employee_doc}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Create employee error: employee_id={request.employee_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create employee",
        )


@router.get("/{employee_id}/faqs")
async def get_employee_faqs(
    employee_id: str = Path(..., description="Employee ID"),
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
        now = datetime.utcnow()
        employee_doc = {
            "employee_id": employee_id,
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
            "employee_id": employee_id,
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
        logger.exception("Create test financial analyst error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create test financial analyst",
        )


@router.delete("/delete/{employee_id}")
async def delete_employee(
    employee_id: str = Path(..., description="Employee ID"),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
    删除数字员工配置。

    \nArgs:
        \n- employee_id: Employee ID
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- Deletion result
    """
    try:
        logger.info("Delete employee request", employee_id=employee_id)

        # Check if employee exists
        existing = await db.digital_employee_configs.find_one(
            {"employee_id": employee_id}
        )
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Employee {employee_id} not found",
            )

        # Delete employee
        result = await db.digital_employee_configs.delete_one(
            {"employee_id": employee_id}
        )

        return {
            "code": 200,
            "message": "success",
            "data": {"deleted_count": result.deleted_count}
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Delete employee error: employee_id={employee_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete employee",
        )


@router.put("/update/{employee_id}")
async def update_employee(
    employee_id: str = Path(..., description="Employee ID"),
    request: UpdateEmployeeRequest = Body(None),
    api_key: str = Depends(get_api_key),
    db=Depends(get_database),
):
    """
    更新数字员工配置（按 employee_id 部分更新）。

    \nArgs:
        \n- employee_id: Employee ID
        \n- request: UpdateEmployeeRequest with fields to update (all optional)
        \n- api_key: API key from auth
        \n- db: Database instance

    \nReturns:
        \n- Updated employee configuration

    \nExample:
        \n- Update kb_ids: {"kb_ids": ["kb_8aa64d4d6698"]}
        \n- Update name: {"name": "陈晓燕"}
    """
    try:
        logger.info(f"Update employee request employee_id={employee_id} update_fields={request.model_dump(exclude_none=True)}")

        # Check if employee exists
        existing = await db.digital_employee_configs.find_one(
            {"employee_id": employee_id}
        )
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Employee {employee_id} not found",
            )

        # Build update document with only non-None fields
        update_doc = {"updated_at": datetime.utcnow()}

        # Core fields
        if request.name is not None:
            update_doc["name"] = request.name
        if request.position is not None:
            update_doc["position"] = request.position
        if request.intro is not None:
            update_doc["intro"] = request.intro

        # Personality/Style fields
        if request.persona is not None:
            update_doc["persona"] = request.persona
        if request.tone is not None:
            update_doc["tone"] = request.tone
        if request.style is not None:
            update_doc["style"] = request.style
        if request.style_desc is not None:
            update_doc["style_desc"] = request.style_desc
        if request.language is not None:
            update_doc["language"] = request.language

        # Configuration fields
        if request.kb_ids is not None:
            update_doc["kb_ids"] = request.kb_ids
        if request.web_search_enabled is not None:
            update_doc["web_search_enabled"] = request.web_search_enabled
        if request.is_multimodal is not None:
            update_doc["is_multimodal"] = request.is_multimodal

        # FAQ settings
        if request.faq_sim_threshold is not None:
            update_doc["faq_sim_threshold"] = request.faq_sim_threshold
        if request.faq_top_k is not None:
            update_doc["faq_top_k"] = request.faq_top_k

        # Prologue settings
        if request.prologue is not None:
            update_doc["prologue"] = request.prologue
        if request.is_opening_questions is not None:
            update_doc["is_opening_questions"] = request.is_opening_questions

        # Custom prompt
        if request.is_my_prompt is not None:
            update_doc["is_my_prompt"] = request.is_my_prompt
        if request.my_prompt is not None:
            update_doc["my_prompt"] = request.my_prompt

        # Display settings
        if request.is_show_sign is not None:
            update_doc["is_show_sign"] = request.is_show_sign
        if request.portrait is not None:
            update_doc["portrait"] = request.portrait
        if request.model_image is not None:
            update_doc["model_image"] = request.model_image

        # Status
        if request.onduty_status is not None:
            update_doc["onduty_status"] = request.onduty_status
        if request.status is not None:
            update_doc["status"] = request.status

        # Metadata
        if request.metadata is not None:
            update_doc["metadata"] = request.metadata
        if request.hot_questions is not None:
            update_doc["hot_questions"] = request.hot_questions

        # Perform update if there are fields to update
        if len(update_doc) > 1:  # More than just updated_at
            await db.digital_employee_configs.update_one(
                {"employee_id": employee_id},
                {"$set": update_doc}
            )
            logger.info(
                "Employee updated successfully",
                employee_id=employee_id,
                updated_fields=list(update_doc.keys())
            )
        else:
            logger.warning(
                "No fields to update",
                employee_id=employee_id
            )

        # Fetch updated document
        updated = await db.digital_employee_configs.find_one(
            {"employee_id": employee_id}
        )
        updated.pop("_id", None)
        updated["created_at"] = updated["created_at"].isoformat() + "Z"
        updated["updated_at"] = updated["updated_at"].isoformat() + "Z"
        if updated.get("synced_at"):
            updated["synced_at"] = updated["synced_at"].isoformat() + "Z"

        return {"code": 200, "message": "success", "data": updated}

    except HTTPException as e:
        print(e)
        logger.exception(e)
        raise e
    except Exception as e:
        print(e)
        logger.exception(e)
        raise e

