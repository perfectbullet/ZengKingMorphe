"""
Digital Employee management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from datetime import datetime

from app.models.schemas import CreateEmployeeRequest, UpdateEmployeeRequest
from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


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
        logger.error(
            f"Get employee error: employee_id={employee_id}, error={str(e)}",
            exc_info=True,
        )
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
        logger.error(
            f"Create employee error: employee_id={request.employee_id}, error={str(e)}",
            exc_info=True,
        )
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
        logger.error(
            f"Create test financial analyst error: error={str(e)}",
            exc_info=True,
        )
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
        logger.error(
            f"Delete employee error: employee_id={employee_id}, error={str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete employee",
        )

