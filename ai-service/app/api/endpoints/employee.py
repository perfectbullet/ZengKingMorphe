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

