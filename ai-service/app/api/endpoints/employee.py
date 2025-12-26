"""
Digital Employee management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from datetime import datetime

from app.models.schemas import (
    CreateEmployeeRequest,
    UpdateEmployeeRequest
)
from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/detail/{employee_id}")
async def get_employee(
    employee_id: str = Path(..., description="Employee ID"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
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
        
        employee = await db.employee_configs.find_one({"employee_id": employee_id})
        
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        # Convert MongoDB document to dict
        employee.pop("_id", None)
        employee["created_at"] = employee["created_at"].isoformat() + "Z"
        employee["updated_at"] = employee["updated_at"].isoformat() + "Z"
        if employee.get("synced_at"):
            employee["synced_at"] = employee["synced_at"].isoformat() + "Z"
        
        return {
            "code": 200,
            "message": "success",
            "data": employee
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get employee error: employee_id={employee_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get employee"
        )

