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


@router.post("/create")
async def create_employee(
    request: CreateEmployeeRequest,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    创建一个新的数字员工。
    
    Args:
        request: Create employee request
        current_user: Current user from auth
        db: Database instance
        
    Returns:
        Created employee data
    """
    try:
        logger.info("Create employee request", employee_id=request.employee_id)
        
        # Check if employee already exists
        existing = await db.employee_configs.find_one({"employee_id": request.employee_id})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Employee ID already exists"
            )
        
        # Create employee document
        employee_doc = request.model_dump()
        employee_doc["status"] = "active"
        employee_doc["created_at"] = datetime.utcnow()
        employee_doc["updated_at"] = datetime.utcnow()
        
        await db.employee_configs.insert_one(employee_doc)
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "employee_id": request.employee_id,
                "status": "active",
                "created_at": employee_doc["created_at"].isoformat() + "Z"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Create employee error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create employee"
        )


@router.put("/{employee_id}")
async def update_employee(
    employee_id: str = Path(..., description="Employee ID"),
    request: UpdateEmployeeRequest = ...,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    更新数字员工配置。
    
    Args:
        employee_id: Employee ID
        request: Update employee request
        current_user: Current user from auth
        db: Database instance
        
    Returns:
        Updated employee data
    """
    try:
        logger.info("Update employee request", employee_id=employee_id)
        
        # Build update document
        update_doc = {
            k: v for k, v in request.model_dump(exclude_unset=True).items()
            if v is not None
        }
        update_doc["updated_at"] = datetime.utcnow()
        
        result = await db.employee_configs.update_one(
            {"employee_id": employee_id},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "employee_id": employee_id,
                "updated_at": update_doc["updated_at"].isoformat() + "Z"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Update employee error", employee_id=employee_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update employee"
        )


@router.get("/{employee_id}")
async def get_employee(
    employee_id: str = Path(..., description="Employee ID"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取数字员工配置。
    
    Args:
        employee_id: Employee ID
        current_user: Current user from auth
        db: Database instance
        
    Returns:
        Employee configuration
    """
    try:
        logger.info("Get employee request", employee_id=employee_id)
        
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
        logger.error("Get employee error", employee_id=employee_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get employee"
        )


@router.delete("/{employee_id}")
async def delete_employee(
    employee_id: str = Path(..., description="Employee ID"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    删除数字员工。
    
    Args:
        employee_id: Employee ID
        current_user: Current user from auth
        db: Database instance
        
    Returns:
        Success message
    """
    try:
        logger.info("Delete employee request", employee_id=employee_id)
        
        result = await db.employee_configs.delete_one({"employee_id": employee_id})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        return {
            "code": 200,
            "message": "Digital employee deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delete employee error", employee_id=employee_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete employee"
        )


@router.get("/list")
async def list_employees(
    domain: str = Query(None, description="Filter by domain"),
    status: str = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取数字员工列表。
    
    Args:
        domain: Filter by domain (optional)
        status: Filter by status (optional)
        page: Page number
        page_size: Page size
        current_user: Current user from auth
        db: Database instance
        
    Returns:
        List of employees
    """
    try:
        logger.info("List employees request", domain=domain, status=status)
        
        # Build query
        query = {}
        if domain:
            query["domain"] = domain
        if status:
            query["status"] = status
        
        # Get total count
        total = await db.employee_configs.count_documents(query)
        
        # Get paginated results
        cursor = db.employee_configs.find(query).skip((page - 1) * page_size).limit(page_size)
        employees = await cursor.to_list(length=page_size)
        
        # Format results
        items = []
        for emp in employees:
            emp.pop("_id", None)
            items.append({
                "employee_id": emp["employee_id"],
                "name": emp["name"],
                "domain": emp["domain"],
                "role": emp["role"],
                "status": emp["status"]
            })
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": items
            }
        }
        
    except Exception as e:
        logger.error("List employees error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list employees"
        )
