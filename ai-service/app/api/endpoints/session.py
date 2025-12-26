"""
Session management API endpoints.
"""
import hashlib
import httpx
import os
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from app.models.schemas import SessionResponse, CreateSessionRequest, ExternalEmployeeAPIResponse
from app.models.database import DigitalEmployeeConfigModel
from app.api.middleware.auth import get_api_key
from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


async def fetch_external_employee_data(employee_id: str) -> Optional[dict]:
    """
    从外部API获取数字员工数据。
    
    Args:
        employee_id: 员工ID
        
    Returns:
        外部API返回的data字段数据，或None（失败时）
    """
    # 从环境变量读取外部API地址
    external_api_url = os.getenv(
        "EXTERNAL_EMPLOYEE_API_URL",
        "http://192.168.9.39/edu-api/avatar/api/digitalEmployee/get"
    )
    
    logger.info(f"Fetching external employee data: employee_id={employee_id}, api_url={external_api_url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(external_api_url, params={"employeeId": employee_id})
            response.raise_for_status()
            
            api_response = ExternalEmployeeAPIResponse(**response.json())
            
            if not api_response.success:
                logger.error(
                    f"{'!' * 100}\n"
                    f"外部API调用失败！强烈谴责！\n"
                    f"API URL: {external_api_url}\n"
                    f"Employee ID: {employee_id}\n"
                    f"Error: {api_response.error}\n"
                    f"{'!' * 100}"
                )
                return None
            
            logger.info(f"Successfully fetched employee data: employee_id={employee_id}")
            return api_response.data.model_dump()
            
    except httpx.HTTPError as e:
        logger.error(
            f"{'!' * 100}\n"
            f"外部API HTTP请求失败！强烈谴责！\n"
            f"API URL: {external_api_url}\n"
            f"Employee ID: {employee_id}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Details: {str(e)}\n"
            f"{'!' * 100}",
            exc_info=True
        )
        return None
    except Exception as e:
        logger.error(
            f"{'!' * 100}\n"
            f"外部API调用发生未知错误！强烈谴责！\n"
            f"API URL: {external_api_url}\n"
            f"Employee ID: {employee_id}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Details: {str(e)}\n"
            f"{'!' * 100}",
            exc_info=True
        )
        return None


async def sync_digital_employee_config(db, external_data: dict) -> Optional[str]:
    """
    同步数字员工配置到MongoDB。
    
    Args:
        db: Database instance
        external_data: 外部API返回的data字段数据
        
    Returns:
        employee_id (str) or None
    """
    try:
        employee_info = external_data["employee"]
        setting_info = external_data["setting"]
        
        # Convert employee.id to employee_id string
        employee_id = str(employee_info["id"])
        
        # Extract kb_ids from ragDatasets
        kb_ids = [
            dataset["rag_dataset_id"] 
            for dataset in setting_info["knowledge"]["rag_datasets"]
            if dataset.get("is_enable") == 1
        ]
        
        # Extract hot_questions from prologue
        prologue_config = setting_info["prologue"]
        hot_questions = prologue_config.get("my_questions", [])
        
        # Extract chat rules
        chat_rule = setting_info["rule"]["chat_rule"]
        unusual_rule = setting_info["rule"]["unusual_rule"]
        llm_reply = unusual_rule.get("llm_reply", {})
        
        # Build employee config document
        config_doc = DigitalEmployeeConfigModel(
            employee_id=employee_id,
            external_employee_id=employee_info["id"],
            team_id=employee_info["team_id"],
            name=employee_info["name"],
            position=employee_info["position"],
            employee_type=employee_info["type"],
            tone=employee_info["tone"],
            language=employee_info["language"],
            gender=employee_info["gender"],
            intro=employee_info.get("intro"),
            portrait=employee_info.get("portrait"),
            model_image=employee_info.get("model_image"),
            digital_code=employee_info.get("digital_code"),
            onduty_status=employee_info["onduty_status"],
            # Knowledge configuration
            kb_ids=kb_ids,
            faq_count=len(setting_info["knowledge"]["faqs"]),
            # Prologue configuration
            prologue=prologue_config.get("prologue"),
            is_opening_questions=prologue_config.get("is_opening_questions", False),
            hot_questions=hot_questions,
            # Chat rules
            is_multimodal=chat_rule.get("is_multimodal", False),
            faq_sim_threshold=chat_rule.get("faq_sim_threshold", 0.0),
            faq_top_k=chat_rule.get("faq_top_k", 1),
            # LLM configuration
            web_search_enabled=llm_reply.get("is_web_search", False) if llm_reply else False,
            is_show_sign=llm_reply.get("is_show_sign", False) if llm_reply else False,
            is_my_prompt=llm_reply.get("is_my_prompt", False) if llm_reply else False,
            my_prompt=llm_reply.get("my_prompt") if llm_reply else None,
            # Role configuration
            persona=setting_info["role"].get("persona"),
            style=setting_info["role"].get("style"),
            style_desc=setting_info["role"].get("style_desc"),
            # Timestamps
            external_update_time=employee_info["update_time"],
            external_create_time=employee_info["create_time"],
        ).model_dump()
        
        # Upsert to MongoDB
        await db.digital_employee_configs.update_one(
            {"employee_id": employee_id},
            {"$set": config_doc},
            upsert=True
        )
        
        logger.info(f"Digital employee config synced: employee_id={employee_id}, kb_ids={kb_ids}, faq_count={len(setting_info['knowledge']['faqs'])}")
        
        return employee_id
        
    except Exception as e:
        logger.error(f"Failed to sync digital employee config: {e}", exc_info=True)
        return None


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    创建新会话（集成外部API调用和FAQ向量化）。
    
    Args:
        - request: Session creation request
        - api_key: API key from auth
        - db: Database instance
        
    Returns:
        - Created session information
    """
    try:
        logger.info(f"Create session request: user_id={request.user_id}, employee_id={request.employee_id}")
        
        # Step 1: Fetch external employee data
        external_data = await fetch_external_employee_data(request.employee_id)
        
        if external_data:
            # Step 2: Sync employee config to MongoDB
            synced_employee_id = await sync_digital_employee_config(db, external_data)
            
            if synced_employee_id:
                # Step 3: Trigger FAQ vectorization task (async background)
                from app.services.task_processor import task_processor
                
                faqs = external_data["setting"]["knowledge"]["faqs"]
                prologue_faqs = external_data["setting"]["prologue"].get("faqs", [])
                all_faqs = faqs + prologue_faqs
                
                if all_faqs:
                    task_id = await task_processor.submit_faq_vectorization_task(
                        employee_id=synced_employee_id,
                        faqs=all_faqs
                    )
                    logger.info(f"FAQ vectorization task submitted: task_id={task_id}, faq_count={len(all_faqs)}")
        else:
            logger.warning(f"Failed to fetch external employee data for {request.employee_id}, using existing config")
        
        # Step 4: Check if employee exists (either from sync or existing)
        employee = await db.employee_configs.find_one({"employee_id": request.employee_id})
        if not employee:
            # Also check digital_employee_configs collection
            employee = await db.digital_employee_configs.find_one({"employee_id": request.employee_id})
            
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Employee {request.employee_id} not found and external sync failed"
            )
        
        # Step 5: Generate session ID
        timestamp = datetime.utcnow().timestamp()
        session_id = f"sess_{hashlib.md5(f'{request.user_id}_{timestamp}'.encode()).hexdigest()[:12]}"
        
        # Step 6: Create session document
        session_doc = {
            "session_id": session_id,
            "user_id": request.user_id,
            "employee_id": request.employee_id,
            "status": "active",
            "message_count": 0,
            "context_messages": [],
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "ended_at": None,
            "metadata": request.metadata
        }
        
        # Insert into database
        await db.sessions.insert_one(session_doc)
        
        logger.info(f"Session created: session_id={session_id}, user_id={request.user_id}")
        
        # Format response
        session_doc.pop("_id", None)
        session_doc["created_at"] = session_doc["created_at"].isoformat() + "Z"
        session_doc["last_activity"] = session_doc["last_activity"].isoformat() + "Z"
        
        return {
            "code": 201,
            "message": "Session created successfully",
            "data": session_doc
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create session: error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str = Path(..., description="Session ID"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取会话信息。
    
    \nArgs:
        \n- session_id: Session ID
        \n- api_key: API key from auth
        \n- db: Database instance
        
    \nReturns:
        \n- Session information
    """
    try:
        logger.info(f"Get session request: session_id={session_id}")
        
        # Get session from database
        session = await db.sessions.find_one({"session_id": session_id})
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Convert MongoDB document to dict
        session.pop("_id", None)
        session["created_at"] = session["created_at"].isoformat() + "Z"
        session["last_activity"] = session["last_activity"].isoformat() + "Z"
        if session.get("ended_at"):
            session["ended_at"] = session["ended_at"].isoformat() + "Z"
        
        return SessionResponse(
            code=200,
            message="success",
            data=session
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session error: session_id={session_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session"
        )


@router.delete("/{session_id}")
async def end_session(
    session_id: str = Path(..., description="Session ID"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    结束会话。
    
    \nArgs:
        \n- session_id: Session ID
        \n- api_key: API key from auth
        \n- db: Database instance
        
    \nReturns:
        \n- Success message
    """
    try:
        logger.info(f"End session request: session_id={session_id}")
        
        # Update session status
        
        result = await db.sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": "ended",
                    "ended_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {
            "code": 200,
            "message": "Session ended successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"End session error: session_id={session_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end session"
        )


@router.get("/{session_id}/conversations")
async def get_session_conversations(
    session_id: str = Path(..., description="Session ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size"),
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    获取会话下的所有对话记录。
    
    按时间顺序返回该会话中的所有问答记录，支持分页查询。
    
    Args:
        - session_id: Session ID
        - page: Page number (starting from 1)
        - page_size: Records per page (1-200)
        - api_key: API key from auth
        - db: Database instance
        
    Returns:
        - Conversation records for the session with pagination info
    """
    try:
        logger.info(f"Get session conversations request: session_id={session_id}, page={page}, page_size={page_size}")
        
        # First check if session exists
        session = await db.sessions.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Build query filter
        query_filter = {"session_id": session_id}
        
        # Get total count
        total = await db.conversations.count_documents(query_filter)
        
        # Calculate pagination
        skip = (page - 1) * page_size
        total_pages = (total + page_size - 1) // page_size
        
        # Query conversations sorted by created_at (ascending order to show conversation flow)
        cursor = db.conversations.find(query_filter).sort(
            "created_at", 1  # 1 for ascending (oldest first)
        ).skip(skip).limit(page_size)
        
        conversations = await cursor.to_list(length=page_size)
        
        # Format data (remove MongoDB _id field, format datetime)
        formatted_conversations = []
        for conv in conversations:
            conv.pop("_id", None)
            if "created_at" in conv and hasattr(conv["created_at"], "isoformat"):
                conv["created_at"] = conv["created_at"].isoformat()
            if "updated_at" in conv and conv.get("updated_at") and hasattr(conv["updated_at"], "isoformat"):
                conv["updated_at"] = conv["updated_at"].isoformat()
            formatted_conversations.append(conv)
        
        logger.info(f"Retrieved session conversations: session_id={session_id}, total={total}, page={page}, returned={len(formatted_conversations)}")
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "session_id": session_id,
                "conversations": formatted_conversations,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session conversations error: session_id={session_id}, error={str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session conversations"
        )
