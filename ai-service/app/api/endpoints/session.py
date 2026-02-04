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
    # 🔧 测试模式：如果是 hutao，直接加载本地测试数据
    if employee_id == "hutao":
        try:
            import json
            from pathlib import Path
            
            test_data_file = Path(__file__).parent.parent.parent.parent / "outer_api_docs" / "按员工id返回的数据-hutao.json"
            logger.info(f"🧪 Using local test data for hutao: {test_data_file}")
            
            with open(test_data_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            # 使用 Pydantic 模型解析，保持与API模式一致
            api_response = ExternalEmployeeAPIResponse(**test_data)
            
            if not api_response.success:
                logger.warning(f"Test data indicates failure: status={api_response.status}")
                return None
            
            faq_count = len(api_response.data.setting.knowledge.faqs)
            logger.info(f"✅ Successfully loaded hutao test data: {faq_count} FAQs")
            
            # 返回 model_dump() 结果，自动转换为下划线命名
            return api_response.data.model_dump()
                
        except FileNotFoundError:
            logger.error(f"❌ Test data file not found: {test_data_file}")
            return None
        except json.JSONDecodeError as e:
            logger.exception(f"❌ Test data JSON decode error: {test_data_file}")
            return None
        except Exception as e:
            logger.exception(f"❌ Failed to load test data: {test_data_file}")
            return None
    
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
                    f"哎呀，调用外部接口失败了呢。\n"
                    f"外部接口是: {external_api_url}\n"
                    f"Employee ID: {employee_id}\n"
                    f"Error: {api_response.error}\n"
                    f"{'!' * 100}"
                )
                return None
            
            logger.info(f"Successfully fetched employee data: employee_id={employee_id}")
            return api_response.data.model_dump()
            
    except httpx.HTTPError as e:
        error_message = str(e).replace('{', '{{').replace('}', '}}')
        logger.exception(
            f"{'!' * 50}\n"
            f"外部API HTTP请求失败！\n"
            f"API URL: {external_api_url}\n"
            f"Employee ID: {employee_id}"
        )
        return None
    except Exception as e:
        logger.exception(
            f"{'!' * 50}\n"
            f"外部API调用发生未知错误！\n"
            f"API URL: {external_api_url}\n"
            f"Employee ID: {employee_id}"
        )
        return None


async def sync_digital_employee_config(db, request_employee_id: str, external_data: dict) -> Optional[str]:
    """
    同步数字员工配置、FAQs到MongoDB。
    
    Args:
        db: Database instance
        request_employee_id: 请求参数中的 employee_id（作为系统主键）
        external_data: 外部API返回的data字段数据
        
    Returns:
        employee_id (str) or None
    """
    try:
        employee_info = external_data["employee"]
        setting_info = external_data["setting"]
        setting_info["employee_id"] = employee_info["employee_id"]  # Ensure employee_id is included in setting
        
        # Use request_employee_id as the primary key, not the external id
        employee_id = request_employee_id
        employee_name = employee_info["name"]
        
        # Extract kb_ids from ragDatasets
        kb_ids = [
            dataset["rag_dataset_id"] 
            for dataset in setting_info["knowledge"]["rag_datasets"]
            if dataset.get("is_enable") == 1
        ]
        
        # Extract hot_questions from prologue
        prologue_config = setting_info["prologue"]
        hot_questions = prologue_config.get("my_questions", [])
        
        # Extract chat rules (handle optional fields)
        rule_config = setting_info.get("rule", {})
        chat_rule = rule_config.get("chat_rule") or {}
        unusual_rule = rule_config.get("unusual_rule") or {}
        llm_reply = unusual_rule.get("llm_reply", {}) if unusual_rule else {}
        
        # Build employee config document
        config_doc = DigitalEmployeeConfigModel(
            employee_id=employee_id,  # Use request parameter as primary key
            external_employee_id=employee_info["employee_id"],  # Store external id separately
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
            # Timestamps (handle None values)
            external_update_time=employee_info.get("update_time"),
            external_create_time=employee_info.get("create_time"),
        ).model_dump()
        
        # Upsert to MongoDB
        await db.digital_employee_configs.update_one(
            {"employee_id": employee_id},
            {"$set": config_doc},
            upsert=True
        )
        
        logger.info(
            f"Digital employee config synced: employee_id={employee_id} ({employee_name}), "
            f"external_id={employee_info['employee_id']}, kb_ids={kb_ids}, faq_count={len(setting_info['knowledge']['faqs'])}"
        )
        
        # Sync FAQs to MongoDB faqs collection
        faqs_data = setting_info["knowledge"]["faqs"]
        if faqs_data:
            synced_count = 0
            failed_faqs = []
            current_time = datetime.utcnow()
            logger.info(f"Syncing FAQ: employee_id={employee_id}, faqs_data={faqs_data[0]}")
            for faq_item in faqs_data:
                try:
                    external_faq_id = faq_item.get("faq_id", "")
                    faq_id = f"faq_{employee_id}_{external_faq_id}"
                    
                    # Extract answer texts from answer objects
                    answer_objects = faq_item.get("answers", [])
                    answer_texts = [ans for ans in answer_objects]
                    
                    # Build combined_text for embedding (question + similar questions)
                    similar_questions = faq_item.get("similar_questions", [])
                    combined_text = faq_item.get("question_name", "")
                    if similar_questions:
                        combined_text += " " + " ".join(similar_questions)
                    
                    # Build FAQ document
                    faq_doc = {
                        "faq_id": faq_id,
                        "employee_id": employee_id,
                        "external_faq_id": external_faq_id,
                        "team_id": faq_item.get("team_id", 0),
                        "question_name": faq_item.get("question_name", ""),
                        "similar_questions": similar_questions,
                        "answers": answer_texts,
                        "is_enable": faq_item.get("is_enable", 0),
                        "is_clear": faq_item.get("is_clear", 0),
                        "start_time": faq_item.get("start_time"),
                        "end_time": faq_item.get("end_time"),
                        "update_time": faq_item.get("update_time", ""),
                        "create_time": faq_item.get("create_time", ""),
                        "combined_text": combined_text,
                        "keywords": [],  # Will be populated by vectorization task
                        "vector_id": None,  # Will be set after vectorization
                        "es_indexed": False,  # Will be set after ElasticSearch indexing
                        "created_at": current_time,
                        "synced_at": current_time,
                    }
                    
                    # Upsert FAQ document (idempotent)
                    await db.faqs.update_one(
                        {"faq_id": faq_id},
                        {"$set": faq_doc},
                        upsert=True
                    )
                    
                    synced_count += 1
                    
                except Exception as faq_error:
                    logger.exception(f"Failed to sync FAQ: faq_id={external_faq_id}")
                    failed_faqs.append(external_faq_id)
            
            logger.info(
                f"FAQs synced to MongoDB: employee_id={employee_id}, "
                f"total={len(faqs_data)}, synced={synced_count}, failed={len(failed_faqs)}"
            )
            
            if failed_faqs:
                logger.warning(f"Failed FAQ IDs: {failed_faqs}")
        
        return employee_id
        
    except Exception as e:
        logger.exception(f"Failed to sync digital employee config: employee_id={employee_id}")
        return None


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    api_key: str = Depends(get_api_key),
    db = Depends(get_database)
):
    """
    创建新会话（集成外部API调用和FAQ向量化）。
    
    **请求参数**：
    - `user_id` (required, str): 用户唯一标识，用于关联用户身份和对话历史
    - `employee_id` (required, str): 数字员工ID，用于调用外部API获取员工配置
    - `session_id` (optional, str): 客户端指定的会话ID，支持幂等创建（格式: sess_{12位MD5})
    - `metadata` (optional, dict): 会话元数据，用于记录会话上下文信息
        - `platform`: 来源平台（web/mobile/desktop）
        - `device`: 设备类型（desktop/mobile/tablet）
        - `source`: 来源页面（homepage/chatbot/embed）
        - `user_agent`: 浏览器User-Agent
        - `ip_address`: 客户端IP地址
    
    **响应数据**：
    - `session_id` (str): 自动生成的会话ID（格式: sess_{12位MD5哈希}）
    - `user_id` (str): 用户ID
    - `employee_id` (str): 员工ID
    - `status` (str): 会话状态（"active" | "ended"）
    - `message_count` (int): 消息数量（初始为0）
    - `context_messages` (list): 上下文消息列表（初始为空）
    - `created_at` (str): 创建时间（ISO 8601格式）
    - `last_activity` (str): 最后活动时间（ISO 8601格式）
    - `ended_at` (str|null): 结束时间（初始为null）
    - `metadata` (dict): 会话元数据（来自请求）
    
    Returns:

        - 201: Created session information with session_id and metadata
        - 200: Existing session returned for idempotent request

    """
    def _format_session_doc(session_doc: dict) -> dict:
        formatted = session_doc.copy()
        formatted.pop("_id", None)
        formatted["created_at"] = formatted["created_at"].isoformat() + "Z"
        formatted["last_activity"] = formatted["last_activity"].isoformat() + "Z"
        if formatted.get("ended_at"):
            formatted["ended_at"] = formatted["ended_at"].isoformat() + "Z"
        return formatted

    try:
        logger.info(f"Create session request: user_id={request.user_id}, employee_id={request.employee_id}")
        
        # Step 1: Fetch external employee data
        external_data = await fetch_external_employee_data(request.employee_id)
        
        if external_data:
            # Step 2: 同步数字员工配置、FAQs到MongoDB
            synced_employee_id = await sync_digital_employee_config(db, request.employee_id, external_data)
            
            if synced_employee_id:
                # Step 3: Trigger FAQ vectorization task (async background)
                from app.services.task_processor import task_processor
                
                faqs_count = len(external_data["setting"]["knowledge"]["faqs"])
                
                if faqs_count > 0:
                    employee_name = external_data["employee"]["name"]
                    # task_id = await task_processor.submit_faq_vectorization_task_by_employee_id(
                    #     employee_id=synced_employee_id
                    # )
                    # logger.info(f"FAQ vectorization task submitted: task_id={task_id}, employee_id={synced_employee_id} ({employee_name}), faq_count={faqs_count}")
        else:
            logger.warning(f"Failed to fetch external employee data for {request.employee_id}, using existing config")
        
        # Step 4: Check if employee exists (either from sync or existing)
        employee = await db.digital_employee_configs.find_one({"employee_id": request.employee_id})
            
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Employee {request.employee_id} not found and external sync failed"
            )
        
        # Step 5: Generate session ID
        timestamp = datetime.utcnow().timestamp()
        session_id = request.session_id or f"sess_{hashlib.md5(f'{request.user_id}_{timestamp}'.encode()).hexdigest()[:12]}"
        
        # Step 6: Check if session already exists (idempotent creation)
        existing_session = await db.sessions.find_one({"session_id": session_id})
        if existing_session:
            logger.info(f"Session already exists, returning existing session: session_id={session_id}")
            formatted_session = _format_session_doc(existing_session)
            return {
                "code": 200,
                "message": "Session already exists",
                "data": formatted_session
            }
        
        # Step 7: Create session document
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
        
        formatted_session = _format_session_doc(session_doc)
        return {
            "code": 201,
            "message": "Session created successfully",
            "data": formatted_session
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create session")
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
    
    Args:

        - session_id: Session ID
        - api_key: API key from auth
        - db: Database instance
        
    Returns:

        - Session information
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
        logger.exception(f"Get session error: session_id={session_id}")
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
        logger.exception(f"End session error: session_id={session_id}")
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
        logger.exception(f"Get session conversations error: session_id={session_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session conversations"
        )
