"""
WebSocket API endpoints for real-time stream chunks updates.
"""

from typing import Optional, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import asyncio
from datetime import datetime

from app.core.database import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Configuration constants
INITIAL_CHUNK_COUNT = 20  # Number of recent chunks to fetch on connection
POLL_INTERVAL = 0.5  # Polling interval in seconds
HEARTBEAT_INTERVAL = 30  # Heartbeat interval in seconds
MAX_CHUNKS_PER_POLL = 100  # Maximum chunks to fetch per poll


@router.websocket("/ws/chunks")
async def websocket_stream_chunks(
    websocket: WebSocket,
    user_id: str = Query(..., description="User ID"),
    employee_id: str = Query(..., description="Employee ID"),
    session_id: str = Query(..., description="Session ID"),
):
    """
    WebSocket endpoint for real-time stream chunks updates.

    Connection parameters:
    - user_id: User ID (required)
    - employee_id: Digital employee ID (required)
    - session_id: Session ID (required)

    Behavior:
    1. On connection, fetch the latest N chunks and send them
    2. Poll for new chunks periodically (created_at >= last_timestamp, excluding sent)
    3. Send new chunks to the client as they arrive
    4. Send heartbeat messages periodically to keep connection alive

    Message formats:
    - Chunk data: Full chunk object with chunk_id, chunk_type, chunk_data, etc.
    - Heartbeat: {"type": "heartbeat", "timestamp": "ISO8601 timestamp"}
    """
    await websocket.accept()
    logger.info(
        "WebSocket connected",
        user_id=user_id,
        employee_id=employee_id,
        session_id=session_id
    )

    db = None
    last_timestamp: Optional[datetime] = None
    sent_chunk_ids: Set[str] = set()

    try:
        db = await get_database()

        # Build query filter
        query_filter = {
            "user_id": user_id,
            "employee_id": employee_id,
            "session_id": session_id,
        }

        # Initial query: get the latest N chunks
        initial_chunks = await db.stream_chunks.find(
            query_filter
        ).sort("created_at", -1).limit(INITIAL_CHUNK_COUNT).to_list(length=INITIAL_CHUNK_COUNT)

        if initial_chunks:
            # Reverse to send in chronological order (oldest to newest)
            for chunk in reversed(initial_chunks):
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in sent_chunk_ids:
                    await send_chunk(websocket, chunk)
                    sent_chunk_ids.add(chunk_id)
                    last_timestamp = chunk.get("created_at")

            logger.info(
                "Sent initial chunks",
                count=len(sent_chunk_ids),
                last_timestamp=last_timestamp.isoformat() if last_timestamp else None
            )
        else:
            logger.info("No existing chunks found for session")

        # Start heartbeat and polling tasks
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
        poll_task = asyncio.create_task(
            poll_new_chunks(websocket, db, query_filter, last_timestamp, sent_chunk_ids)
        )

        # Wait for connection close or task completion
        done, pending = await asyncio.wait(
            [heartbeat_task, poll_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected",
            user_id=user_id,
            session_id=session_id
        )
    except Exception as e:
        logger.error(
            "WebSocket error",
            error=str(e),
            user_id=user_id,
            session_id=session_id,
            exc_info=True
        )
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
    finally:
        logger.debug(
            "WebSocket connection cleanup",
            user_id=user_id,
            session_id=session_id,
            sent_chunks_count=len(sent_chunk_ids)
        )


async def send_chunk(websocket: WebSocket, chunk: dict) -> None:
    """
    Send a chunk data message to the WebSocket client.

    Args:
        websocket: WebSocket connection
        chunk: Chunk document from MongoDB
    """
    # Create a copy to avoid modifying the original
    chunk_copy = chunk.copy()

    # Remove MongoDB _id field
    chunk_copy.pop("_id", None)

    # Convert datetime fields to ISO format strings
    if "timestamp" in chunk_copy and isinstance(chunk_copy["timestamp"], datetime):
        chunk_copy["timestamp"] = chunk_copy["timestamp"].isoformat() + "Z"
    if "created_at" in chunk_copy and isinstance(chunk_copy["created_at"], datetime):
        chunk_copy["created_at"] = chunk_copy["created_at"].isoformat() + "Z"

    await websocket.send_json(chunk_copy)


async def send_heartbeat(websocket: WebSocket) -> None:
    """
    Send periodic heartbeat messages to keep the connection alive.

    Args:
        websocket: WebSocket connection
    """
    try:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            logger.debug("Heartbeat sent")
    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.debug("Heartbeat task stopped")
        raise


async def poll_new_chunks(
    websocket: WebSocket,
    db,
    query_filter: dict,
    last_timestamp: Optional[datetime],
    sent_chunk_ids: Set[str]
) -> None:
    """
    Poll for new chunks and send them to the client.

    Args:
        websocket: WebSocket connection
        db: MongoDB database instance
        query_filter: Base query filter (user_id, employee_id, session_id)
        last_timestamp: Timestamp of the last sent chunk
        sent_chunk_ids: Set of chunk_ids that have already been sent
    """
    # Use a mutable container to track state across iterations
    state = {
        "last_timestamp": last_timestamp,
        "sent_chunk_ids": sent_chunk_ids
    }

    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL)

            # Build query for new data
            new_query = query_filter.copy()

            current_last = state["last_timestamp"]
            current_sent = state["sent_chunk_ids"]

            if current_last:
                # Query for chunks created at or after last_timestamp, excluding already sent ones
                new_query["created_at"] = {"$gte": current_last}
                if current_sent:
                    new_query["chunk_id"] = {"$nin": list(current_sent)}

                logger.debug(
                    "Polling for new chunks",
                    last_timestamp=current_last.isoformat() if current_last else None,
                    sent_count=len(current_sent)
                )
            else:
                # No last_timestamp, fetch only the latest one
                latest = await db.stream_chunks.find_one(
                    query_filter,
                    sort=[("created_at", -1)]
                )
                if latest:
                    chunk_id = latest.get("chunk_id")
                    if chunk_id and chunk_id not in current_sent:
                        await send_chunk(websocket, latest)
                        current_sent.add(chunk_id)
                        state["last_timestamp"] = latest.get("created_at")
                continue

            # Query new chunks (ascending by created_at)
            cursor = db.stream_chunks.find(new_query).sort("created_at", 1)
            new_chunks = await cursor.to_list(length=MAX_CHUNKS_PER_POLL)

            if new_chunks:
                logger.info(
                    "Found new chunks",
                    count=len(new_chunks),
                    last_timestamp=current_last.isoformat() if current_last else None
                )

            # Send each new chunk
            for chunk in new_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in current_sent:
                    await send_chunk(websocket, chunk)
                    current_sent.add(chunk_id)
                    state["last_timestamp"] = chunk.get("created_at")
                    logger.info(
                        "Sent new chunk",
                        chunk_id=chunk_id,
                        chunk_type=chunk.get("chunk_type"),
                        sequence=chunk.get("sequence"),
                        created_at=chunk.get("created_at").isoformat() if chunk.get("created_at") else None
                    )

    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.debug("Poll task stopped")
        raise
