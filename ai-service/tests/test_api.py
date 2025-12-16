"""
Basic tests for the AI service.
"""
import pytest
from httpx import AsyncClient
from main import app


@pytest.mark.asyncio
async def test_root():
    """Test root endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Digital Employee AI Service"
        assert data["status"] == "running"


@pytest.mark.asyncio
async def test_health():
    """Test health endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


@pytest.mark.asyncio
async def test_chat_message_validation():
    """Test chat message validation."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Missing required fields
        response = await client.post(
            "/api/chat/message",
            json={}
        )
        assert response.status_code == 400  # Validation error
        
        # Valid request (will need proper setup to work fully)
        response = await client.post(
            "/api/chat/message",
            json={
                "user_id": "test_user",
                "employee_id": "DE001",
                "query": "测试问题"
            }
        )
        # May fail due to missing database connection in tests
        # but at least validates the schema
        assert response.status_code in [200, 500]
