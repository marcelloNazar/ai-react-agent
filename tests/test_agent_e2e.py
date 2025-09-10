import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from app.main import app
import httpx

class TestAgentE2E:
    """End-to-end tests specifically for AI Agent functionality."""

    def test_agent_chat_basic(self, client):
        """Test basic chat functionality."""
        payload = {
            "message": "Hello, how are you?",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "thread_id" in data
        assert "response" in data
        assert "message_history" in data
        assert "status" in data
        assert data["status"] == "completed"

    def test_agent_chat_with_calculation(self, client):
        """Test chat with mathematical calculation."""
        payload = {
            "message": "What is 2 + 2?",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "4" in data["response"] or "four" in data["response"].lower()
        assert data["status"] == "completed"

    def test_agent_chat_with_thread_continuation(self, client):
        """Test conversation continuation with thread_id."""
        # First message
        payload1 = {
            "message": "My name is Alice",
            "context": {}
        }

        response1 = client.post("/agent/chat", json=payload1)
        assert response1.status_code == 200

        data1 = response1.json()
        thread_id = data1["thread_id"]

        # Second message with same thread_id
        payload2 = {
            "message": "What is my name?",
            "thread_id": thread_id,
            "context": {}
        }

        response2 = client.post("/agent/chat", json=payload2)
        assert response2.status_code == 200

        data2 = response2.json()
        assert data2["thread_id"] == thread_id
        # The agent should remember the name from previous conversation
        assert "alice" in data2["response"].lower() or "Alice" in data2["response"]

    def test_agent_stream_endpoint(self, client):
        """Test streaming endpoint."""
        payload = {
            "message": "Tell me a short joke",
            "context": {}
        }

        response = client.post("/agent/stream", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Check that we get SSE formatted data
        content = response.text
        assert "data:" in content
        assert "event" in content

    def test_math_tools(self, client):
        """Test all mathematical operations."""
        test_cases = [
            ("What is 10 + 5?", "15"),
            ("Calculate 20 - 8", "12"),
            ("Multiply 6 by 7", "42"),
            ("Divide 100 by 4", "25")
        ]

        for question, expected_answer in test_cases:
            payload = {
                "message": question,
                "context": {}
            }

            response = client.post("/agent/chat", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert expected_answer in data["response"]
            assert data["status"] == "completed"

    def test_division_by_zero_handling(self, client):
        """Test division by zero error handling."""
        payload = {
            "message": "What is 10 divided by 0?",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        # Should handle the error gracefully
        response_lower = data["response"].lower()
        assert any(keyword in response_lower for keyword in ["cannot", "error", "undefined", "not possible", "zero"])

    def test_invalid_request_format(self, client):
        """Test handling of invalid request formats."""
        # Missing required message field
        payload = {
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 422  # Validation error

    def test_empty_message(self, client):
        """Test handling of empty message."""
        payload = {
            "message": "",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        # Should still work, just with empty input
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_async_client_functionality(self, async_client):
        """Test async client functionality."""
        payload = {
            "message": "Hello from async client",
            "context": {}
        }

        response = await async_client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert data["status"] == "completed"

    def test_context_passing(self, client):
        """Test that context is properly passed to the agent."""
        context = {
            "user_preference": "concise answers",
            "session_info": "test_session"
        }

        payload = {
            "message": "Give me a brief explanation of AI",
            "context": context
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"

    def test_multiple_tool_calls(self, client):
        """Test requests that might require multiple tool calls."""
        payload = {
            "message": "Calculate 5 + 3 and then multiply the result by 2",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        # Should show the final result: (5+3)*2 = 16
        assert "16" in data["response"]
        assert data["status"] == "completed"

    def test_web_search_bh_weather(self, client):
        """Test web search for weather in Belo Horizonte."""
        payload = {
            "message": "How is the weather in Belo Horizonte, Brazil?",
            "context": {}
        }

        response = client.post("/agent/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"
        # Should contain weather information or indicate search availability
        response_lower = data["response"].lower()
        assert any(keyword in response_lower for keyword in ["weather", "temperature", "belo horizonte", "not available"])