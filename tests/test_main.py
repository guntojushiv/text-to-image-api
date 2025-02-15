import pytest
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app

client = TestClient(app)


def test_generate_image():
    response = client.post(
        "/generate", json={"prompt": "A futuristic cityscape"})
    assert response.status_code == 200
    assert "image" in response.json()
