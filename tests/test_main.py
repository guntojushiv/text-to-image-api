from fastapi.testclient import TestClient  
from main import app  
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


client = TestClient(app)


def test_generate_image():
    response = client.post("/generate?prompt=A futuristic cityscape")
    assert response.status_code == 200
    assert "image" in response.json()
    assert response.json()["image"] is not None


def test_analyze_image():
    sample_image_path = "tests/sample_image.jpg"

    assert os.path.exists(
        sample_image_path), f"Test image '{sample_image_path}' not found."

    with open(sample_image_path, "rb") as image_file:
        files = {"image": ("sample_image.jpg", image_file, "image/jpeg")}
        response = client.post("/analyze", files=files)

    assert response.status_code == 200
    assert "similarity_score" in response.json()
    assert isinstance(response.json()["similarity_score"], float)
