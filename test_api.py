from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_sentiment_endpoint():
    response = client.post("/sentiment?text=I love this project!")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    
    assert data["prediction"][0]["label"] == "POSITIVE"