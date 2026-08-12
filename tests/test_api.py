from fastapi.testclient import TestClient

from app.backend.main import app


client = TestClient(app)


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={
            "text": (
                "The government announced a new policy today. "
                "Officials said the policy will take effect next month."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "label" in data
    assert "decision_score" in data

    assert data["prediction"] in ["Real", "Fake"]
    assert data["label"] in [0, 1]


def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code in [400, 422]