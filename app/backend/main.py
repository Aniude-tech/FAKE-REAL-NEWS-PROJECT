from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predictor import predict_news


app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting whether a news article is likely to be real or fake.",
    version="1.0.0"
)


class NewsRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "Fake News Detector API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: NewsRequest):

    try:
        result = predict_news(request.text)

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )