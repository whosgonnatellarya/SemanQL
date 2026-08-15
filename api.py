from fastapi import FastAPI
from pydantic import BaseModel

from main import run_pipeline
from layer3 import run_layer3

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.post("/analyze")
def analyze(request: QuestionRequest):
    """fast analysis: layers 1 and 2 only"""
    return run_pipeline(request.question)


@app.post("/analyze/deep")
def analyze_deep(request: QuestionRequest):
    """deep analysis: layers 1, 2, and 3"""
    standard = run_pipeline(request.question)
    deep = run_layer3(request.question)
    return {
        **standard,
        "layer3": deep,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
