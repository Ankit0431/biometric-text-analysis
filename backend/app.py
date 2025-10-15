from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


class VerifyRequest(BaseModel):
    user_id: str
    text: str
    lang: str = "en"
    domain_hint: str = "chat"


@app.post("/verify")
async def verify(req: VerifyRequest):
    # Placeholder: in-progress implementation returns a mocked decision
    if not req.text or len(req.text.split()) < 10:
        raise HTTPException(status_code=400, detail="text too short for demo")
    return {
        "decision": "challenge",
        "score": 0.5,
        "reasons": ["DEMO"],
        "thresholds": {"high": 0.84, "med": 0.72},
    }
