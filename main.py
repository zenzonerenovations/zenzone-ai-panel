"""
main.py — ZenZone AI Panel  |  FastAPI application entry point
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File as FFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import (
    Conversation, Message, AIResponse, FileAttachment,
    get_db, init_db,
)
from debate import run_debate
from file_handler import process_upload

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="ZenZone AI Panel", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup():
    init_db()


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    file_text: Optional[str] = None
    file_name: Optional[str] = None
    is_image: Optional[bool] = False


class ConversationRename(BaseModel):
    title: str


# ── Conversations ─────────────────────────────────────────────────────────────

@app.post("/api/conversations")
def create_conversation(db: Session = Depends(get_db)):
    conv = Conversation(title="New Conversation")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return _conv_summary(conv)


@app.get("/api/conversations")
def list_conversations(db: Session = Depends(get_db)):
    convs = (db.query(Conversation)
               .order_by(Conversation.updated_at.desc())
               .all())
    return [_conv_summary(c) for c in convs]


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: int, db: Session = Depends(get_db)):
    conv = _get_conv_or_404(conv_id, db)
    return {
        **_conv_summary(conv),
        "messages": [_serialize_message(m) for m in conv.messages],
    }


@app.patch("/api/conversations/{conv_id}")
def rename_conversation(conv_id: int, body: ConversationRename,
                        db: Session = Depends(get_db)):
    conv = _get_conv_or_404(conv_id, db)
    conv.title = body.title[:120]
    db.commit()
    return _conv_summary(conv)


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: int, db: Session = Depends(get_db)):
    conv = _get_conv_or_404(conv_id, db)
    db.delete(conv)
    db.commit()
    return {"ok": True}


# ── Chat (SSE streaming) ──────────────────────────────────────────────────────

@app.post("/api/conversations/{conv_id}/chat")
async def chat(conv_id: int, body: ChatRequest, db: Session = Depends(get_db)):
    conv = _get_conv_or_404(conv_id, db)

    # Build chat history for the AI (user / assistant turns only)
    history = [
        {"role": m.role, "content": m.content}
        for m in conv.messages
    ]

    # Persist user message
    user_msg = Message(
        conversation_id=conv_id,
        role="user",
        content=body.question,
    )
    db.add(user_msg)

    # Auto-title the conversation from the first message
    if len(conv.messages) == 0:
        conv.title = body.question[:80] + ("…" if len(body.question) > 80 else "")

    conv.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user_msg)

    # Attach file metadata if present
    if body.file_name:
        att = FileAttachment(
            message_id=user_msg.id,
            filename=body.file_name,
            file_type="text" if not body.is_image else "image",
            is_image=body.is_image or False,
            text_content=body.file_text,
        )
        db.add(att)
        db.commit()

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_progress(msg: str):
            await queue.put(("status", msg))

        task = asyncio.create_task(
            run_debate(body.question, history, body.file_text, on_progress)
        )

        # Stream progress events while debate runs
        while not task.done():
            try:
                kind, data = await asyncio.wait_for(queue.get(), timeout=0.3)
                yield f"event: {kind}\ndata: {json.dumps({'message': data})}\n\n"
            except asyncio.TimeoutError:
                yield "event: ping\ndata: {}\n\n"

        # Drain any remaining progress messages
        while not queue.empty():
            kind, data = queue.get_nowait()
            yield f"event: {kind}\ndata: {json.dumps({'message': data})}\n\n"

        try:
            result = task.result()
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
            return

        # Persist assistant message + individual AI responses
        new_db = next(get_db())
        try:
            asst_msg = Message(
                conversation_id=conv_id,
                role="assistant",
                content=result["synthesis"],
                debate_summary=result["summary"],
            )
            new_db.add(asst_msg)
            new_db.commit()
            new_db.refresh(asst_msg)

            for ai_name in ("claude", "chatgpt", "grok"):
                ai_r = AIResponse(
                    message_id=asst_msg.id,
                    ai_name=ai_name,
                    round1_response=result["round1"][ai_name],
                    round2_response=result["round2"][ai_name],
                    changed_position=result["changes"][ai_name],
                )
                new_db.add(ai_r)

            # Touch conversation timestamp
            c = new_db.query(Conversation).filter(Conversation.id == conv_id).first()
            if c:
                c.updated_at = datetime.utcnow()
            new_db.commit()
        finally:
            new_db.close()

        yield f"event: result\ndata: {json.dumps(result)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── File upload ───────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = FFile(...)):
    result = await process_upload(file)
    return result


# ── Trends ────────────────────────────────────────────────────────────────────

@app.get("/api/trends")
def get_trends(db: Session = Depends(get_db)):
    ai_names = ("claude", "chatgpt", "grok")
    stats = {}
    for name in ai_names:
        rows = db.query(AIResponse).filter(AIResponse.ai_name == name).all()
        total   = len(rows)
        changed = sum(1 for r in rows if r.changed_position)
        stats[name] = {
            "total":        total,
            "changed":      changed,
            "stable":       total - changed,
            "change_rate":  round(changed / total * 100, 1) if total else 0,
        }

    # Recent debate events (last 50 AI responses)
    recent_responses = (
        db.query(AIResponse)
          .order_by(AIResponse.created_at.desc())
          .limit(50)
          .all()
    )
    timeline = []
    for r in recent_responses:
        msg = r.message
        if msg:
            timeline.append({
                "date":    r.created_at.isoformat(),
                "ai":      r.ai_name,
                "changed": r.changed_position,
                "preview": msg.content[:80] + "…" if len(msg.content) > 80 else msg.content,
                "conv_id": msg.conversation_id,
            })

    # Total conversations & messages
    total_convs = db.query(Conversation).count()
    total_msgs  = db.query(Message).filter(Message.role == "user").count()

    return {
        "ai_stats":       stats,
        "timeline":       timeline,
        "total_conversations": total_convs,
        "total_questions":     total_msgs,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_conv_or_404(conv_id: int, db: Session) -> Conversation:
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


def _conv_summary(conv: Conversation) -> dict:
    return {
        "id":         conv.id,
        "title":      conv.title,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
    }


def _serialize_message(msg: Message) -> dict:
    d = {
        "id":             msg.id,
        "role":           msg.role,
        "content":        msg.content,
        "debate_summary": msg.debate_summary,
        "created_at":     msg.created_at.isoformat(),
        "attachments":    [
            {"filename": a.filename, "file_type": a.file_type, "is_image": a.is_image}
            for a in msg.attachments
        ],
        "ai_responses": [],
    }
    if msg.role == "assistant":
        d["ai_responses"] = [
            {
                "ai_name":          r.ai_name,
                "round1":           r.round1_response,
                "round2":           r.round2_response,
                "changed_position": r.changed_position,
            }
            for r in msg.ai_responses
        ]
    return d


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
