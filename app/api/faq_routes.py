import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.repositories.faq_repository import (
    delete_all_faqs,
    delete_faq,
    get_all_faqs,
    get_faqs_by_character,
    update_faq,
)

router = APIRouter(prefix="/faqs", tags=["FAQs"])

VALID_EMOTIONS = {"happy", "sad", "angry", "disgust", "surprise", "neutral"}


class FAQEmotionUpdate(BaseModel):
    emotion: Optional[str] = None


class FAQUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    tag: Optional[str] = None
    language: Optional[str] = None
    emotion: Optional[str] = None


@router.get("")
async def list_faqs(character_id: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    if character_id:
        faqs = await get_faqs_by_character(db, character_id.lower())
    else:
        faqs = await get_all_faqs(db)
    return [
        {
            "id": str(f.id),
            "character_id": f.character_id,
            "question": f.question,
            "answer": f.answer,
            "emotion": f.emotion,
            "tag": f.tag,
            "language": f.language,
            "audio_url": f.audio_url,
            "created_at": f.created_at,
        }
        for f in faqs
    ]


@router.patch("/{faq_id}/emotion") #bta3deel el emotion bas
async def set_faq_emotion(faq_id: uuid.UUID, body: FAQEmotionUpdate, db: AsyncSession = Depends(get_db)):
    if body.emotion and body.emotion.lower() not in VALID_EMOTIONS:
        raise HTTPException(status_code=422, detail=f"emotion must be one of: {', '.join(sorted(VALID_EMOTIONS))}")
    faq = await update_faq(db, faq_id, {"emotion": body.emotion.lower() if body.emotion else None})
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    return {"id": str(faq.id), "emotion": faq.emotion, "question": faq.question}


@router.patch("/{faq_id}")  
async def update_faq_endpoint(faq_id: uuid.UUID, body: FAQUpdate, db: AsyncSession = Depends(get_db)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if "emotion" in updates and updates["emotion"].lower() not in VALID_EMOTIONS:
        raise HTTPException(status_code=422, detail=f"emotion must be one of: {', '.join(sorted(VALID_EMOTIONS))}")
    if not updates:
        raise HTTPException(status_code=422, detail="No fields provided to update")
    faq = await update_faq(db, faq_id, updates)
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    return {"id": str(faq.id), "emotion": faq.emotion, "question": faq.question, "answer": faq.answer}


@router.delete("/{faq_id}")
async def delete_faq_endpoint(faq_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    deleted = await delete_faq(db, faq_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="FAQ not found")
    return {"deleted": str(faq_id)}


@router.delete("")
async def delete_all_faqs_endpoint(db: AsyncSession = Depends(get_db)):
    count = await delete_all_faqs(db)
    return {"deleted_count": count}
