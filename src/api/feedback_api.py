from fastapi import APIRouter, HTTPException, status, Depends
from typing import List

from src.models.feedback import (
    FeedbackCreate,
    FeedbackResponse,
    BulkFeedbackCreate,
    FeedbackStats,
)
from src.services.feedback_processor import feedback_processor, FeedbackProcessor

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
    responses={404: {"description": "Not found"}},
)

# Dependency injection için bir fonksiyon
def get_feedback_processor():
    return feedback_processor

@router.post("/submit", status_code=status.HTTP_201_CREATED, summary="Submit a single feedback")
def submit_feedback(
    feedback: FeedbackCreate,
    processor: FeedbackProcessor = Depends(get_feedback_processor)
):
    """
    Tek bir geri bildirim kaydı oluşturur.

    - **interaction_id**: Etkileşim ID'si (zorunlu).
    - **student_id**: Öğrenci ID'si (zorunlu).
    - **feedback_type**: Geri bildirim türü (zorunlu).
    - **comment**: İsteğe bağlı metin yorumu.
    """
    try:
        result = processor.process_feedback(feedback)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Geri bildirim işlenirken bir hata oluştu: {e}",
        )

@router.post("/bulk", status_code=status.HTTP_201_CREATED, summary="Submit multiple feedbacks")
def submit_bulk_feedback(
    bulk_feedback: BulkFeedbackCreate,
    processor: FeedbackProcessor = Depends(get_feedback_processor)
):
    """
    Toplu olarak birden fazla geri bildirim kaydı oluşturur.
    """
    try:
        result = processor.process_bulk_feedback(bulk_feedback)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Toplu geri bildirim işlenirken bir hata oluştu: {e}",
        )

# TODO: Servis tam olarak implemente edildiğinde response_model=List[FeedbackResponse] eklenmeli.
@router.get("/{interaction_id}", summary="Get feedbacks for an interaction")
def get_feedbacks_for_interaction(
    interaction_id: str,
    processor: FeedbackProcessor = Depends(get_feedback_processor)
):
    """
    Belirli bir etkileşime (`interaction_id`) ait tüm geri bildirimleri getirir.
    """
    try:
        feedbacks = processor.get_feedbacks_by_interaction(interaction_id)
        return feedbacks
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Geri bildirimler getirilirken bir hata oluştu: {e}",
        )

@router.get("/stats/", response_model=FeedbackStats, summary="Get feedback statistics")
def get_feedback_statistics(
    processor: FeedbackProcessor = Depends(get_feedback_processor)
):
    """
    Tüm geri bildirimler için genel istatistikleri getirir.
    """
    try:
        stats = processor.get_feedback_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"İstatistikler getirilirken bir hata oluştu: {e}",
        )