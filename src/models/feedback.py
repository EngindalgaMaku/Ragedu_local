from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class FeedbackType(str, Enum):
    """Geri bildirim türlerini tanımlayan enum."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    UNCLEAR = "unclear"
    OFF_TOPIC = "off_topic"

class FeedbackBase(BaseModel):
    """Geri bildirim için temel model."""
    interaction_id: str = Field(..., description="İlişkili etkileşim ID'si")
    student_id: str = Field(..., description="Geri bildirimi veren öğrenci ID'si")
    feedback_type: FeedbackType = Field(..., description="Geri bildirim türü")
    comment: Optional[str] = Field(None, description="Ek yorumlar")

class FeedbackCreate(FeedbackBase):
    """Yeni bir geri bildirim oluşturmak için kullanılan model."""
    pass

class FeedbackResponse(FeedbackBase):
    """API'den geri dönen geri bildirim modeli."""
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class BulkFeedbackCreate(BaseModel):
    """Toplu geri bildirim göndermek için model."""
    feedbacks: List[FeedbackCreate]

class FeedbackStats(BaseModel):
    """Geri bildirim istatistiklerini temsil eden model."""
    total_feedbacks: int
    positive_feedbacks: int
    negative_feedbacks: int
    by_type: dict[FeedbackType, int]
