from src.models.feedback import FeedbackCreate, BulkFeedbackCreate
from src.analytics.database import get_experiment_db
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackProcessor:
    """
    Geri bildirimleri işleyen, analiz eden ve ilgili sistemleri güncelleyen servis.
    """

    def __init__(self):
        """
        Servis başlatıldığında veritabanı bağlantısı gibi bağımlılıklar yüklenebilir.
        """
        logger.info("FeedbackProcessor başlatıldı.")
        self.db = get_experiment_db()

    def process_feedback(self, feedback: FeedbackCreate) -> dict:
        """
        Tek bir geri bildirimi işler ve veritabanına kaydeder.
        """
        logger.info(f"Geri bildirim işleniyor: {feedback.interaction_id}")
        
        try:
            feedback_id = self.db.add_feedback(feedback)
            # TODO: Belirsizlik skoru hesaplama ve öğrenci profili güncelleme mantığı eklenecek.
            return {"status": "success", "message": "Feedback processed successfully", "feedback_id": feedback_id}
        except Exception as e:
            logger.error(f"Geri bildirim işlenirken hata oluştu: {e}", exc_info=True)
            raise

    def process_bulk_feedback(self, bulk_feedback: BulkFeedbackCreate) -> dict:
        """
        Toplu geri bildirimleri işler.
        """
        count = len(bulk_feedback.feedbacks)
        logger.info(f"{count} adet geri bildirim toplu olarak işleniyor.")
        processed_ids = []
        
        for feedback in bulk_feedback.feedbacks:
            try:
                result = self.process_feedback(feedback)
                processed_ids.append(result.get("feedback_id"))
            except Exception as e:
                logger.warning(f"Toplu geri bildirim işlenirken bir geri bildirimde hata oluştu: {e}")
                # Hata durumunda bile devam et, ancak logla.
                continue
            
        return {"status": "success", "message": f"{len(processed_ids)} of {count} feedbacks processed", "processed_ids": processed_ids}

    def get_feedbacks_by_interaction(self, interaction_id: str) -> List[dict]:
        """
        Belirli bir etkileşime ait geri bildirimleri veritabanından getirir.
        """
        logger.info(f"'{interaction_id}' için geri bildirimler getiriliyor.")
        try:
            return self.db.get_feedbacks_by_interaction_id(interaction_id)
        except Exception as e:
            logger.error(f"'{interaction_id}' için geri bildirimler getirilirken hata oluştu: {e}", exc_info=True)
            raise

    def get_feedback_stats(self) -> dict:
        """
        Geri bildirim istatistiklerini veritabanından hesaplar.
        """
        logger.info("Geri bildirim istatistikleri hesaplanıyor.")
        try:
            return self.db.get_feedback_statistics()
        except Exception as e:
            logger.error(f"Geri bildirim istatistikleri hesaplanırken hata oluştu: {e}", exc_info=True)
            raise

# Servisin bir örneğini oluşturarak dışa aktarma (Singleton pattern)
feedback_processor = FeedbackProcessor()