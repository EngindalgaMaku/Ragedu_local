/**
 * Adaptive Assessment Service (ADIM 3)
 *
 * Manages adaptive triggering logic for progressive assessment flow:
 * - Automatic follow-up triggers based on emoji scores
 * - Deep analysis triggers based on confidence levels
 * - Smart timing management with delays
 * - Contextual recommendations
 *
 * This service implements the intelligence behind the progressive assessment system.
 */

interface TriggerConditions {
  emojiScore: number;
  emojiType: string;
  confidenceLevel?: number;
  hasQuestions?: boolean;
  applicationScore?: number;
  userHistory?: UserAssessmentHistory;
}

interface UserAssessmentHistory {
  totalAssessments: number;
  averageConfidence: number;
  recentTrend: "improving" | "declining" | "stable";
  commonWeakAreas: string[];
  interventionHistory: number;
}

interface TriggerDecision {
  shouldTriggerFollowUp: boolean;
  shouldTriggerDeepAnalysis: boolean;
  delaySeconds: number;
  priority: "low" | "medium" | "high" | "urgent";
  reasons: string[];
  recommendations: string[];
}

interface AssessmentContext {
  topicComplexity?: "basic" | "intermediate" | "advanced";
  studentLevel?: "beginner" | "intermediate" | "advanced";
  timeSpent?: number; // seconds
  previousAttempts?: number;
}

export class AdaptiveAssessmentService {
  private static instance: AdaptiveAssessmentService;
  private readonly EMOJI_THRESHOLDS = {
    VERY_LOW: 0.0, // ❌
    LOW: 0.2, // 😐
    MODERATE: 0.7, // 😊
    HIGH: 1.0, // 👍
  };

  private readonly CONFIDENCE_THRESHOLDS = {
    CRITICAL: 2,
    LOW: 3,
    MODERATE: 4,
    HIGH: 5,
  };

  private constructor() {}

  static getInstance(): AdaptiveAssessmentService {
    if (!AdaptiveAssessmentService.instance) {
      AdaptiveAssessmentService.instance = new AdaptiveAssessmentService();
    }
    return AdaptiveAssessmentService.instance;
  }

  /**
   * Main decision engine for adaptive triggering
   */
  evaluateTriggerConditions(
    conditions: TriggerConditions,
    context?: AssessmentContext
  ): TriggerDecision {
    const decision: TriggerDecision = {
      shouldTriggerFollowUp: false,
      shouldTriggerDeepAnalysis: false,
      delaySeconds: 30,
      priority: "low",
      reasons: [],
      recommendations: [],
    };

    // 1. Initial emoji-based triggers
    this.evaluateEmojiTriggers(conditions, decision);

    // 2. Confidence-based triggers (if available)
    if (conditions.confidenceLevel !== undefined) {
      this.evaluateConfidenceTriggers(conditions, decision);
    }

    // 3. Context-based adjustments
    if (context) {
      this.applyContextualAdjustments(context, decision);
    }

    // 4. Historical pattern analysis
    if (conditions.userHistory) {
      this.evaluateHistoricalPatterns(conditions.userHistory, decision);
    }

    // 5. Generate recommendations
    this.generateRecommendations(conditions, decision, context);

    // 6. Adjust priority based on multiple factors
    this.calculatePriority(conditions, decision);

    return decision;
  }

  private evaluateEmojiTriggers(
    conditions: TriggerConditions,
    decision: TriggerDecision
  ): void {
    const { emojiScore, emojiType } = conditions;

    if (emojiScore <= this.EMOJI_THRESHOLDS.VERY_LOW) {
      // ❌ - Critical case
      decision.shouldTriggerFollowUp = true;
      decision.shouldTriggerDeepAnalysis = true;
      decision.delaySeconds = 15; // Faster response for critical cases
      decision.reasons.push("Çok düşük emoji puanı (❌)");
      decision.reasons.push("Öğrenci konuyu anlamadı");
    } else if (emojiScore <= this.EMOJI_THRESHOLDS.LOW) {
      // 😐 - Confused case
      decision.shouldTriggerFollowUp = true;
      decision.delaySeconds = 20;
      decision.reasons.push("Düşük emoji puanı (😐)");
      decision.reasons.push("Karışıklık belirtileri");
    } else if (emojiScore >= this.EMOJI_THRESHOLDS.HIGH) {
      // 👍 - Excellent case, but still check for completeness
      decision.shouldTriggerFollowUp = true;
      decision.delaySeconds = 45; // Longer delay for good performance
      decision.reasons.push("Yüksek performans doğrulama");
    } else if (emojiScore >= this.EMOJI_THRESHOLDS.MODERATE) {
      // 😊 - Good case, optional follow-up
      decision.shouldTriggerFollowUp = true;
      decision.delaySeconds = 60;
      decision.reasons.push("Orta seviye performans takibi");
    }
  }

  private evaluateConfidenceTriggers(
    conditions: TriggerConditions,
    decision: TriggerDecision
  ): void {
    const confidence = conditions.confidenceLevel!;

    if (confidence <= this.CONFIDENCE_THRESHOLDS.CRITICAL) {
      decision.shouldTriggerDeepAnalysis = true;
      decision.delaySeconds = Math.min(decision.delaySeconds, 10);
      decision.reasons.push(`Kritik güven seviyesi (${confidence}/5)`);
    } else if (confidence <= this.CONFIDENCE_THRESHOLDS.LOW) {
      decision.shouldTriggerDeepAnalysis = true;
      decision.reasons.push(`Düşük güven seviyesi (${confidence}/5)`);
    }

    // Additional questions trigger
    if (conditions.hasQuestions) {
      decision.shouldTriggerDeepAnalysis = true;
      decision.reasons.push("Öğrencinin ek soruları var");
    }

    // Application understanding check
    if (
      conditions.applicationScore !== undefined &&
      conditions.applicationScore < 0.3
    ) {
      decision.shouldTriggerDeepAnalysis = true;
      decision.reasons.push("Uygulama anlayışı yetersiz");
    }
  }

  private applyContextualAdjustments(
    context: AssessmentContext,
    decision: TriggerDecision
  ): void {
    // Topic complexity adjustments
    if (context.topicComplexity === "advanced") {
      decision.shouldTriggerFollowUp = true;
      decision.delaySeconds = Math.max(decision.delaySeconds, 45);
      decision.reasons.push("İleri seviye konu - ek takip gerekli");
    } else if (context.topicComplexity === "basic") {
      decision.delaySeconds = Math.min(decision.delaySeconds, 20);
    }

    // Student level adjustments
    if (context.studentLevel === "beginner") {
      decision.shouldTriggerFollowUp = true;
      decision.reasons.push("Yeni başlayan öğrenci - ek destek");
    }

    // Time spent analysis
    if (context.timeSpent && context.timeSpent < 30) {
      decision.reasons.push("Çok hızlı tamamlama - anlayış kontrolü");
      decision.shouldTriggerFollowUp = true;
    } else if (context.timeSpent && context.timeSpent > 300) {
      decision.reasons.push("Uzun süre harcanmış - zorluk göstergesi");
      decision.shouldTriggerDeepAnalysis = true;
    }

    // Previous attempts
    if (context.previousAttempts && context.previousAttempts > 1) {
      decision.shouldTriggerDeepAnalysis = true;
      decision.reasons.push("Tekrarlanan girişimler - derinlemesine analiz");
    }
  }

  private evaluateHistoricalPatterns(
    history: UserAssessmentHistory,
    decision: TriggerDecision
  ): void {
    // Declining trend
    if (history.recentTrend === "declining") {
      decision.shouldTriggerFollowUp = true;
      decision.shouldTriggerDeepAnalysis = true;
      decision.reasons.push("Düşen performans trendi");
    }

    // Low average confidence
    if (history.averageConfidence < 2.5) {
      decision.shouldTriggerDeepAnalysis = true;
      decision.reasons.push("Genel düşük güven seviyesi");
    }

    // High intervention history
    if (history.interventionHistory > 3) {
      decision.shouldTriggerDeepAnalysis = true;
      decision.delaySeconds = Math.min(decision.delaySeconds, 15);
      decision.reasons.push("Sık müdahale geçmişi");
    }

    // Common weak areas
    if (history.commonWeakAreas.length > 2) {
      decision.shouldTriggerDeepAnalysis = true;
      decision.reasons.push(
        `${history.commonWeakAreas.length} yaygın zayıf alan`
      );
    }
  }

  private generateRecommendations(
    conditions: TriggerConditions,
    decision: TriggerDecision,
    context?: AssessmentContext
  ): void {
    const recommendations: string[] = [];

    // Based on emoji feedback
    if (conditions.emojiScore <= this.EMOJI_THRESHOLDS.VERY_LOW) {
      recommendations.push("Alternatif açıklama yöntemleri dene");
      recommendations.push("Bireysel öğretmen desteği al");
      recommendations.push("Temel kavramları tekrar et");
    } else if (conditions.emojiScore <= this.EMOJI_THRESHOLDS.LOW) {
      recommendations.push("Ek örnekler ve açıklamalar iste");
      recommendations.push("Karışık noktaları belirle ve netleştir");
    } else if (conditions.emojiScore >= this.EMOJI_THRESHOLDS.HIGH) {
      recommendations.push("İleri seviye konulara geç");
      recommendations.push("Bilgini başkalarıyla paylaş");
      recommendations.push("Pratik uygulamalar yap");
    }

    // Based on confidence level
    if (conditions.confidenceLevel && conditions.confidenceLevel <= 2) {
      recommendations.push("Temel kavramları pekiştir");
      recommendations.push("Adım adım rehberlik al");
      recommendations.push("Sık sık alıştırma yap");
    } else if (conditions.confidenceLevel && conditions.confidenceLevel >= 4) {
      recommendations.push("Kendini test et");
      recommendations.push("Başkalarına öğret");
      recommendations.push("Yaratıcı uygulamalar geliştir");
    }

    // Based on context
    if (context?.topicComplexity === "advanced") {
      recommendations.push("Konuyu parçalara böl");
      recommendations.push("Görsel materyaller kullan");
    }

    // Based on questions
    if (conditions.hasQuestions) {
      recommendations.push("Sorularını sormaktan çekinme");
      recommendations.push("Grup çalışmasına katıl");
    }

    decision.recommendations = recommendations;
  }

  private calculatePriority(
    conditions: TriggerConditions,
    decision: TriggerDecision
  ): void {
    let priorityScore = 0;

    // Emoji-based priority
    if (conditions.emojiScore <= this.EMOJI_THRESHOLDS.VERY_LOW) {
      priorityScore += 4;
    } else if (conditions.emojiScore <= this.EMOJI_THRESHOLDS.LOW) {
      priorityScore += 3;
    } else if (conditions.emojiScore >= this.EMOJI_THRESHOLDS.HIGH) {
      priorityScore += 1;
    }

    // Confidence-based priority
    if (conditions.confidenceLevel) {
      if (conditions.confidenceLevel <= 2) {
        priorityScore += 3;
      } else if (conditions.confidenceLevel <= 3) {
        priorityScore += 2;
      }
    }

    // Questions boost priority
    if (conditions.hasQuestions) {
      priorityScore += 1;
    }

    // Historical patterns
    if (conditions.userHistory?.recentTrend === "declining") {
      priorityScore += 2;
    }

    if (
      conditions.userHistory?.interventionHistory &&
      conditions.userHistory.interventionHistory > 3
    ) {
      priorityScore += 2;
    }

    // Set priority based on score
    if (priorityScore >= 6) {
      decision.priority = "urgent";
    } else if (priorityScore >= 4) {
      decision.priority = "high";
    } else if (priorityScore >= 2) {
      decision.priority = "medium";
    } else {
      decision.priority = "low";
    }
  }

  /**
   * Utility method to evaluate application understanding quality
   */
  evaluateApplicationUnderstanding(text: string): number {
    if (!text || text.trim().length < 10) return 0.0;

    const qualityIndicators = [
      /örnek/i,
      /uygula/i,
      /kullan/i,
      /yap/i,
      /gerçek/i,
      /pratik/i,
      /hayat/i,
      /durumda/i,
      /çünkü/i,
      /nasıl/i,
    ];

    const uncertaintyWords = [
      /bilmiyorum/i,
      /emin değilim/i,
      /sanırım/i,
      /belki/i,
      /karışık/i,
      /zor/i,
    ];

    let score = 0.5; // Base score

    // Positive indicators
    const positiveMatches = qualityIndicators.filter((pattern) =>
      pattern.test(text)
    ).length;
    score += positiveMatches * 0.1;

    // Negative indicators
    const negativeMatches = uncertaintyWords.filter((pattern) =>
      pattern.test(text)
    ).length;
    score -= negativeMatches * 0.15;

    // Length factor
    if (text.length > 100) score += 0.1;
    if (text.length > 200) score += 0.1;

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Generate contextual timing for triggers
   */
  calculateOptimalDelay(
    conditions: TriggerConditions,
    priority: string
  ): number {
    let baseDelay = 30; // Default 30 seconds

    switch (priority) {
      case "urgent":
        baseDelay = 10;
        break;
      case "high":
        baseDelay = 15;
        break;
      case "medium":
        baseDelay = 30;
        break;
      case "low":
        baseDelay = 60;
        break;
    }

    // Adjust based on emoji score
    if (conditions.emojiScore >= this.EMOJI_THRESHOLDS.HIGH) {
      baseDelay += 15; // Give more time for successful students
    }

    return baseDelay;
  }

  /**
   * Check if student needs immediate intervention
   */
  needsImmediateIntervention(conditions: TriggerConditions): boolean {
    const criticalIndicators = [
      conditions.emojiScore <= this.EMOJI_THRESHOLDS.VERY_LOW,
      conditions.confidenceLevel !== undefined &&
        conditions.confidenceLevel <= 1,
      conditions.userHistory?.recentTrend === "declining" &&
        conditions.userHistory?.averageConfidence < 2,
      conditions.userHistory?.interventionHistory &&
        conditions.userHistory.interventionHistory > 5,
    ];

    return criticalIndicators.filter(Boolean).length >= 2;
  }

  /**
   * Generate intervention recommendations
   */
  generateInterventionPlan(conditions: TriggerConditions): string[] {
    const plan: string[] = [];

    if (this.needsImmediateIntervention(conditions)) {
      plan.push("Öğretmen ile birebir görüşme planla");
      plan.push("Temel kavramları yeniden öğren");
      plan.push("Ek pratik materyalleri kullan");
      plan.push("Küçük adımlarla ilerle");
      plan.push("Düzenli takip yap");
    } else if (conditions.emojiScore <= this.EMOJI_THRESHOLDS.LOW) {
      plan.push("Konuyu farklı kaynaklardan öğren");
      plan.push("Grup çalışmasına katıl");
      plan.push("Ek örnekler iste");
    } else {
      plan.push("Mevcut performansını sürdür");
      plan.push("İlerlemeni takip et");
    }

    return plan;
  }
}

// Export singleton instance
export const adaptiveAssessment = AdaptiveAssessmentService.getInstance();
