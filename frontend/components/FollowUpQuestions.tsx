/**
 * Follow-Up Questions Component (ADIM 3 - Stage 2)
 *
 * Handles the second stage of progressive assessment:
 * - Confidence level check (1-5 scale)
 * - Additional questions inquiry
 * - Application understanding evaluation
 *
 * This component is shown 30 seconds after initial emoji feedback.
 */

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Loader2, MessageSquare, Target, Brain } from "lucide-react";

interface FollowUpQuestionsProps {
  onSubmit: (data: FollowUpData) => void;
  loading: boolean;
  interactionId: number;
}

interface FollowUpData {
  has_questions: boolean;
  confidence_level: number;
  application_understanding: string;
  comment?: string;
}

const FollowUpQuestions: React.FC<FollowUpQuestionsProps> = ({
  onSubmit,
  loading,
  interactionId,
}) => {
  const [formData, setFormData] = useState<FollowUpData>({
    has_questions: false,
    confidence_level: 3,
    application_understanding: "",
    comment: "",
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (formData.confidence_level < 1 || formData.confidence_level > 5) {
      newErrors.confidence_level = "Güven seviyesi 1-5 arasında olmalıdır";
    }

    if (!formData.application_understanding.trim()) {
      newErrors.application_understanding = "Bu alan zorunludur";
    } else if (formData.application_understanding.length < 10) {
      newErrors.application_understanding = "En az 10 karakter girmelisiniz";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    onSubmit(formData);
  };

  const confidenceLevels = [
    {
      value: 1,
      label: "1 - Çok az güveniyorum",
      description: "Konuyu hiç anlamadım",
    },
    {
      value: 2,
      label: "2 - Az güveniyorum",
      description: "Birçok noktada karışık",
    },
    {
      value: 3,
      label: "3 - Orta düzeyde güveniyorum",
      description: "Temel noktaları anladım",
    },
    {
      value: 4,
      label: "4 - Çok güveniyorum",
      description: "Konuyu iyi anladım",
    },
    {
      value: 5,
      label: "5 - Tamamen güveniyorum",
      description: "Konuyu mükemmel anladım",
    },
  ];

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <MessageSquare className="h-5 w-5" />
          <span>Takip Değerlendirmesi</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Has Questions */}
          <div className="space-y-3">
            <Label className="text-base font-medium">Başka soru var mı?</Label>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="has-questions"
                checked={formData.has_questions}
                onCheckedChange={(checked) =>
                  setFormData((prev) => ({ ...prev, has_questions: !!checked }))
                }
              />
              <Label
                htmlFor="has-questions"
                className="text-sm font-normal cursor-pointer"
              >
                Evet, bu konu hakkında daha fazla şey öğrenmek istiyorum
              </Label>
            </div>
          </div>

          {/* Confidence Level */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4" />
              <Label className="text-base font-medium">
                Bu konuda kendini ne kadar güvende hissediyorsun? (1-5)
              </Label>
            </div>
            <RadioGroup
              value={formData.confidence_level.toString()}
              onValueChange={(value) =>
                setFormData((prev) => ({
                  ...prev,
                  confidence_level: parseInt(value),
                }))
              }
              className="space-y-3"
            >
              {confidenceLevels.map((level) => (
                <div
                  key={level.value}
                  className="flex items-start space-x-3 p-3 rounded-lg border hover:bg-gray-50"
                >
                  <RadioGroupItem
                    value={level.value.toString()}
                    id={`confidence-${level.value}`}
                    className="mt-0.5"
                  />
                  <div className="flex-1">
                    <Label
                      htmlFor={`confidence-${level.value}`}
                      className="cursor-pointer"
                    >
                      <div className="font-medium">{level.label}</div>
                      <div className="text-sm text-gray-600 mt-1">
                        {level.description}
                      </div>
                    </Label>
                  </div>
                </div>
              ))}
            </RadioGroup>
            {errors.confidence_level && (
              <p className="text-sm text-red-600">{errors.confidence_level}</p>
            )}
          </div>

          {/* Application Understanding */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <Brain className="h-4 w-4" />
              <Label htmlFor="application" className="text-base font-medium">
                Bu bilgiyi gerçek hayatta nasıl kullanırsın?
              </Label>
            </div>
            <Textarea
              id="application"
              placeholder="Bu konuyu günlük hayatımda nasıl uygulayabilirim? Hangi durumlarda işime yarar? Örnekler verebilir misin..."
              value={formData.application_understanding}
              onChange={(e) =>
                setFormData((prev) => ({
                  ...prev,
                  application_understanding: e.target.value,
                }))
              }
              className="min-h-[100px] resize-none"
              maxLength={1000}
            />
            <div className="flex justify-between text-sm text-gray-500">
              <span>
                {formData.application_understanding.length}/1000 karakter
              </span>
              {errors.application_understanding && (
                <span className="text-red-600">
                  {errors.application_understanding}
                </span>
              )}
            </div>
          </div>

          {/* Optional Comment */}
          <div className="space-y-3">
            <Label htmlFor="comment" className="text-base font-medium">
              Ek yorumlarınız (İsteğe bağlı)
            </Label>
            <Textarea
              id="comment"
              placeholder="Bu konuyla ilgili başka düşünceleriniz var mı?"
              value={formData.comment}
              onChange={(e) =>
                setFormData((prev) => ({ ...prev, comment: e.target.value }))
              }
              className="min-h-[80px] resize-none"
              maxLength={500}
            />
            <div className="text-sm text-gray-500">
              {formData.comment?.length || 0}/500 karakter
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end pt-4">
            <Button type="submit" disabled={loading} className="min-w-[120px]">
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Gönderiliyor...
                </>
              ) : (
                "Değerlendirmeyi Gönder"
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};

export default FollowUpQuestions;
