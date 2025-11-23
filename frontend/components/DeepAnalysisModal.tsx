/**
 * Deep Analysis Modal Component (ADIM 3 - Stage 3)
 *
 * Handles the third stage of progressive assessment:
 * - Confusion areas identification
 * - Concept mapping requests
 * - Alternative explanation requests
 * - Related topic exploration
 *
 * This modal is triggered when students need deeper analysis based on low scores.
 */

import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Loader2, Brain, AlertCircle, Plus, X } from "lucide-react";

interface DeepAnalysisModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: DeepAnalysisData) => void;
  loading: boolean;
  interactionId: number;
}

interface DeepAnalysisData {
  confusion_areas: string[];
  requested_topics?: string[];
  alternative_explanation_request?: string;
  comment?: string;
}

const DeepAnalysisModal: React.FC<DeepAnalysisModalProps> = ({
  open,
  onClose,
  onSubmit,
  loading,
  interactionId,
}) => {
  const [formData, setFormData] = useState<DeepAnalysisData>({
    confusion_areas: [],
    requested_topics: [],
    alternative_explanation_request: "",
    comment: "",
  });

  const [newConfusionArea, setNewConfusionArea] = useState("");
  const [newRequestedTopic, setNewRequestedTopic] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Predefined common confusion areas
  const commonConfusionAreas = [
    "Temel kavramları anlamadım",
    "Örnekler yetersiz",
    "Adımlar çok hızlı",
    "Matematiksel formüller karışık",
    "Pratik uygulama belirsiz",
    "İlişkiler net değil",
    "Terminoloji zor",
  ];

  // Predefined related topics
  const relatedTopics = [
    "Temel konular",
    "İleri seviye uygulamalar",
    "Pratik örnekler",
    "Benzer kavramlar",
    "Alternatif yaklaşımlar",
    "Gerçek hayat uygulamaları",
  ];

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (formData.confusion_areas.length === 0) {
      newErrors.confusion_areas = "En az bir karışıklık alanı seçmelisiniz";
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

  const handleConfusionAreaToggle = (area: string, checked: boolean) => {
    setFormData((prev) => ({
      ...prev,
      confusion_areas: checked
        ? [...prev.confusion_areas, area]
        : prev.confusion_areas.filter((a) => a !== area),
    }));
  };

  const addCustomConfusionArea = () => {
    if (
      newConfusionArea.trim() &&
      !formData.confusion_areas.includes(newConfusionArea.trim())
    ) {
      setFormData((prev) => ({
        ...prev,
        confusion_areas: [...prev.confusion_areas, newConfusionArea.trim()],
      }));
      setNewConfusionArea("");
    }
  };

  const removeConfusionArea = (area: string) => {
    setFormData((prev) => ({
      ...prev,
      confusion_areas: prev.confusion_areas.filter((a) => a !== area),
    }));
  };

  const handleTopicToggle = (topic: string, checked: boolean) => {
    setFormData((prev) => ({
      ...prev,
      requested_topics: checked
        ? [...(prev.requested_topics || []), topic]
        : (prev.requested_topics || []).filter((t) => t !== topic),
    }));
  };

  const addCustomTopic = () => {
    if (
      newRequestedTopic.trim() &&
      !(formData.requested_topics || []).includes(newRequestedTopic.trim())
    ) {
      setFormData((prev) => ({
        ...prev,
        requested_topics: [
          ...(prev.requested_topics || []),
          newRequestedTopic.trim(),
        ],
      }));
      setNewRequestedTopic("");
    }
  };

  const removeTopic = (topic: string) => {
    setFormData((prev) => ({
      ...prev,
      requested_topics: (prev.requested_topics || []).filter(
        (t) => t !== topic
      ),
    }));
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>Detaylı Analiz Talebi</span>
          </DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Confusion Areas */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-4 w-4 text-orange-500" />
              <Label className="text-base font-medium">
                Hangi kısmı daha açık olmyabiliriz?
              </Label>
            </div>

            {/* Common confusion areas */}
            <div className="grid grid-cols-1 gap-2">
              {commonConfusionAreas.map((area) => (
                <div
                  key={area}
                  className="flex items-center space-x-2 p-2 rounded-lg border hover:bg-gray-50"
                >
                  <Checkbox
                    id={`confusion-${area}`}
                    checked={formData.confusion_areas.includes(area)}
                    onCheckedChange={(checked: any) =>
                      handleConfusionAreaToggle(area, !!checked)
                    }
                  />
                  <Label
                    htmlFor={`confusion-${area}`}
                    className="cursor-pointer text-sm"
                  >
                    {area}
                  </Label>
                </div>
              ))}
            </div>

            {/* Add custom confusion area */}
            <div className="flex items-center space-x-2">
              <Input
                placeholder="Başka bir karışıklık alanı ekle..."
                value={newConfusionArea}
                onChange={(e: any) => setNewConfusionArea(e.target.value)}
                onKeyPress={(e: any) =>
                  e.key === "Enter" &&
                  (e.preventDefault(), addCustomConfusionArea())
                }
                className="flex-1"
              />
              <Button
                type="button"
                onClick={addCustomConfusionArea}
                size="sm"
                variant="outline"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>

            {/* Selected confusion areas */}
            {formData.confusion_areas.length > 0 && (
              <div className="space-y-2">
                <Label className="text-sm font-medium">
                  Seçili karışıklık alanları:
                </Label>
                <div className="flex flex-wrap gap-2">
                  {formData.confusion_areas.map((area) => (
                    <Badge
                      key={area}
                      variant="secondary"
                      className="flex items-center gap-1"
                    >
                      {area}
                      <button
                        type="button"
                        onClick={() => removeConfusionArea(area)}
                        className="ml-1 hover:bg-gray-300 rounded-full p-0.5"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {errors.confusion_areas && (
              <p className="text-sm text-red-600">{errors.confusion_areas}</p>
            )}
          </div>

          {/* Related Topics */}
          <div className="space-y-4">
            <Label className="text-base font-medium">
              İlgili hangi konuları keşfetmek istersiniz? (İsteğe bağlı)
            </Label>

            {/* Common topics */}
            <div className="grid grid-cols-2 gap-2">
              {relatedTopics.map((topic) => (
                <div
                  key={topic}
                  className="flex items-center space-x-2 p-2 rounded-lg border hover:bg-gray-50"
                >
                  <Checkbox
                    id={`topic-${topic}`}
                    checked={(formData.requested_topics || []).includes(topic)}
                    onCheckedChange={(checked: any) =>
                      handleTopicToggle(topic, !!checked)
                    }
                  />
                  <Label
                    htmlFor={`topic-${topic}`}
                    className="cursor-pointer text-sm"
                  >
                    {topic}
                  </Label>
                </div>
              ))}
            </div>

            {/* Add custom topic */}
            <div className="flex items-center space-x-2">
              <Input
                placeholder="Başka bir konu ekle..."
                value={newRequestedTopic}
                onChange={(e: any) => setNewRequestedTopic(e.target.value)}
                onKeyPress={(e: any) =>
                  e.key === "Enter" && (e.preventDefault(), addCustomTopic())
                }
                className="flex-1"
              />
              <Button
                type="button"
                onClick={addCustomTopic}
                size="sm"
                variant="outline"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>

            {/* Selected topics */}
            {(formData.requested_topics || []).length > 0 && (
              <div className="space-y-2">
                <Label className="text-sm font-medium">Seçili konular:</Label>
                <div className="flex flex-wrap gap-2">
                  {(formData.requested_topics || []).map((topic) => (
                    <Badge
                      key={topic}
                      variant="outline"
                      className="flex items-center gap-1"
                    >
                      {topic}
                      <button
                        type="button"
                        onClick={() => removeTopic(topic)}
                        className="ml-1 hover:bg-gray-300 rounded-full p-0.5"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Alternative Explanation Request */}
          <div className="space-y-3">
            <Label htmlFor="alt-explanation" className="text-base font-medium">
              Farklı bir açıklama tarzı ister misiniz? (İsteğe bağlı)
            </Label>
            <Textarea
              id="alt-explanation"
              placeholder="Örneğin: Daha çok görsel örneklerle, daha basit terimlerle, adım adım, analojilerle..."
              value={formData.alternative_explanation_request}
              onChange={(e: any) =>
                setFormData((prev) => ({
                  ...prev,
                  alternative_explanation_request: e.target.value,
                }))
              }
              className="min-h-[80px] resize-none"
              maxLength={500}
            />
            <div className="text-sm text-gray-500">
              {formData.alternative_explanation_request?.length || 0}/500
              karakter
            </div>
          </div>

          {/* Additional Comments */}
          <div className="space-y-3">
            <Label htmlFor="deep-comment" className="text-base font-medium">
              Ek yorumlarınız (İsteğe bağlı)
            </Label>
            <Textarea
              id="deep-comment"
              placeholder="Bu konuyla ilgili başka belirtmek istediğiniz noktalar var mı?"
              value={formData.comment}
              onChange={(e: any) =>
                setFormData((prev) => ({ ...prev, comment: e.target.value }))
              }
              className="min-h-[80px] resize-none"
              maxLength={1000}
            />
            <div className="text-sm text-gray-500">
              {formData.comment?.length || 0}/1000 karakter
            </div>
          </div>
        </form>

        <DialogFooter className="flex justify-between">
          <Button
            type="button"
            variant="outline"
            onClick={onClose}
            disabled={loading}
          >
            İptal
          </Button>
          <Button type="submit" onClick={handleSubmit} disabled={loading}>
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Gönderiliyor...
              </>
            ) : (
              "Detaylı Analiz Talep Et"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default DeepAnalysisModal;
