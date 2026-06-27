import cv2
import numpy as np
from ultralytics import YOLO
import os

class SkinConditionAnalyzer:

    def __init__(self):
        model_path = "best.pt"
        if os.path.exists(model_path):
            self.acne_model = YOLO(model_path)
            print("✓ YOLOv8 acne model loaded")
        else:
            self.acne_model = None
            print("⚠ No acne model found, using CV fallback")

    def detect_acne_yolo(self, image):
        """Detect acne using YOLOv8"""
        if self.acne_model is None:
            return 0, []

        results = self.acne_model(image, verbose=False)
        boxes = results[0].boxes

        acne_count = len(boxes)
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 2)
            })

        return acne_count, detections

    def analyze_zone(self, roi, zone_name):
        if roi is None or roi.size == 0:
            return {}

        results = {}
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # --- ACNE via YOLOv8 ---
        acne_count, detections = self.detect_acne_yolo(roi)
        acne_score = min(10, acne_count * 1.5)
        results["acne"] = {
            "score":     round(float(acne_score), 1),
            "count":     acne_count,
            "severity":  self.score_to_severity(acne_score),
            "detections": detections
        }

        # --- PORES ---
        laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
        pore_score = min(10, float(np.var(laplacian)) / 100)
        results["pores"] = {
            "score":    round(pore_score, 1),
            "severity": self.score_to_severity(pore_score)
        }

        # --- OILINESS ---
        hsv         = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel   = hsv[:, :, 2]
        bright_pixels = np.sum(v_channel > 200)
        oiliness_ratio = bright_pixels / v_channel.size
        oiliness_score = min(10, float(oiliness_ratio) * 30)
        results["oiliness"] = {
            "score":    round(oiliness_score, 1),
            "severity": self.score_to_severity(oiliness_score)
        }

        # --- PIGMENTATION ---
        lab         = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel   = lab[:, :, 0]
        pig_score   = min(10, float(np.std(l_channel)) / 5)
        results["pigmentation"] = {
            "score":    round(pig_score, 1),
            "severity": self.score_to_severity(pig_score)
        }

        # --- DRYNESS ---
        s_channel   = hsv[:, :, 1]
        avg_sat     = float(np.mean(s_channel))
        dry_score   = min(10, max(0, (80 - avg_sat) / 8))
        results["dryness"] = {
            "score":    round(dry_score, 1),
            "severity": self.score_to_severity(dry_score)
        }

        return results

    def score_to_severity(self, score):
        if score < 3:   return "low"
        elif score < 6: return "moderate"
        else:           return "high"

    def extract_zones(self, image, landmarks):
        h, w = image.shape[:2]
        zones = {}

        zone_definitions = {
            "forehead":    [10, 338, 297, 332, 284, 251, 389, 356,
                            454, 323, 361, 288, 397, 365, 379, 378,
                            400, 377, 152, 148, 176, 149, 150, 136,
                            172, 58, 132, 93, 234, 127, 162, 21,
                            54, 103, 67, 109],
            "left_cheek":  [50, 101, 118, 117, 116, 123, 147, 213],
            "right_cheek": [280, 330, 347, 346, 345, 352, 376, 433],
            "nose":        [1, 2, 5, 4, 19, 94, 2, 164],
            "chin":        [152, 377, 400, 378, 379, 365, 397, 288]
        }

        for zone_name, indices in zone_definitions.items():
            try:
                pts = np.array([landmarks[i] for i in indices
                               if i in landmarks])
                if len(pts) < 3:
                    continue
                x, y, zw, zh = cv2.boundingRect(pts)
                pad = 8
                x  = max(0, x - pad)
                y  = max(0, y - pad)
                x2 = min(w, x + zw + pad*2)
                y2 = min(h, y + zh + pad*2)
                roi = image[y:y2, x:x2]
                if roi.size > 0:
                    zones[zone_name] = roi
            except Exception:
                continue

        return zones

    def analyze(self, image, landmarks):
        zones      = self.extract_zones(image, landmarks)
        all_results = {}

        for zone_name, roi in zones.items():
            all_results[zone_name] = self.analyze_zone(roi, zone_name)

        conditions = ["acne", "pores", "oiliness", "pigmentation", "dryness"]
        overall    = {}

        for condition in conditions:
            scores = []
            for zone in all_results.values():
                if condition in zone:
                    scores.append(zone[condition]["score"])
            if scores:
                avg = round(sum(scores) / len(scores), 1)
                overall[condition] = {
                    "score":    avg,
                    "severity": self.score_to_severity(avg)
                }

        return {
            "overall": overall,
            "by_zone": all_results
        }