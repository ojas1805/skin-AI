import cv2
import numpy as np
import math

class SkinToneAnalyzer:

    def get_average_color(self, roi):
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        avg = np.mean(lab.reshape(-1, 3), axis=0)
        return avg

    def calculate_ita(self, L, b):
        ita = math.degrees(math.atan((L - 50) / (b + 0.001)))
        return round(ita, 2)

    def ita_to_fitzpatrick(self, ita):
        if   ita > 55:  return {"type": "I",   "description": "Very fair"}
        elif ita > 41:  return {"type": "II",  "description": "Fair"}
        elif ita > 28:  return {"type": "III", "description": "Medium"}
        elif ita > 10:  return {"type": "IV",  "description": "Olive / light brown"}
        elif ita > -30: return {"type": "V",   "description": "Brown"}
        else:           return {"type": "VI",  "description": "Dark brown / black"}

    def get_undertone(self, a, b):
        if b > 20 and a > 10:  return "warm"
        elif b < 10:           return "cool"
        else:                  return "neutral"

    def analyze(self, rois):
        all_lab = []
        for roi in rois:
            lab = self.get_average_color(roi)
            all_lab.append(lab)

        avg_lab = np.mean(all_lab, axis=0)
        L, a, b = avg_lab

        ita         = self.calculate_ita(L, b)
        fitzpatrick = self.ita_to_fitzpatrick(ita)
        undertone   = self.get_undertone(a, b)

        return {
            "ita_angle":   ita,
            "fitzpatrick": fitzpatrick["type"],
            "description": fitzpatrick["description"],
            "undertone":   undertone,
            "lab_values":  {"L": round(float(L),2), "a": round(float(a),2), "b": round(float(b),2)}
        }