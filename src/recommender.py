import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import json
from src.ingredients_db import INGREDIENTS

class SkinRecommender:

    def __init__(self):
        self.client = Groq()

    def build_prompt(self, skin_profile):
        fitzpatrick = skin_profile["tone"]["fitzpatrick"]
        description = skin_profile["tone"]["description"]
        undertone   = skin_profile["tone"]["undertone"]
        overall     = skin_profile["conditions"]["overall"]

        conditions_text = ""
        for condition, data in overall.items():
            conditions_text += f"- {condition}: {data['score']}/10 ({data['severity']})\n"

        ingredients_text = ""
        for ing in INGREDIENTS:
            ingredients_text += f"""
Ingredient: {ing['name']}
Benefits: {', '.join(ing['benefits'])}
Good for: {', '.join(ing['good_for'])}
Avoid for: {', '.join(ing['avoid_for'])}
Safe for Fitzpatrick: {', '.join(ing['fitzpatrick_safe']) if ing['fitzpatrick_safe'] else 'None'}
---"""

        prompt = f"""You are an expert dermatologist specializing in South Asian and diverse skin tones.

PATIENT SKIN PROFILE:
- Fitzpatrick Type: {fitzpatrick} ({description})
- Undertone: {undertone}
- Skin Conditions:
{conditions_text}

AVAILABLE INGREDIENTS AND PRODUCTS:
{ingredients_text}

Based on this skin profile, provide personalized skincare recommendations.

Respond in this exact JSON format with no extra text before or after:
{{
  "summary": "2-3 sentence plain English summary of this person's skin",
  "use": [
    {{
      "product": "product name",
      "reason": "specific reason based on their skin scores",
      "when": "morning or night or both"
    }}
  ],
  "avoid": [
    {{
      "product": "product name",
      "reason": "specific reason why it is bad for their skin"
    }}
  ],
  "routine": {{
    "morning": ["step 1", "step 2", "step 3"],
    "night": ["step 1", "step 2", "step 3"]
  }}
}}"""

        return prompt

    def recommend(self, skin_profile):
        prompt = self.build_prompt(skin_profile)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )

        response_text = response.choices[0].message.content

        try:
            start    = response_text.find("{")
            end      = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            return json.loads(json_str)
        except Exception:
            return {"raw": response_text}