from src.face_detector import FaceDetector
from src.skin_tone import SkinToneAnalyzer
from src.skin_condition import SkinConditionAnalyzer
from src.recommender import SkinRecommender

def analyze_image(image_path):
    detector           = FaceDetector()
    tone_analyzer      = SkinToneAnalyzer()
    condition_analyzer = SkinConditionAnalyzer()
    recommender        = SkinRecommender()

    print(f"\nAnalyzing: {image_path}")

    image, landmarks = detector.get_landmarks(image_path)

    if landmarks is None:
        print("No face detected.")
        return

    # Phase 1: Skin tone
    rois = detector.get_cheek_roi(image, landmarks)
    tone = tone_analyzer.analyze(rois)

    # Phase 2: Skin conditions
    conditions = condition_analyzer.analyze(image, landmarks)

    # Phase 3: Recommendations
    skin_profile = {"tone": tone, "conditions": conditions}

    try:
        recommendations = recommender.recommend(skin_profile)
    except Exception as e:
        print(f"\nRecommender error: {e}")
        recommendations = {}

    # Print tone
    print("\n===== SKIN TONE =====")
    print(f"Fitzpatrick : {tone['fitzpatrick']} ({tone['description']})")
    print(f"Undertone   : {tone['undertone']}")

    # Print conditions
    print("\n===== CONDITIONS =====")
    for condition, data in conditions["overall"].items():
        bar = "█" * int(data["score"]) + "░" * (10 - int(data["score"]))
        print(f"{condition:<15} {bar} {data['score']}/10")

    # Print recommendations
    print("\n===== AI RECOMMENDATION =====")
    if "summary" in recommendations:
        print(f"\nSummary: {recommendations['summary']}")

        print("\n✓ PRODUCTS TO USE:")
        for item in recommendations["use"]:
            print(f"  • {item['product']} ({item['when']})")
            print(f"    → {item['reason']}")

        print("\n✗ PRODUCTS TO AVOID:")
        for item in recommendations["avoid"]:
            print(f"  • {item['product']}")
            print(f"    → {item['reason']}")

        print("\n📋 YOUR ROUTINE:")
        print("  Morning:")
        for step in recommendations["routine"]["morning"]:
            print(f"    {step}")
        print("  Night:")
        for step in recommendations["routine"]["night"]:
            print(f"    {step}")
    else:
        print(f"No recommendations: {recommendations}")

if __name__ == "__main__":
    analyze_image("test_images/your_photo.jpg")