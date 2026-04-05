"""
Stage 4 - Step 0 验收测试: 验证保存的 RoBERTa 模型可正确加载并推理

验收标准（来自 stage4_plan.md）:
  - 能用固定路径加载模型
  - 任意输入文本可返回预测结果
  - 本脚本可独立运行，不依赖其他 Stage 4 模块

Usage:
    python s4_agent/test_classifier_load.py
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "s4_agent" / "artifacts" / "roberta_5class_best"

# ---------------------------------------------------------------------------
# Test inputs: cover all 5 star ratings
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "text": "Absolutely disgusting. The food was cold, the staff was rude, and the place was filthy. Never coming back.",
        "expected_range": (1, 2),
    },
    {
        "text": "Pretty mediocre experience. Nothing was terrible but nothing was great either. The food was okay.",
        "expected_range": (2, 3),
    },
    {
        "text": "Decent place for a quick lunch. Service was fine. I've had better but it's acceptable.",
        "expected_range": (3, 4),
    },
    {
        "text": "Really enjoyed this place! Great food, friendly staff, and very reasonable prices. Will definitely return.",
        "expected_range": (4, 5),
    },
    {
        "text": "Exceptional in every way. The chef came out personally, the tasting menu was flawless, and the ambiance was perfect.",
        "expected_range": (5, 5),
    },
]


def load_model(model_dir: Path):
    """Load tokenizer and model from the saved artifacts directory."""
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Please run step0_train_and_save.py first.")
        sys.exit(1)

    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    with open(model_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Model loaded on {device.upper()}")
    print(f"Val accuracy (from training): {label_map.get('val_accuracy', 'N/A')}")
    print(f"Val macro_f1 (from training): {label_map.get('val_macro_f1', 'N/A')}")
    return tokenizer, model, label_map, device


def predict(text: str, tokenizer, model, device: str, max_length: int = 512) -> dict:
    """
    Run inference on a single text input.

    Returns:
        {
            "predicted_stars": 4,
            "confidence": 0.87,
            "score_distribution": {
                "1_star": 0.01,
                "2_star": 0.03,
                "3_star": 0.09,
                "4_star": 0.87,
                "5_star": 0.00
            }
        }
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
    predicted_idx = int(torch.argmax(logits, dim=-1).item())
    predicted_stars = predicted_idx + 1  # labels are 0-indexed, stars are 1-5

    score_distribution = {
        f"{i+1}_star": round(probs[i], 4) for i in range(len(probs))
    }

    return {
        "predicted_stars": predicted_stars,
        "confidence": round(probs[predicted_idx], 4),
        "score_distribution": score_distribution,
    }


def main() -> None:
    print("=" * 60)
    print("Stage 4 - Step 0 Verification: RoBERTa Classifier Load Test")
    print("=" * 60)

    # 1. Load
    tokenizer, model, label_map, device = load_model(MODEL_DIR)

    # 2. Run predictions on test cases
    print("\n--- Inference Test ---")
    all_passed = True

    for i, case in enumerate(TEST_CASES, 1):
        result = predict(case["text"], tokenizer, model, device)
        predicted = result["predicted_stars"]
        lo, hi = case["expected_range"]
        passed = lo <= predicted <= hi

        status = "PASS" if passed else "WARN"
        if not passed:
            all_passed = False

        print(f"\n[{i}] {status}")
        print(f"  Text      : {case['text'][:80]}...")
        print(f"  Predicted : {predicted} stars (confidence: {result['confidence']:.0%})")
        print(f"  Expected  : {lo}~{hi} stars")
        print(f"  Distribution: { {k: f'{v:.0%}' for k, v in result['score_distribution'].items()} }")

    # 3. Final verdict
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED — model is ready for Stage 4 tool integration.")
    else:
        print("Some predictions were outside expected range.")
        print("This may be acceptable; check individual results above.")
        print("The model is still usable as long as predictions are reasonable.")

    print(f"\nModel path: {MODEL_DIR}")
    print("Next step: proceed to Step 1 (build_vectorstore.py)")


if __name__ == "__main__":
    main()
