"""
demo_cli.py — Command-Line Interface for Quick Inference
=========================================================
Allows testing predictions directly from the terminal without opening the browser.
Great for quick verification or showing your teacher a CLI demo.

Usage:
    python demo_cli.py
    python demo_cli.py "NASA discovered water on Mars"
    python demo_cli.py --model "Logistic Regression" "Lakers win NBA finals"
"""

import argparse
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.model import load_model, load_all_models
from src.data_loader import CATEGORY_EMOJI


def classify_text(text: str, model_name: str = "Linear SVM"):
    try:
        pipeline = load_model(model_name)
    except FileNotFoundError:
        print(f"Error: Model '{model_name}' not found. Please train models first using `python train_model.py`.")
        sys.exit(1)

    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    prob_dict = dict(zip(classes, probabilities))

    emoji = CATEGORY_EMOJI.get(prediction, "📄")

    print("\n" + "=" * 50)
    print("📰 News Article Classifier — CLI Demo")
    print("=" * 50)
    print(f"Input Text: \"{text.strip()[:100]}{'...' if len(text) > 100 else ''}\"")
    print(f"Model:      {model_name}")
    print("-" * 50)
    print(f"Predicted Category: {emoji} {prediction} ({prob_dict[prediction] * 100:.2f}% confidence)")
    print("-" * 50)
    print("All Probabilities:")
    for cat in sorted(classes, key=lambda c: prob_dict[c], reverse=True):
        bar = "█" * int(prob_dict[cat] * 30)
        print(f"  {CATEGORY_EMOJI.get(cat, ' ')} {cat:12s} {prob_dict[cat] * 100:6.2f}% | {bar}")
    print("=" * 50 + "\n")


def interactive_mode(model_name: str = "Linear SVM"):
    print("\n" + "=" * 50)
    print("📰 News Article Classifier — Interactive Mode")
    print(f"Active Model: {model_name}")
    print("Type an article snippet and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    try:
        pipeline = load_model(model_name)
    except FileNotFoundError:
        print(f"Error: Model '{model_name}' not found. Please train models first using `python train_model.py`.")
        return

    while True:
        try:
            user_input = input("\nEnter article text: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting CLI demo. Goodbye!")
                break
            classify_text(user_input, model_name=model_name)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Classify news articles from the command line.")
    parser.add_argument("text", nargs="?", default=None, help="Text snippet to classify")
    parser.add_argument(
        "--model",
        choices=["Naive Bayes", "Logistic Regression", "Linear SVM"],
        default="Linear SVM",
        help="Model to use (default: Linear SVM)"
    )
    args = parser.parse_args()

    if args.text:
        classify_text(args.text, model_name=args.model)
    else:
        interactive_mode(model_name=args.model)


if __name__ == "__main__":
    main()
