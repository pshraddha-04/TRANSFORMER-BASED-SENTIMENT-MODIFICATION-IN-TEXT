#!/usr/bin/env python
"""Quick test to verify the paraphrase model loads and works."""
from transformers import pipeline

print("Loading tuner007/pegasus_paraphrase model...")
try:
    p = pipeline('text2text-generation', model='tuner007/pegasus_paraphrase')
    print("Model loaded successfully")

    test_input = "I hate how slow the response is from support."
    print(f"\nTest input: {test_input}")

    result = p(test_input, max_new_tokens=80)
    output = result[0]["generated_text"]
    print(f"Paraphrased: {output}")
    print(f"\nModel works! Output is different from input: {output.lower() != test_input.lower()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

