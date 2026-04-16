from src.inference.engine import VLLMEngine

if __name__ == "__main__":
    engine = VLLMEngine()

    prompts = [
        "Explain vLLM in one sentence:",
        "What is a KV cache in transformers?",
    ]

    outputs = engine.generate(prompts)

    for i, out in enumerate(outputs):
        print(f"\nPrompt {i}: {prompts[i]}")
        print(f"Output: {out}")