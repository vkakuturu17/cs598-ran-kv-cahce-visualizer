from vllm import LLM, SamplingParams


class VLLMEngine:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # increase if multi-GPU
        )

    def generate(self, prompts, max_tokens=50):
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            text = output.outputs[0].text
            results.append(text)

        return results