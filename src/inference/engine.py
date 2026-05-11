from vllm import LLM, SamplingParams


class VLLMEngine:
    def __init__(
        self,
        model_name="mistralai/Mistral-7B-v0.1",
        kv_events_endpoint="tcp://*:5557",
        kv_events_topic="kv-events",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        num_gpu_blocks_override=None,
        max_model_len=None,
    ):
        kv_events_config = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": kv_events_endpoint,
            "topic": kv_events_topic,
        }

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # increase if multi-GPU
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            num_gpu_blocks_override=num_gpu_blocks_override,
            max_model_len=max_model_len,
            kv_events_config=kv_events_config,
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