LLAVA_MODELS = {
    "[LLaVA] LLaVA v1.6 Mistral 7B": ("llava", "llava-hf/llava-v1.6-mistral-7b-hf"),
    "[LLaVA] LLaVA v1.6 Vicuna 7B": ("llava", "llava-hf/llava-v1.6-vicuna-7b-hf"),
    "[LLaVA] LLaVA v1.6 Vicuna 13B": ("llava", "llava-hf/llava-v1.6-vicuna-13b-hf"),
    "[LLaVA] LLaVA v1.6 34B": ("llava", "llava-hf/llava-v1.6-34b-hf"),
    "[LLaVA] LLaMA3 LLaVA-NeXT 8B": ("llava", "llava-hf/llama3-llava-next-8b-hf"),
    "[LLaVA] LLaVA-NeXT 72B": ("llava", "llava-hf/llava-next-72b-hf"),
    "[LLaVA] LLaVA-NeXT 110B": ("llava", "llava-hf/llava-next-110b-hf"),
}

QWEN_MODELS = {
    "[Qwen]  Qwen2.5-VL 3B Instruct": ("qwen", "Qwen/Qwen2.5-VL-3B-Instruct"),
    "[Qwen]  Qwen2.5-VL 7B Instruct": ("qwen", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "[Qwen]  Qwen2.5-VL 32B Instruct": ("qwen", "Qwen/Qwen2.5-VL-32B-Instruct"),
    "[Qwen]  Qwen2.5-VL 72B Instruct": ("qwen", "Qwen/Qwen2.5-VL-72B-Instruct"),
}

MODEL_CHOICES = {**LLAVA_MODELS, **QWEN_MODELS}


def list_models() -> list:
    """Return list of all available model names."""
    return list(MODEL_CHOICES.keys())


def get_backend(model_key: str) -> str:
    """Return 'llava' or 'qwen' for a model key."""
    return MODEL_CHOICES[model_key][0]


def get_checkpoint(model_key: str) -> str:
    """Return the HuggingFace checkpoint ID for a model key."""
    return MODEL_CHOICES[model_key][1]
