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

# Qwen3-VL (released Sept 2025+). Same qwen_vl_utils pipeline as Qwen2.5-VL,
# loaded via AutoModelForImageTextToText. Requires transformers >= 4.57 and
# qwen-vl-utils >= 0.0.14 (images resize to multiples of 32 px, not 28 px).
QWEN3_MODELS = {
    "[Qwen3] Qwen3-VL 2B Instruct": ("qwen3", "Qwen/Qwen3-VL-2B-Instruct"),
    "[Qwen3] Qwen3-VL 4B Instruct": ("qwen3", "Qwen/Qwen3-VL-4B-Instruct"),
    "[Qwen3] Qwen3-VL 8B Instruct": ("qwen3", "Qwen/Qwen3-VL-8B-Instruct"),
    "[Qwen3] Qwen3-VL 32B Instruct": ("qwen3", "Qwen/Qwen3-VL-32B-Instruct"),
}

# InternVL3.5 (OpenGVLab, released Aug 2025), Transformers-native "-HF"
# checkpoints. Standard pipeline: tokenizing chat template -> generate ->
# token slicing. Requires transformers >= 4.52.1 (covered by the >= 4.57 pin).
# Not gated: no HF token required.
INTERNVL_MODELS = {
    "[InternVL] InternVL3.5 1B": ("internvl", "OpenGVLab/InternVL3_5-1B-HF"),
    "[InternVL] InternVL3.5 2B": ("internvl", "OpenGVLab/InternVL3_5-2B-HF"),
    "[InternVL] InternVL3.5 4B": ("internvl", "OpenGVLab/InternVL3_5-4B-HF"),
    "[InternVL] InternVL3.5 8B": ("internvl", "OpenGVLab/InternVL3_5-8B-HF"),
    "[InternVL] InternVL3.5 14B": ("internvl", "OpenGVLab/InternVL3_5-14B-HF"),
    "[InternVL] InternVL3.5 38B": ("internvl", "OpenGVLab/InternVL3_5-38B-HF"),
}

MODEL_CHOICES = {**LLAVA_MODELS, **QWEN_MODELS, **QWEN3_MODELS, **INTERNVL_MODELS}

# Family -> model-dict mapping, used by the notebooks to build a two-level
# (family, then model) selection widget. Add new families here.
FAMILY_GROUPS = {
    "LLaVA-NeXT": LLAVA_MODELS,
    "Qwen2.5-VL": QWEN_MODELS,
    "Qwen3-VL": QWEN3_MODELS,
    "InternVL3.5": INTERNVL_MODELS,
}


def list_models() -> list:
    """Return list of all available model names."""
    return list(MODEL_CHOICES.keys())


def get_backend(model_key: str) -> str:
    """Return 'llava', 'qwen', 'qwen3', or 'internvl' for a model key."""
    return MODEL_CHOICES[model_key][0]


def get_checkpoint(model_key: str) -> str:
    """Return the HuggingFace checkpoint ID for a model key."""
    return MODEL_CHOICES[model_key][1]
