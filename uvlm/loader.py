import time

from .registry import MODEL_CHOICES
from .utils import is_colab, get_hf_token


# ---------------------------------------------------------------------------
# Vision budgets (v4.1.0)
#
# Every backend runs its NATIVE preprocessing by default. Budgets are
# explicit, per-backend, expressed through each family's official
# preprocessing parameters, and returned in the model context so run
# manifests can record exactly what was used.
#
# Official knobs (transformers source, checked 2026-09):
#   qwen/qwen3 : min_pixels / max_pixels
#   internvl   : crop_to_patches, min_patches, max_patches
#                (the processor class defaults force crop_to_patches=True at
#                 call time even though the -HF checkpoint configs say false)
#   gemma4     : max_soft_tokens in {70, 140, 280, 560, 1120} (default 280)
#   llava      : image_grid_pinpoints (LLaVA-Next any-res grids)
# ---------------------------------------------------------------------------

_GEMMA_SOFT_TOKENS = {70, 140, 280, 560, 1120}

# Cross-family parity presets. Anchor: one 448x448 InternVL tile ~0.20 MP;
# "medium" reproduces UVLM <= 4.0.x's implicit Qwen budget (~0.5 MP) and maps
# it to the nearest per-family equivalents.
VISION_BUDGET_PRESETS = {
    "low": {
        "qwen":     {"min_pixels": 64 * 28 * 28,  "max_pixels": 256 * 28 * 28},
        "internvl": {"crop_to_patches": True, "min_patches": 1, "max_patches": 1},
        "gemma4":   {"max_soft_tokens": 140},
        "llava":    {"image_grid_pinpoints": [[336, 336]]},
    },
    "medium": {
        "qwen":     {"min_pixels": 256 * 28 * 28, "max_pixels": 640 * 28 * 28},
        "internvl": {"crop_to_patches": True, "min_patches": 1, "max_patches": 3},
        "gemma4":   {"max_soft_tokens": 280},
        "llava":    {"image_grid_pinpoints": [[336, 672], [672, 336], [672, 672]]},
    },
    "high": {
        "qwen":     {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28},
        "internvl": {"crop_to_patches": True, "min_patches": 1, "max_patches": 6},
        "gemma4":   {"max_soft_tokens": 560},
        "llava":    {"image_grid_pinpoints": [[336, 672], [672, 336], [672, 672],
                                              [1008, 336], [336, 1008]]},
    },
}


def _resolve_vision_budget(vision_budget, backend):
    """Return (call_time_images_kwargs, resolved_record) for this backend.

    vision_budget:
      None / "native" -> ({}, None): no overrides anywhere (the default).
      "low"/"medium"/"high" -> the preset row for this backend.
      dict -> raw knob overrides for THIS backend, validated and passed
              through verbatim.

    Where budgets act: qwen/qwen3 at processor init; llava at load time on
    both the model config and the processor (the two must agree); internvl
    and gemma4 at call time through apply_chat_template's images kwargs.
    """
    if vision_budget in (None, "native"):
        return {}, None
    if isinstance(vision_budget, str):
        if vision_budget not in VISION_BUDGET_PRESETS:
            raise ValueError(
                "vision_budget must be None, 'native', one of "
                f"{sorted(VISION_BUDGET_PRESETS)}, or a dict; got {vision_budget!r}")
        key = "qwen" if backend in ("qwen", "qwen3") else backend
        resolved = dict(VISION_BUDGET_PRESETS[vision_budget].get(key, {}))
    elif isinstance(vision_budget, dict):
        resolved = dict(vision_budget)
    else:
        raise TypeError(f"vision_budget: unsupported type {type(vision_budget)}")

    allowed = {
        "qwen": {"min_pixels", "max_pixels"},
        "qwen3": {"min_pixels", "max_pixels"},
        "internvl": {"crop_to_patches", "min_patches", "max_patches"},
        "gemma4": {"max_soft_tokens"},
        "llava": {"image_grid_pinpoints"},
    }[backend]
    bad = set(resolved) - allowed
    if bad:
        raise ValueError(f"vision_budget keys {sorted(bad)} are not valid for "
                         f"backend '{backend}' (valid: {sorted(allowed)})")
    if backend == "gemma4" and "max_soft_tokens" in resolved:
        if resolved["max_soft_tokens"] not in _GEMMA_SOFT_TOKENS:
            raise ValueError("gemma4 max_soft_tokens must be one of "
                             f"{sorted(_GEMMA_SOFT_TOKENS)}")
    return resolved, resolved



def load_model(
    model_name: str,
    precision: str = "4bit",
    device_map: str = "auto",
    low_cpu_mem_usage: bool = True,
    hf_token=None,
    offload_folder=None,
    vision_budget=None,
    qwen_min_pixels: int | None = None,
    qwen_max_pixels: int | None = None,
) -> dict:
    """
    Load a VLM model and processor.

    Returns dict with keys:
        model, processor, model_id, backend, device_map_mode, main_device,
        gpu_name, load_time_minutes, vision_budget, images_kwargs,
        qwen_min_pixels, qwen_max_pixels, hf_token
    """
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        LlavaNextProcessor,
        LlavaNextForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
    )

    start = time.time()

    token = get_hf_token(hf_token)
    auth_kwargs = {}
    if token:
        from huggingface_hub import login
        try:
            login(token=token)
        except Exception:
            pass
        auth_kwargs["token"] = token

    backend, model_id = MODEL_CHOICES[model_name]

    images_kwargs, budget_record = _resolve_vision_budget(vision_budget, backend)
    if backend in ("qwen", "qwen3") and (qwen_min_pixels or qwen_max_pixels):
        # Legacy qwen kwargs win over vision_budget and are recorded the same way.
        images_kwargs = {}
        budget_record = {}
        if qwen_min_pixels:
            budget_record["min_pixels"] = qwen_min_pixels
        if qwen_max_pixels:
            budget_record["max_pixels"] = qwen_max_pixels
    if budget_record:
        print(f"Vision budget ({backend}): {budget_record}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    quantization_config = None
    if precision == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif precision == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    load_kwargs_common = {}
    if device_map == "auto":
        load_kwargs_common["device_map"] = "auto"
    elif device_map == "offload":
        load_kwargs_common["device_map"] = "auto"
        if offload_folder is None:
            offload_folder = "/content/offload" if is_colab() else "./offload"
        load_kwargs_common["offload_folder"] = offload_folder
    # cuda0: no device_map kwarg — model loaded to CPU then moved to cuda

    print(f"Loading {model_name} ({precision}) ...")

    if backend == "llava":
        processor = LlavaNextProcessor.from_pretrained(model_id, **auth_kwargs)

        if quantization_config:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.float16,
                low_cpu_mem_usage=low_cpu_mem_usage,
                quantization_config=quantization_config,
                **load_kwargs_common,
                **auth_kwargs,
            )
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.float16,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **load_kwargs_common,
                **auth_kwargs,
            )
            if device_map == "cuda0" and torch.cuda.is_available():
                model = model.to("cuda")

    elif backend == "qwen":
        _qwen_proc_kwargs = dict(auth_kwargs)
        if budget_record:
            _qwen_proc_kwargs.update(budget_record)
        processor = AutoProcessor.from_pretrained(model_id, **_qwen_proc_kwargs)

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if quantization_config:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch_dtype,
                quantization_config=quantization_config,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **load_kwargs_common,
                **auth_kwargs,
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype="auto" if precision == "fp16" else torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **load_kwargs_common,
                **auth_kwargs,
            )
    elif backend in ("qwen3", "internvl", "gemma4"):
        # Both families load via the generic AutoModelForImageTextToText class.
        #   qwen3    : Qwen3-VL — keeps the Qwen min/max pixel budget;
        #              requires transformers >= 4.57 and qwen-vl-utils >= 0.0.14.
        #   internvl : InternVL3.5 "-HF" checkpoints — plain AutoProcessor.
        #              Not gated.
        #   gemma4   : Gemma 4 E2B/E4B/12B Instruct — plain AutoProcessor;
        #              requires transformers >= 5.
        from transformers import AutoModelForImageTextToText

        processor_kwargs = dict(auth_kwargs)
        if backend == "qwen3" and budget_record:
            processor_kwargs.update(budget_record)

        extra_model_kwargs = {}
        if backend == "internvl":
            # The -HF checkpoints ship embed_tokens and lm_head untied with
            # different values while the config requests tying; transformers
            # keeps them untied (correct) but warns at every load. Aligning
            # the config removes the warning without changing behavior.
            extra_model_kwargs["tie_word_embeddings"] = False

        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

        # These families are trained in BF16. Prefer BF16 when the GPU supports it
        # natively (RTX 30xx+, A100, L4...) to avoid FP16 overflow issues;
        # fall back to FP16 on older GPUs (e.g. T4), FP32 on CPU.
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        if quantization_config:
            if torch_dtype == torch.bfloat16 and precision == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            if backend == "gemma4":
                # Gemma 4's Per-Layer Embedding tables are lookup-only and
                # designed to live off-GPU: permit non-quantized modules to
                # spill to CPU RAM. Scoped to Gemma so an overflowing
                # Qwen/InternVL load fails with transformers' explicit memory
                # error instead of dispatching quantized layers to CPU, which
                # bitsandbytes cannot execute (meta-tensor crash).
                quantization_config.llm_int8_enable_fp32_cpu_offload = True
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch_dtype,
                quantization_config=quantization_config,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **load_kwargs_common,
                **extra_model_kwargs,
                **auth_kwargs,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype="auto" if precision == "fp16" else torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **load_kwargs_common,
                **extra_model_kwargs,
                **auth_kwargs,
            )
            if device_map == "cuda0" and torch.cuda.is_available():
                model = model.to("cuda")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if backend == "llava" and budget_record:
        # LLaVA-Next keeps two copies of image_grid_pinpoints: the processor's
        # (how the image is cropped) and the model config's (how many patches
        # the model expects when splitting image features). They must agree,
        # and at load time; overriding only the processor at call time makes
        # generation fail with a split_with_sizes mismatch.
        _pins = budget_record["image_grid_pinpoints"]
        model.config.image_grid_pinpoints = _pins
        processor.image_processor.image_grid_pinpoints = _pins

    print("Device map:", getattr(model, "hf_device_map", "single-device"))

    main_device = None if hasattr(model, "hf_device_map") else model.device

    load_time_minutes = (time.time() - start) / 60
    print(f"Model loaded in {load_time_minutes:.2f} min on {gpu_name} ({precision})")

    return {
        "model": model,
        "processor": processor,
        "model_id": model_id,
        "backend": backend,
        "device_map_mode": device_map,
        "main_device": main_device,
        "gpu_name": gpu_name,
        "load_time_minutes": load_time_minutes,
        "vision_budget": budget_record,
        "images_kwargs": images_kwargs if backend in ("internvl", "gemma4") else {},
        "qwen_min_pixels": qwen_min_pixels,
        "qwen_max_pixels": qwen_max_pixels,
        "hf_token": token,
    }
