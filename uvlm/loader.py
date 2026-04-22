import time

from .registry import MODEL_CHOICES
from .utils import is_colab, get_hf_token


def load_model(
    model_name: str,
    precision: str = "4bit",
    device_map: str = "auto",
    low_cpu_mem_usage: bool = True,
    hf_token=None,
    offload_folder=None,
    qwen_min_pixels: int = 256 * 28 * 28,
    qwen_max_pixels: int = 640 * 28 * 28,
) -> dict:
    """
    Load a VLM model and processor.

    Returns dict with keys:
        model, processor, model_id, backend, device_map_mode, main_device,
        gpu_name, load_time_minutes, qwen_min_pixels, qwen_max_pixels, hf_token
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
                torch_dtype=torch.float16,
                low_cpu_mem_usage=low_cpu_mem_usage,
                quantization_config=quantization_config,
                **load_kwargs_common,
                **auth_kwargs,
            )
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **load_kwargs_common,
                **auth_kwargs,
            )
            if device_map == "cuda0" and torch.cuda.is_available():
                model = model.to("cuda")

    elif backend == "qwen":
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=qwen_min_pixels,
            max_pixels=qwen_max_pixels,
            **auth_kwargs,
        )

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if quantization_config:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **load_kwargs_common,
                **auth_kwargs,
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto" if precision == "fp16" else torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **load_kwargs_common,
                **auth_kwargs,
            )
    else:
        raise ValueError(f"Unknown backend: {backend}")

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
        "qwen_min_pixels": qwen_min_pixels,
        "qwen_max_pixels": qwen_max_pixels,
        "hf_token": token,
    }
