def _move_inputs_to_model_if_needed(inputs: dict, model_ctx: dict) -> dict:
    import torch
    main_device = model_ctx.get("main_device")
    if main_device is None:
        return inputs
    return {k: (v.to(main_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def run_inference(
    image_path: str,
    prompt: str,
    model_ctx: dict,
    max_new_tokens: int = 50,
    do_sample: bool = True,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> tuple:
    """
    Run inference on a single image.

    Returns:
        (raw_response: str, generated_token_count: int)
    """
    import requests
    from io import BytesIO
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import GenerationConfig

    model = model_ctx["model"]
    processor = model_ctx["processor"]
    backend = model_ctx["backend"]

    if isinstance(image_path, str) and image_path.startswith("http"):
        image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    if backend == "llava":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        prompt_string = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=image,
            text=prompt_string,
            return_tensors="pt",
        )

        inputs = _move_inputs_to_model_if_needed(inputs, model_ctx)

        output = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        generated_token_count = len(output[0]) - len(inputs["input_ids"][0])

        raw = processor.decode(output[0], skip_special_tokens=True).strip()

        if "[/INST]" in raw:
            raw = raw.split("[/INST]")[-1].strip()

        if "ASSISTANT:" in raw:
            raw = raw.split("ASSISTANT:")[-1].strip()
        elif "Assistant:" in raw:
            raw = raw.split("Assistant:")[-1].strip()

        if "assistant" in raw.lower() and "user" in raw.lower():
            raw_lower = raw.lower()
            idx = raw_lower.rfind("assistant")
            if idx != -1:
                after_assistant = raw[idx + len("assistant"):].lstrip()
                if after_assistant:
                    raw = after_assistant

        return raw, generated_token_count

    if backend == "qwen":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = _move_inputs_to_model_if_needed(inputs, model_ctx)

        gen_cfg = GenerationConfig.from_model_config(model.config)
        gen_cfg.do_sample = bool(do_sample)
        gen_cfg.max_new_tokens = int(max_new_tokens)

        if gen_cfg.do_sample:
            gen_cfg.temperature = float(temperature)
            gen_cfg.top_p = float(top_p)

        generated_ids = model.generate(
            **inputs,
            generation_config=gen_cfg,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        generated_token_count = len(generated_ids_trimmed[0])

        raw = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return raw, generated_token_count

    raise ValueError(f"Unknown backend='{backend}'. Expected 'llava' or 'qwen'.")
