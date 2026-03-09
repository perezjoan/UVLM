# UVLM: Complete Project Documentation

## Executive Summary

**UVLM (Universal Vision-Language Model Loader)** is a Google Colab-based benchmarking framework that provides a unified interface for loading, configuring, and evaluating multiple Vision-Language Models (VLMs) on custom image analysis tasks. The tool abstracts the substantial architectural differences between VLM families — currently LLaVA-NeXT and Qwen2.5-VL — behind a single inference function, enabling researchers to compare models using identical prompts and evaluation protocols without writing model-specific code.

UVLM is distributed as a main Colab notebook (`UVLM.ipynb`) plus two pre-configured benchmark notebooks (`bench_1_no_reasoning.ipynb`, `bench_2_with_reasoning.ipynb`) and requires only a Google account with access to a GPU runtime.

**Current version**: v2.2.2  
**License**: Apache License 2.0  
**Repository**: https://github.com/perezjoan/UVLM  
**Paper**: Perez, J., Fusco, G. (2026). UVLM: A Universal Vision-Language Model Loader for Reproducible Multimodal Benchmarking. *arXiv preprint*.

---

## 1. Project Context and Motivation

### 1.1 Research Background

UVLM was developed in the context of two research projects in urban morphology:

- **SAGAI** (Streetscape Analysis with Generative Artificial Intelligence): A workflow for scoring street-level urban scenes using VLMs and open-access geospatial data (Perez & Fusco, 2025, *Geomatica*).
- **emc2**: A European initiative studying the 15-minute city model in peripheral urban areas (Fusco et al., 2024, AESOP), which has also produced open-source geospatial tools such as PPCA (Perez & Fusco, 2025, *SoftwareX*).

The primary use case involves analyzing French street-level photographs to extract:

1. **Street frontage length** (metric estimation in meters)
2. **Pedestrian entrance counts** (numeric counting)
3. **Vegetation type classification** (multi-category system)

UVLM will be integrated into the SAGAI workflow as its vision-language inference engine, replacing the current single-model implementation.

### 1.2 The Problem

VLM families differ fundamentally in:

- **Vision encoding**: CLIP-based (LLaVA) vs redesigned ViT with RMSNorm/SwiGLU (Qwen)
- **Processor classes**: Dedicated `LlavaNextProcessor` vs generic `AutoProcessor` with separate `process_vision_info()`
- **Tokenization**: Joint image-text encoding (LLaVA) vs separate vision preprocessing (Qwen)
- **Decoding**: Full-sequence decode + string cleaning (LLaVA) vs token-level trimming before decode (Qwen)
- **Generation configuration**: Direct keyword arguments (LLaVA) vs `GenerationConfig` object (Qwen)
- **Memory management**: Standard pixel input (LLaVA) vs configurable visual token budget with min/max pixel constraints (Qwen)

These are not superficial API variations — they reflect different transformer logic in how visual information is encoded, merged with text tokens, and decoded into language. Researchers who wish to compare models must write and maintain separate inference pipelines for each family, even when the evaluation task is identical.

---

## 2. Architecture Overview

### 2.1 Three-Block Design

The notebook is organized into four cells: one Markdown header (version, license, usage guide, feature summary) followed by three code cells, each handling a distinct stage of the benchmarking workflow:

#### Block 1: Model Loading & Hardware Configuration

**Executed once per session.**

Responsibilities:

- Install dependencies: `transformers` (from git HEAD), `accelerate`, `bitsandbytes`, `qwen-vl-utils`
- Present dropdown UI for model selection (11 checkpoints)
- Auto-detect backend (LLaVA vs Qwen) from model ID
- Load the correct processor and model classes
- Support precision modes: FP16, 8-bit quantization, 4-bit quantization (via BitsAndBytes)
- Configure device placement: GPU-only (`cuda:0`), auto (`accelerate` decides), or GPU + CPU offload
- Optional Hugging Face token authentication for gated models
- For Qwen models: configure visual token budget parameters (min/max pixel constraints)
- Display persistent status widget with load time and device allocation

**Supported models:**

| Family | Model | Parameters | Checkpoint ID |
|--------|-------|------------|---------------|
| LLaVA-NeXT | Mistral 7B | 7B | `llava-hf/llava-v1.6-mistral-7b-hf` |
| | Vicuna 7B | 7B | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| | Vicuna 13B | 13B | `llava-hf/llava-v1.6-vicuna-13b-hf` |
| | 34B | 34B | `llava-hf/llava-v1.6-34b-hf` |
| | LLaMA3 8B | 8B | `llava-hf/llama3-llava-next-8b-hf` |
| | 72B | 72B | `llava-hf/llava-next-72b-hf` |
| | 110B | 110B | `llava-hf/llava-next-110b-hf` |
| Qwen2.5-VL | 3B Instruct | 3B | `Qwen/Qwen2.5-VL-3B-Instruct` |
| | 7B Instruct | 7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| | 32B Instruct | 32B | `Qwen/Qwen2.5-VL-32B-Instruct` |
| | 72B Instruct | 72B | `Qwen/Qwen2.5-VL-72B-Instruct` |

**Note on large models**: The 72B and 110B checkpoints exceed single-GPU memory even with 4-bit quantization. While UVLM includes them in the registry and supports `device_map="auto"`, their effective use requires multi-GPU environments beyond free-tier Colab. Multi-GPU parallelism with batched inference is not yet implemented. In practice, models up to 34B parameters can be loaded on a single Colab GPU (T4 or A100) using 4-bit quantization.

#### Block 2: Inference Configuration & Prompt Builder

**Re-executable to modify tasks and prompts.**

Responsibilities:

- Mount Google Drive for image and output access
- Provide widget-based multi-task prompt form (up to 10 tasks via `IntSlider`)
- For each task: column name, task prompt, theory section, format specification, task type
- Configure global generation parameters: temperature, top-p, max tokens (default: 50, range: 1–1500), optional fixed random seed
- Per-task toggles for consensus validation and advanced reasoning
- Define the core `run_inference()` function (backend-agnostic forward pass)
- Define the `parse_response()` function (type-specific output parsing)
- Define the `compute_consensus()` function (multi-run agreement)

**Task types:**

| Type | Description | Parser behavior |
|------|-------------|-----------------|
| `numeric` | Integer/float extraction | Extracts the **last** number via regex `r"-?\d+(?:\.\d+)?"` |
| `category` | Classification labels | Strips common prefixes ("The answer is:", "Based on the image," etc.), returns cleaned text |
| `boolean` | Yes/no answers | Normalizes variations (yes/true/present → `1`, no/false/absent → `0`) |
| `text` | Free-form responses | Returns cleaned text as-is |

**Important**: `parse_numeric` extracts the *last* number found in the response (not the first). This design ensures that when advanced reasoning produces intermediate calculations before the final answer, the correct value is captured.

#### Block 3: Batch Execution Engine

**Re-executable for different image sets.**

Responsibilities:

- Iterate over all images in a user-specified Google Drive folder
- Execute all configured tasks sequentially for each image
- Write results to CSV: one row per image, one column per task, plus `{col}_raw` for raw responses and `{col}_truncated` for truncation flags
- Resume mode: detect already-processed images and skip them
- Schema upgrading: append missing columns when new tasks are added between runs
- Per-task error handling: if one task fails, remaining tasks still execute; errors logged as "NA"
- Periodic checkpoint saves (every 3 images)
- Truncation detection on every task using exact tokenizer token count, with console alarm and CSV flag

---

## 3. Key Features

### 3.1 Dual-Backend Inference

The `run_inference()` function routes each call to the appropriate backend based on the loaded model family. The two pipelines differ substantially:

**LLaVA-NeXT pipeline:**

```python
# 1. Build conversation with chat template
conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
prompt_string = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

# 2. Tokenize image + text jointly
inputs = processor(images=image, text=prompt_string, return_tensors="pt")

# 3. Generate with direct keyword arguments
output = model.generate(**inputs, do_sample=do_sample, temperature=temperature,
                         top_p=top_p, max_new_tokens=tokens_to_use)

# 4. Decode FULL output (includes echoed prompt)
raw = processor.decode(output[0], skip_special_tokens=True).strip()

# 5. String-based response extraction (patterns vary by base LLM)
if "[/INST]" in raw:       # Mistral
    raw = raw.split("[/INST]")[-1].strip()
if "ASSISTANT:" in raw:    # Vicuna
    raw = raw.split("ASSISTANT:")[-1].strip()
# Additional handling for LLaMA3 "assistant"/"user" patterns
```

**Qwen2.5-VL pipeline:**

```python
# 1. Build message list with explicit content types
messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 2. Separate vision preprocessing
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt")

# 3. Generate with GenerationConfig object
gen_cfg = GenerationConfig.from_model_config(model.config)
gen_cfg.do_sample = bool(do_sample)
gen_cfg.max_new_tokens = int(tokens_to_use)
if gen_cfg.do_sample:
    gen_cfg.temperature = float(temperature)
    gen_cfg.top_p = float(top_p)
generated_ids = model.generate(**inputs, generation_config=gen_cfg)

# 4. Trim prompt tokens BEFORE decoding
generated_ids_trimmed = [out_ids[len(in_ids):]
                         for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]

# 5. Decode only the generated tokens (clean output, no post-processing)
raw = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)[0].strip()
```

**Key architectural differences:**

| Aspect | LLaVA-NeXT | Qwen2.5-VL |
|--------|-----------|-------------|
| Processor | Dedicated `LlavaNextProcessor` | Generic `AutoProcessor` + `process_vision_info()` |
| Model class | `LlavaNextForConditionalGeneration` | `Qwen2_5_VLForConditionalGeneration` |
| Generation config | Direct keyword arguments | `GenerationConfig` object from model config |
| Prompt removal | String matching on decoded text | Token ID trimming before decode |
| Decode method | `processor.decode(output[0])` | `processor.batch_decode(trimmed_ids)` |

### 3.2 Prompt Engineering

Each task prompt is assembled by concatenating four user-defined fields:

1. **Role** (global, shared across all tasks): Defines the model's persona
2. **Task** (per-task): The specific question or instruction
3. **Theory** (per-task): Definitions, scoring rules, edge cases
4. **Format** (per-task): Expected output structure

```python
full_prompt = f"""
{role_txt}

{task_txt}

{theory_txt}

{format_txt}
""".strip()
```

When advanced reasoning is enabled for a task, the format field is automatically overridden with a structured chain-of-thought directive (see Section 3.4).

### 3.3 Consensus Validation

**Purpose**: Improve reliability of VLM outputs by running each task multiple times and determining the answer by majority vote.

**Configuration**: Per-task checkbox + dropdown with options from 2 to 5 runs (default: 2). Consensus is not available for `text` task type.

**Implementation** (`compute_consensus()`):

1. Collect all parsed values from all runs (all runs are equal peers — no run is treated as primary)
2. Filter out NA values (runs where the parser failed to extract a valid answer)
3. Apply majority voting over remaining valid values:
   - For `numeric` with tolerance > 0: group values within the specified percentage of each other; tolerance is percentage-based (e.g., 10% means values within 10% of each other are equivalent)
   - For all other cases: exact match via `Counter.most_common(1)`
4. Compute agreement ratio = count(most_common) / total_runs (including failed runs)
5. Consensus reached if agreement_ratio > 0.5

**Output columns**: `{col}_consensus` (YES/NO), `{col}_agreement` (ratio), `{col}_runs` (JSON list of all values)

```python
def compute_consensus(parsed_values, task_type, numeric_tolerance=0.0):
    valid_values = [v for v in parsed_values if not is_na_value(v)]

    if not valid_values:
        return {"final_value": "NA", "consensus_reached": False,
                "agreement_ratio": 0.0, "all_values": parsed_values}

    if task_type == "numeric" and numeric_tolerance > 0:
        # Group values within tolerance percentage, take largest group average
        ...
    else:
        counter = Counter(valid_values)
        final_value, agreement_count = counter.most_common(1)[0]

    agreement_ratio = agreement_count / len(parsed_values)  # ALL runs, including NA
    consensus_reached = agreement_ratio > 0.5

    return {"final_value": final_value, "consensus_reached": consensus_reached,
            "agreement_ratio": round(agreement_ratio, 2), "all_values": parsed_values}
```

**NA filtering**: The `is_na_value()` helper recognizes "NA", "N/A", "NAN", "NONE", "NULL", empty strings, and `None`. This ensures that parsing failures do not interfere with the voting process, while the agreement ratio is still computed over all runs to preserve the reliability metric.

### 3.4 Reasoning Support

UVLM supports two approaches to multi-step visual reasoning:

**User-defined reasoning**: Users can implement custom chain-of-thought strategies by writing task prompts that request step-by-step explanations, and increasing the max-token slider (up to 1500) to accommodate longer outputs. This gives full control over reasoning structure and token allocation.

**Built-in advanced reasoning mode** (reference implementation): A per-task checkbox enables a standardized CoT template, primarily intended for benchmarking. When enabled:

- The format field is automatically replaced with a structured CoT directive:

```python
ADVANCED_REASONING_FORMATS = {
    "numeric":  "First, describe what you observe and explain your reasoning step by step.\n"
                "Then, on the last line, write only: ANSWER: <integer>",
    "category": "First, describe what you observe and explain your reasoning step by step.\n"
                "Then, on the last line, write only: ANSWER: <category number>",
    "boolean":  "First, describe what you observe and explain your reasoning step by step.\n"
                "Then, on the last line, write only: ANSWER: <yes or no>",
}
```

- Max tokens is automatically overridden to `ADVANCED_REASONING_MAX_TOKENS = 1024` to accommodate the reasoning trace
- The response parser (`parse_advanced_reasoning_response()`) scans the **last 5 lines** of output for `ANSWER:` pattern (case-insensitive regex)
- If found: extracts the value and applies the standard type-specific parser
- If not found: **graceful fallback** to standard parsing on the full response
- Reasoning trace stored in `{col}_reasoning` CSV column (full text, not truncated)

In practice, users are encouraged to design their own reasoning prompts tailored to their specific tasks rather than relying on the built-in mode, which applies a generic template across all task types.

### 3.5 Truncation Detection

**Purpose**: Alert the user when a model response was cut off by the token limit, which typically produces incomplete reasoning and unreliable parsed answers.

**Mechanism**: The `run_inference()` function stores the exact number of generated tokens in a global variable `_last_generated_tokens`, computed directly from the model output tensor before any text decoding or cleaning. For LLaVA, this is `len(output[0]) - len(inputs["input_ids"][0])`; for Qwen, `len(generated_ids_trimmed[0])`. After each call, the truncation detector compares this count against the effective token limit:

```python
def check_truncation(max_tokens: int) -> tuple:
    return _last_generated_tokens >= max_tokens, _last_generated_tokens
```

This approach avoids re-tokenizing the cleaned response text, which would produce inaccurate counts for LLaVA models where the raw response may still contain prompt fragments after string-based cleaning.

**Applies to all modes**: standard, consensus, and advanced reasoning — not limited to chain-of-thought tasks.

**Output**:

- CSV column `{col}_truncated` for every task (YES/NO)
- Console alarm: `🚨 {col}: TRUNCATION DETECTED — response used {token_count}/{max_tokens} tokens. Increase max_tokens!`

This allows users to identify token budget issues across their specific prompt, task, and model combination.

### 3.6 Response Parsing

All parsing functions return `"NA"` on failure:

```python
def parse_numeric(raw):
    """Extract the LAST number from the response."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", raw)
    return numbers[-1] if numbers else "NA"

def parse_category(raw):
    """Strip common prefixes, return cleaned text."""
    # Removes: "The answer is:", "Answer:", "Category:", "Based on the image,",
    #          "This is a", "I would classify this as", etc.
    # Takes first line if multi-line (under 100 chars)
    # Strips trailing periods

def parse_boolean(raw):
    """Normalize to 1/0."""
    # Positive: yes, true, y, 1, correct, present, visible, exists, found, detected
    # Negative: no, false, n, 0, incorrect, absent, not visible, none, cannot, don't
    # Checks negatives FIRST (handles "not present" before matching "present")

def parse_text(raw):
    """Return cleaned text with whitespace normalization."""
```

### 3.7 CSV Schema and Resume Mode

**Header structure:**

```python
header = (["image_name"]
          + [spec["column"] for spec in TASK_SPECS]
          + reasoning_columns   # {col}_reasoning (advanced reasoning tasks only)
          + truncated_columns   # {col}_truncated (ALL tasks)
          + consensus_columns   # {col}_consensus, {col}_agreement, {col}_runs
          + [f"{spec['column']}_raw" for spec in TASK_SPECS])
```

**Schema upgrading**: When new tasks are added between runs:

```python
missing_cols = [c for c in header if c not in df.columns]
for c in missing_cols:
    df[c] = "NA"
```

**Resume logic**: For each image, checks if a task column already has a non-empty, non-NA, non-ERROR value. If so, skips that task for that image.

### 3.8 Reproducibility

When the "Fixed seed" checkbox is enabled:

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 4. Version History

| Version | Key Changes |
|---------|-------------|
| v1.0 | Basic dual-backend loader, single-task inference |
| v2.0 | Multi-task prompt builder, batch execution engine |
| v2.1 | Consensus validation feature |
| v2.2 | Advanced reasoning (chain-of-thought) support |
| v2.2.1 | NA value filtering fix in consensus voting; `is_na_value()` helper ensures parsing failures do not influence majority vote |
| v2.2.2 | Truncation detection on all tasks using exact generated token count from model output (`{col}_truncated` column + console alarm); advanced reasoning auto-overrides to `ADVANCED_REASONING_MAX_TOKENS = 1024`; max tokens slider range extended to 1500 for user-defined reasoning; consensus runs extended to 2–5; reasoning column no longer truncated; UTF-8 encoding fix; pre-configured benchmark notebooks with dynamic output filenames |

---

## 5. Benchmark Dataset

- **120 images** of French street frontages
- **Zenodo archive**: [link to be added upon publication]
- **CSV output** for downstream statistical analysis
- Five analysis tasks: sidewalk detection, motor vehicle counting, pedestrian entrance counting, street frontage length estimation, and vegetation type classification
- Benchmark prompts provided as supplementary material

---

## 6. Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` (git HEAD) | Model loading, processors, generation |
| `accelerate` | Device placement, memory management |
| `bitsandbytes` | 4-bit and 8-bit quantization |
| `qwen-vl-utils` | Qwen vision preprocessing (`process_vision_info`) |
| `torch` | PyTorch backend |
| `Pillow` | Image loading and conversion |
| `ipywidgets` | Interactive Colab UI |
| `pandas` | CSV management |

**Runtime**: Google Colab with GPU (T4 free-tier or A100 Pro).

---

## 7. Repository Structure

| File | Description |
|------|-------------|
| `UVLM.ipynb` | Complete notebook (1 markdown cell + 3 code blocks) |
| `README.md` | Repository landing page with quick start guide |
| `UVLM_Project_Complete_Documentation.md` | This documentation |
| `figure1_architecture.svg` | Architecture diagram (Figure 1 from the paper) |
| `figure2_prompt_form.svg` | Prompt builder example (Figure 2) |
| `VERSIONS.txt` | Version history and changelog |
| `LICENSE` | Apache License 2.0 |

The benchmark dataset (120 images), prompts, full results, and pre-configured benchmark notebooks are provided as supplementary materials alongside the paper (link to be added upon publication).

---

## 8. Limitations and Future Work

### Current Limitations

- Only supports LLaVA-NeXT and Qwen2.5-VL families
- Google Colab dependency (single-GPU VRAM limits for free-tier)
- Sequential image processing (no batching across images)
- Single-image inference only (no video frame analysis)
- Largest models (72B+) require multi-GPU setups not available on free-tier Colab

### Planned Future Work

- **Additional VLM families**: InternVL, BLIP-2, CogVLM
- **Multi-GPU batching**: Parallel inference across images on multi-device setups
- **Video frame analysis**: Temporal visual tasks
- **API mode**: Cloud deployment for integration with automated pipelines
- **Automatic prompt optimization**: Reduce prompt engineering burden
- **SAGAI integration**: UVLM as the VLM inference engine within the SAGAI workflow

---

*Document version: v2.2.2 — March 2026*
*Corresponding author: Joan Perez (Urban Geo Analytics)*
