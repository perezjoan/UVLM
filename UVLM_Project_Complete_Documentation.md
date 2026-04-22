# UVLM: Complete Project Documentation

## Executive Summary

**UVLM (Universal Vision-Language Model Loader)** is a pip-installable Python package for reproducible benchmarking of Vision-Language Models (VLMs). It provides a unified interface for loading, configuring, and evaluating multiple VLM architectures on custom image analysis tasks. The tool abstracts the substantial architectural differences between VLM families — currently LLaVA-NeXT and Qwen2.5-VL — behind a single inference function, enabling researchers to compare models using identical prompts and evaluation protocols without writing model-specific code.

UVLM is distributed as a Python package (`uvlm/`) installable from GitHub, with two interactive notebook interfaces: a Google Colab notebook for zero-install cloud access and a local Jupyter notebook for researchers with their own GPU hardware.

**Current version**: v3.0.0  
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

### 2.1 Package Structure

Starting with v3.0.0, UVLM is organized as a modular Python package. The core logic is split into eight modules, with interactive notebook interfaces as thin wrappers:

```
UVLM/
├── pyproject.toml              # Package metadata and dependencies
├── uvlm/                       # Core Python package
│   ├── __init__.py             # Version, public API exports
│   ├── registry.py             # Model registries (11 checkpoints)
│   ├── loader.py               # load_model() — model + processor loading
│   ├── inference.py            # run_inference() — dual-backend forward pass
│   ├── parsers.py              # parse_response() — type-specific output parsing
│   ├── consensus.py            # compute_consensus() — multi-run agreement
│   ├── batch.py                # run_batch() — batch execution with CSV resume
│   ├── prompts.py              # Prompt templates and assembly
│   └── utils.py                # set_seed(), is_colab(), get_hf_token()
├── notebooks/
│   ├── UVLM_colab.ipynb        # Google Colab interface (installs from GitHub)
│   └── UVLM_local.ipynb        # Local Jupyter interface (local GPU)
```

**Design principle**: Widget UI lives in the notebooks, core logic in the package. The package provides a programmatic API (`load_model`, `run_inference`, `run_batch`). The notebooks provide the interactive widget forms. This means UVLM can also be used as a plain Python library in scripts without any notebook.

### 2.2 Deployment Modes

#### Google Colab (zero install)

The Colab notebook (`notebooks/UVLM_colab.ipynb`) installs the package automatically via `!pip install git+https://github.com/perezjoan/UVLM.git`, mounts Google Drive for image access, and retrieves the HF token from Colab secrets. Requires only a Google account with GPU runtime (T4 free-tier or A100 Pro).

#### Local Jupyter Notebook

The local notebook (`notebooks/UVLM_local.ipynb`) assumes the package is already installed via `pip install git+https://github.com/perezjoan/UVLM.git`. Images are read from local folders. The HF token is retrieved from the `HF_TOKEN` environment variable or `huggingface-cli login` cache. Requires a local NVIDIA GPU with CUDA.

**Note**: PyTorch with CUDA must be installed separately to match the local GPU driver, e.g.: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

#### Python Script (advanced)

The package can be used programmatically without any notebook:

```python
from uvlm import load_model, run_inference, parse_response

ctx = load_model("[Qwen]  Qwen2.5-VL 7B Instruct", precision="4bit")
raw, tokens = run_inference("photo.jpg", "Count the cars", ctx)
result = parse_response(raw, "numeric")
```

### 2.3 Three-Block Workflow

Both notebooks follow the same three-block workflow inherited from v2.x:

#### Block 1: Model Loading & Hardware Configuration

**Executed once per session.**

Calls `uvlm.load_model()` which:

- Auto-detects backend (LLaVA vs Qwen) from model ID via the registry
- Loads the correct processor and model classes
- Supports precision modes: FP16, 8-bit quantization, 4-bit quantization (via BitsAndBytes)
- Configures device placement: GPU-only (`cuda:0`), auto (`accelerate` decides), or GPU + CPU offload
- For Qwen models: configures visual token budget parameters (min/max pixel constraints)
- Returns a `model_ctx` dict containing model, processor, backend, device info, and load time

The `model_ctx` dict replaces all global variables from v2.x. It is passed to all subsequent functions.

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

**Note on large models**: The 72B and 110B checkpoints exceed single-GPU memory even with 4-bit quantization. While UVLM includes them in the registry and supports `device_map="auto"`, their effective use requires multi-GPU environments. Multi-GPU parallelism with batched inference is not yet implemented. In practice, models up to 34B parameters can be loaded on a single GPU (T4, L4, or A100) using 4-bit quantization.

#### Block 2: Inference Configuration & Prompt Builder

**Re-executable to modify tasks and prompts.**

The notebook provides a widget-based form. The apply callback builds a `task_specs` list and generation parameters, which are passed to `run_batch()` in Block 3.

- Multi-task prompt form (up to 10 tasks via `IntSlider`)
- For each task: column name, task prompt, theory section, format specification, task type
- Global generation parameters: temperature, top-p, max tokens (default: 50, range: 1–1500), optional fixed random seed
- Per-task toggles for consensus validation and advanced reasoning
- Qwen-specific pixel settings (shown conditionally when a Qwen model is loaded)

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

Calls `uvlm.run_batch()` which:

- Iterates over all images in a user-specified folder (Google Drive on Colab, local path on Jupyter)
- Executes all configured tasks sequentially for each image
- Writes results to CSV: one row per image, one column per task, plus `{col}_raw` for raw responses and `{col}_truncated` for truncation flags
- Resume mode: detects already-processed images and skips them
- Schema upgrading: appends missing columns when new tasks are added between runs
- Per-task error handling: if one task fails, remaining tasks still execute; errors logged as "NA"
- Periodic checkpoint saves (every 3 images)
- Truncation detection on every task using exact token count, with console alarm and CSV flag

---

## 3. Key Features

### 3.1 Dual-Backend Inference

The `run_inference()` function routes each call to the appropriate backend based on the `model_ctx["backend"]` value. The function accepts the model context dict and returns a `(raw_response, token_count)` tuple. The two pipelines differ substantially:

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

Each task prompt is assembled by the `build_prompt()` function from four user-defined fields:

1. **Role** (global, shared across all tasks): Defines the model's persona
2. **Task** (per-task): The specific question or instruction
3. **Theory** (per-task): Definitions, scoring rules, edge cases
4. **Format** (per-task): Expected output structure

```python
from uvlm.prompts import build_prompt

full_prompt = build_prompt(role_txt, task_txt, theory_txt, format_txt)
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
from uvlm.consensus import compute_consensus

result = compute_consensus(parsed_values, task_type, numeric_tolerance=0.0)
# Returns: {"final_value": ..., "consensus_reached": bool,
#           "agreement_ratio": float, "all_values": list}
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

**Mechanism**: The `run_inference()` function returns the exact number of generated tokens as the second element of its return tuple, computed directly from the model output tensor before any text decoding or cleaning. For LLaVA, this is `len(output[0]) - len(inputs["input_ids"][0])`; for Qwen, `len(generated_ids_trimmed[0])`. The `check_truncation()` utility compares this count against the effective token limit:

```python
from uvlm.utils import check_truncation

is_truncated, token_count = check_truncation(generated_tokens, max_tokens)
```

This approach avoids re-tokenizing the cleaned response text, which would produce inaccurate counts for LLaVA models where the raw response may still contain prompt fragments after string-based cleaning.

**Applies to all modes**: standard, consensus, and advanced reasoning — not limited to chain-of-thought tasks.

**Output**:

- CSV column `{col}_truncated` for every task (YES/NO)
- Console alarm: `{col}: TRUNCATION DETECTED — response used {token_count}/{max_tokens} tokens. Increase max_tokens!`

This allows users to identify token budget issues across their specific prompt, task, and model combination.

### 3.6 Response Parsing

All parsing functions in `uvlm/parsers.py` return `"NA"` on failure:

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
          + [spec["column"] for spec in task_specs]
          + reasoning_columns   # {col}_reasoning (advanced reasoning tasks only)
          + truncated_columns   # {col}_truncated (ALL tasks)
          + consensus_columns   # {col}_consensus, {col}_agreement, {col}_runs
          + [f"{spec['column']}_raw" for spec in task_specs])
```

**Schema upgrading**: When new tasks are added between runs:

```python
missing_cols = [c for c in header if c not in df.columns]
for c in missing_cols:
    df[c] = "NA"
```

**Resume logic**: For each image, checks if a task column already has a non-empty, non-NA, non-ERROR value. If so, skips that task for that image.

### 3.8 Reproducibility

When the "Fixed seed" checkbox is enabled, `set_seed()` from `uvlm/utils.py` is called:

```python
from uvlm.utils import set_seed

set_seed(42)  # Sets random, numpy, torch, cuda, cudnn deterministic
```

### 3.9 Environment Detection and Token Retrieval

The `uvlm/utils.py` module provides environment-aware utilities:

- `is_colab()`: Detects if running in Google Colab, used to adapt paths and token handling
- `get_hf_token(token=None)`: Retrieves HuggingFace token from multiple sources in priority order: (1) explicit argument, (2) Colab secrets if in Colab, (3) `HF_TOKEN` environment variable, (4) `huggingface-cli` login cache. Returns `None` if not found, never crashes.

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
| v3.0.0 | Refactored monolithic Colab notebook into pip-installable Python package (`uvlm/`) with 8 modules; added local Jupyter notebook interface; eliminated all global state (model_ctx dict pattern); added programmatic API for script usage; added pyproject.toml for GitHub-based pip install; added environment detection and unified HF token retrieval; no behavioral changes to inference, parsing, or consensus logic |

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
| `torch` | PyTorch backend (install separately with CUDA for GPU) |
| `transformers` | Model loading, processors, generation |
| `accelerate` | Device placement, memory management |
| `bitsandbytes` | 4-bit and 8-bit quantization |
| `qwen-vl-utils` | Qwen vision preprocessing (`process_vision_info`) |
| `Pillow` | Image loading and conversion |
| `ipywidgets` | Interactive notebook UI |
| `pandas` | CSV management and batch output |
| `numpy` | Numerical operations |
| `requests` | URL-based image loading |
| `huggingface-hub` | Token management and model downloads |

**Runtime**: Google Colab with GPU (T4 free-tier or A100 Pro), or a local machine with an NVIDIA GPU and CUDA.

---

## 7. Repository Structure

```
UVLM/
├── pyproject.toml                          # Package metadata and dependencies
├── README.md                               # Repository landing page with quick start guide
├── LICENSE                                 # Apache License 2.0
├── .gitignore                              # Python/Jupyter artifacts
├── uvlm/                                   # Core Python package
│   ├── __init__.py                         # Version and public API exports
│   ├── loader.py                           # load_model()
│   ├── inference.py                        # run_inference()
│   ├── parsers.py                          # parse_response(), parse_advanced_reasoning_response()
│   ├── consensus.py                        # compute_consensus()
│   ├── batch.py                            # run_batch()
│   ├── prompts.py                          # TASK_TYPES, ADVANCED_REASONING_FORMATS, build_prompt()
│   ├── registry.py                         # MODEL_CHOICES, list_models()
│   └── utils.py                            # set_seed(), is_colab(), get_hf_token(), check_truncation()
├── notebooks/
│   ├── UVLM_colab.ipynb                    # Google Colab interface
│   └── UVLM_local.ipynb                    # Local Jupyter interface
├── figure1_architecture.svg                # Architecture diagram
├── figure2_prompt_form.svg                 # Prompt builder example
├── UVLM_Project_Complete_Documentation.md  # This documentation
└── VERSIONS.txt                            # Version history and changelog
```

---

## 8. Limitations and Future Work

### Current Limitations

- Only supports LLaVA-NeXT and Qwen2.5-VL families
- Sequential image processing (no batching across images)
- Single-image inference only (no video frame analysis)
- Largest models (72B+) require multi-GPU setups not available on free-tier Colab or most consumer GPUs

### Planned Future Work

- **Additional VLM families**: InternVL, BLIP-2, CogVLM, DeepSeek-VL, Molmo, GLM-V
- **Multi-GPU batching**: Parallel inference across images on multi-device setups
- **Video frame analysis**: Temporal visual tasks
- **API mode**: Cloud deployment for integration with automated pipelines
- **Automatic prompt optimization**: Reduce prompt engineering burden
- **SAGAI integration**: UVLM as the VLM inference engine within the SAGAI workflow

---

*Document version: v3.0.0 — April 2026*
*Corresponding author: Joan Perez (Urban Geo Analytics)*
