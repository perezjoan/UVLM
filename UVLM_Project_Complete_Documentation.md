# UVLM: Complete Project Documentation

## Executive Summary

**UVLM (Unified Vision-Language Model Loader)** is a pip-installable Python package for reproducible benchmarking of Vision-Language Models (VLMs). It provides a unified interface for loading, configuring, and evaluating multiple VLM architectures on custom image analysis tasks. The tool abstracts the substantial architectural differences between VLM families — currently LLaVA-NeXT, Qwen2.5-VL, Qwen3-VL, InternVL3.5, and Gemma 4 — behind a single inference function, enabling researchers to compare models using identical prompts, evaluation protocols, and explicit, comparable vision budgets without writing model-specific code.

UVLM is distributed as a Python package (`uvlm/`) installable from GitHub, with two interactive notebook interfaces: a Google Colab notebook for zero-install cloud access and a local Jupyter notebook for researchers with their own GPU hardware.

**Current version**: v4.1.0  
**License**: Apache License 2.0  
**Repository**: https://github.com/perezjoan/UVLM  
**Paper**: Perez, J., Fusco, G. (2026). UVLM: A Unified Vision-Language Model Loader for Reproducible Multimodal Benchmarking. *Software, 5(3), 30, 20p.*. https://www.mdpi.com/2674-113X/5/3/30

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
- **Input resolution regimes**: any-resolution tile grids (LLaVA-NeXT), smooth pixel budgets (Qwen), dynamic 448 px tiling (InternVL3.5), fixed soft-token caps (Gemma 4) — four different answers to the same question of how much of the image the model gets to see

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
│   ├── registry.py             # Model registries (24 checkpoints, 5 families)
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

The Colab notebook (`notebooks/UVLM_colab.ipynb`) installs the package automatically via `!pip install git+https://github.com/perezjoan/UVLM.git`, mounts Google Drive for image access, and retrieves the HF token from Colab secrets when present (needed only for the gated Gemma 4 family; every other registry checkpoint downloads without an account). Requires only a Google account with GPU runtime (T4 free-tier or A100 Pro).

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

### 2.3 Two-Block Workflow

Since v4.1.0 both notebooks follow a two-block workflow (v2.x–v4.0.x used three blocks; the
run step is now a button inside Block 2):

#### Block 1: Model Loading & Hardware Configuration

**Re-executable to switch models; loading a new model automatically releases the previous one.**

Calls `uvlm.load_model()` which:

- Auto-detects the backend (`llava`, `qwen`, `qwen3`, `internvl`, `gemma4`) from the model name via the registry
- Loads the correct processor and model classes (transformers 5 families route through `AutoModelForImageTextToText`)
- Supports precision modes: FP16, 8-bit quantization, 4-bit quantization (via BitsAndBytes)
- Configures device placement: GPU-only (`cuda:0`), auto (`accelerate` decides), or GPU + CPU offload.
  On small-VRAM cards prefer GPU-only for 4-bit models: auto placement may propose CPU spill, which
  4-bit quantization correctly refuses with a clear memory error
- Applies the optional **vision budget** (Section 3.2): one parameter controlling how much of each image
  every family gets to see, `native` by default
- Returns a `model_ctx` dict containing model, processor, backend, device info, load time, and the
  resolved `vision_budget` (for run manifests)

The notebook adds an **Unload model** button that releases the current model from GPU and RAM and
reports the reclaimed memory; `on_load` also releases automatically, so switching models never stacks
them in VRAM.

**Supported models:**

| Family | Backend | Models (24 checkpoints) |
|--------|---------|--------------------------|
| LLaVA-NeXT | `llava` | Mistral 7B, Vicuna 7B, Vicuna 13B, 34B, LLaMA3 8B, 72B, 110B (`llava-hf/...`) |
| Qwen2.5-VL | `qwen` | 3B, 7B, 32B, 72B Instruct (`Qwen/Qwen2.5-VL-...`) |
| Qwen3-VL | `qwen3` | 2B, 4B, 8B, 32B Instruct (`Qwen/Qwen3-VL-...`) |
| InternVL3.5 | `internvl` | 1B, 2B, 4B, 8B, 14B, 38B (`OpenGVLab/InternVL3_5-...-HF`) |
| Gemma 4 | `gemma4` | E2B, E4B, 12B Instruct (`google/gemma-4-...-it`, gated: one-time license + token) |

**Note on large models**: The 72B and 110B checkpoints exceed single-GPU memory even with 4-bit quantization; their effective use requires multi-GPU environments (not yet implemented). In practice, models up to roughly 34B parameters load on a single GPU (T4, L4, or A100) using 4-bit quantization. **Note on Gemma 4**: under this stack the family runs in FP16 with CPU offload of its Per-Layer Embedding tables; because generic offload hooks copy those tables (about 4.4 GiB on E2B) to the GPU per forward pass, Gemma 4 is not usable on 8 GB cards and realistically needs 12 to 16 GB or more per tier.

#### Block 2: Inference Configuration, Prompt Builder & Run

**Re-executable to modify tasks and prompts; carries two buttons.**

The notebook provides a widget-based form with two actions:

- **Apply paths + tasks + settings** validates the image folder and builds the `task_specs` list and
  generation parameters. It is model-independent: it can be applied before any model is loaded and
  never needs re-applying when models change.
- **Run analysis** executes the applied configuration against the **currently loaded** model, resolving
  the output CSV name (`Score_Analysis_<model>.csv`) at click time. Switching models between runs is
  simply: load in Block 1, click Run again; each model writes its own CSV and resume mode stays
  per-model.

Form contents:

- Multi-task prompt form (up to 10 tasks via `IntSlider`)
- For each task: column name, task prompt, theory section, format specification, task type
- Global generation parameters: temperature, top-p, max tokens (default: 50, range: 1–1500), optional fixed random seed
- Per-task toggles for consensus validation and advanced reasoning

(The v3.x Qwen-only pixel panel is retired: input resolution is now the cross-family vision budget
set in Block 1, Section 3.2.)

**Task types:**

| Type | Description | Parser behavior |
|------|-------------|-----------------|
| `numeric` | Integer/float extraction | Extracts the **last** number via regex `r"-?\d+(?:\.\d+)?"` |
| `category` | Classification labels | Strips common prefixes ("The answer is:", "Based on the image," etc.), returns cleaned text |
| `boolean` | Yes/no answers | Normalizes variations (yes/true/present → `1`, no/false/absent → `0`) |
| `text` | Free-form responses | Returns cleaned text as-is |

**Important**: `parse_numeric` extracts the *last* number found in the response (not the first). This design ensures that when advanced reasoning produces intermediate calculations before the final answer, the correct value is captured.

#### The batch engine behind the Run button

The Run button calls `uvlm.run_batch()` which:

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

### 3.1 Multi-Backend Inference

The `run_inference()` function routes each call to the appropriate backend based on the
`model_ctx["backend"]` value. The function accepts the model context dict and returns a
`(raw_response, token_count)` tuple. Three pipelines coexist: the two legacy paths below
(LLaVA-NeXT and Qwen2.5-VL, kept verbatim from v3.x), and a unified transformers-5 path shared
by `qwen3`, `internvl`, and `gemma4`, which builds the message list and calls
`processor.apply_chat_template(..., tokenize=True, return_dict=True)` in one step, forwarding
any call-time vision-budget kwargs (Section 3.2) into the processor. The legacy pipelines differ
substantially:

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

### 3.2 Vision Budgets and Input Resolution

Introduced in v4.1.0. Every VLM family answers the same question — how much of the image does the
model get to see? — with a different mechanism, and until v4.0.x UVLM handled this inconsistently:
Qwen was silently capped at about 0.5 MP while every other family ran uncapped at its own default.
v4.1.0 makes the choice explicit, symmetric, and recorded.

**Principle: native by default.** `load_model(vision_budget=None)` (the default) applies no override
anywhere: each model runs exactly the preprocessing its authors shipped. Budgets are opt-in.

**One parameter, four mechanisms.** A budget is translated into each family's own official
preprocessing control and applied where that family's architecture requires:

| Family | Official knob | Applied | Mechanism |
|--------|---------------|---------|-----------|
| Qwen / Qwen3 | `min_pixels` / `max_pixels` | processor init | image smoothly resized so its area fits the pixel budget, then cut into 28 px patches |
| InternVL3.5 | `crop_to_patches`, `min_patches`, `max_patches` | call time (`apply_chat_template` kwargs) | image cropped into up to N fixed 448 px tiles plus a thumbnail |
| Gemma 4 | `max_soft_tokens` in {70, 140, 280, 560, 1120} | call time | encoder output pooled down to at most N vision tokens |
| LLaVA-NeXT | `image_grid_pinpoints` | load time, on **both** `model.config` and the processor | any-resolution grid list reduced; the two copies must agree or generation fails with a patch-count mismatch |

**Presets.** `vision_budget` accepts `"low"`, `"medium"`, `"high"`, calibrated so each stop is the
same class of visual information across families. `"medium"` reproduces the implicit Qwen budget of
UVLM <= 4.0.x (~0.5 MP) and maps it to ~3 InternVL tiles, Gemma's default 280 soft tokens, and a
3-grid LLaVA list. Raw per-backend dicts pass through verbatim after validation, e.g.
`vision_budget={"max_patches": 6}` for InternVL or `{"max_soft_tokens": 1120}` for Gemma.

**Worked example** (a 2048 x 1152 = 2.36 MP photograph):

| Family | medium | high | native |
|--------|--------|------|--------|
| Qwen3 | ~0.5 MP (downscaled ~4.7x) | ~1 MP | full 2.36 MP (default ceiling ~12.8 MP) |
| InternVL3.5 | 3 tiles (~0.6 MP) | 6 tiles (~1.2 MP) | 12 tiles (~2.4 MP) |
| Gemma 4 | 280 soft tokens | 560 | 280 (native equals medium) |
| LLaVA-NeXT | 3 grids | 5 grids | 5 grids (native equals high) |

**High versus native.** These are different concepts, not adjacent rungs. `high` is the top of the
calibrated cross-family ladder; `native` means "whatever each author shipped", which is a
heterogeneous bag — note that Gemma at native sees *less* than at high, and LLaVA's native equals
high. Native is the paper-faithful setting; presets are the comparable settings. Neither is
neutral; a benchmark must state which it uses.

**Comparability.** A shared preset equalizes the *information budget* (the order of magnitude of
visual signal per image) but deliberately not the *preprocessing architecture*: a tiler preserves
local detail a downscaler blurs, a token pooler trades spatially. That residual difference is part
of what is being benchmarked — each family's vision pipeline is a design decision inseparable from
the checkpoint. The well-posed comparison is therefore "model plus its native preprocessing, under
a stated, comparable budget", which is exactly what presets provide. Identical input tensors across
architectures are impossible even at native.

**Memory.** Budgets are also the memory lever: InternVL generation memory scales roughly linearly
with tile count (about 110 MB per tile on an 8B at 4-bit), and Qwen2.5-VL at native may process
images at its very large default ceiling, exceeding small GPUs where UVLM previously capped it
silently — on such hardware select a preset. Gemma 4's memory floor is set by its Per-Layer
Embedding tables, not by resolution: no budget makes it fit an 8 GB card.

**Recording.** The resolved budget is returned in `model_ctx["vision_budget"]` (None for native),
so downstream run manifests can prove which budget produced which results. Legacy
`qwen_min_pixels` / `qwen_max_pixels` kwargs remain supported, override `vision_budget` for the
Qwen families, and are recorded the same way.

### 3.3 Prompt Engineering

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

### 3.4 Consensus Validation

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

### 3.5 Reasoning Support

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

### 3.6 Truncation Detection

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

### 3.7 Response Parsing

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

### 3.8 CSV Schema and Resume Mode

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

### 3.9 Reproducibility

When the "Fixed seed" checkbox is enabled, `set_seed()` from `uvlm/utils.py` is called:

```python
from uvlm.utils import set_seed

set_seed(42)  # Sets random, numpy, torch, cuda, cudnn deterministic
```


Since v4.1.0 the resolved vision budget is part of the reproducibility record: `model_ctx["vision_budget"]` states exactly which per-family preprocessing overrides (if any) produced a given CSV, and belongs in any run manifest.

### 3.10 Environment Detection and Token Retrieval

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
| v4.0.0–v4.0.1 | Migration to transformers 5; three new families (Qwen3-VL, InternVL3.5, Gemma 4) through the unified `AutoModelForImageTextToText` path with `apply_chat_template` tokenization; registry expanded to 24 checkpoints across 5 families |
| v4.1.0 | **Vision budgets**: explicit, optional, per-family input-resolution control (`vision_budget` on `load_model`: presets low/medium/high or raw per-backend dicts), native preprocessing by default, resolved budget returned in `model_ctx` for manifests; previous implicit ~0.5 MP Qwen cap removed (reproduce with `"medium"`); notebooks restructured to two blocks (Apply is model-independent, Run resolves the loaded model and its output CSV at click time), Unload button plus automatic release on every load; fixed: 4-bit CPU-offload allowance scoped to Gemma 4 (overflowing Qwen/InternVL loads fail with the standard memory error instead of a meta-tensor crash), InternVL config aligned with its untied checkpoints (removes a per-load warning), LLaVA budgets applied at load time to both model config and processor. Tested matrix for this release: qwen3, internvl, gemma4; legacy backends (qwen, llava) functional, best-effort |

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

- Gemma 4 runs FP16-only under this stack, and its Per-Layer Embedding offload pattern makes 8 GB cards infeasible (12 to 16 GB or more per tier)
- Legacy backends (LLaVA-NeXT, Qwen2.5-VL) are functional but outside the tested matrix since v4.1.0; Qwen2.5-VL at native resolution may exceed small GPUs (use a budget preset)
- Sequential image processing (no batching across images)
- Single-image inference only (no video frame analysis)
- Largest models (72B+) require multi-GPU setups not available on free-tier Colab or most consumer GPUs

### Planned Future Work

- **Additional VLM families**: BLIP-2, CogVLM, DeepSeek-VL, Molmo, GLM-V
- **Multi-GPU batching**: Parallel inference across images on multi-device setups
- **Video frame analysis**: Temporal visual tasks
- **Automatic prompt optimization**: Reduce prompt engineering burden

---

*Document version: v4.1.0 — September 2026*
*Corresponding author: Joan Perez (Urban Geo Analytics)*
