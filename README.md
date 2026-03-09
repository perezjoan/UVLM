# UVLM: Universal Vision-Language Model Loader

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v2.2.2-brightgreen)](https://github.com/perezjoan/UVLM/releases)
[![Colab Compatible](https://img.shields.io/badge/Google%20Colab-Compatible-yellow.svg)](https://colab.research.google.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

**UVLM** is an open-source Google Colab framework for **reproducible benchmarking of Vision-Language Models (VLMs)**. It provides a unified interface for loading, configuring, and evaluating multiple VLM architectures on custom image analysis tasks — without writing model-specific inference code.

UVLM currently supports two major model families — **LLaVA-NeXT** and **Qwen2.5-VL** — which differ fundamentally in their vision encoding, tokenization, and decoding strategies. The framework abstracts these differences behind a single inference function, enabling researchers to compare models using **identical prompts and evaluation protocols**.

💡 **Unified. Reproducible. Accessible. No coding required.**

---

## 🧠 What does UVLM do?

UVLM combines model loading, prompt engineering, and batch evaluation into a single notebook:

- ✅ **11 VLM checkpoints** — 7 LLaVA-NeXT + 4 Qwen2.5-VL models, from 3B to 110B parameters
- 🔧 **Dual-backend abstraction** — automatically routes inference to the correct pipeline (LLaVA or Qwen)
- 📝 **Multi-task prompt builder** — configure up to 10 analysis tasks per run with a widget-based UI
- 🔁 **Consensus validation** — majority voting across 2–5 repeated inferences for improved reliability
- 🧠 **Flexible reasoning support** — adjustable token budget (up to 1,500) for custom chain-of-thought prompts, plus a built-in CoT reference mode for benchmarking
- 🚨 **Truncation detection** — exact token counting flags responses that hit the generation limit, with per-task CSV diagnostics
- 📊 **Batch execution** — process entire image folders with resume capability and CSV output
- ⚡ **Quantization support** — FP16, 8-bit, and 4-bit precision via BitsAndBytes

It requires **no local hardware** — everything runs on Google Colab with free-tier GPU resources.

---

## 📐 Architecture

UVLM is organized into **three sequential blocks**, each handling a distinct stage of the benchmarking workflow:

<p align="center">
  <img src="figure1_architecture.svg" alt="UVLM Architecture Diagram" width="100%"/>
</p>

### Supported Models

| Family | Model | Parameters | Checkpoint ID |
|--------|-------|------------|---------------|
| **LLaVA-NeXT** | Mistral 7B | 7B | `llava-hf/llava-v1.6-mistral-7b-hf` |
| | Vicuna 7B | 7B | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| | Vicuna 13B | 13B | `llava-hf/llava-v1.6-vicuna-13b-hf` |
| | 34B | 34B | `llava-hf/llava-v1.6-34b-hf` |
| | LLaMA3 8B | 8B | `llava-hf/llama3-llava-next-8b-hf` |
| | 72B | 72B | `llava-hf/llava-next-72b-hf` |
| | 110B | 110B | `llava-hf/llava-next-110b-hf` |
| **Qwen2.5-VL** | 3B Instruct | 3B | `Qwen/Qwen2.5-VL-3B-Instruct` |
| | 7B Instruct | 7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| | 32B Instruct | 32B | `Qwen/Qwen2.5-VL-32B-Instruct` |
| | 72B Instruct | 72B | `Qwen/Qwen2.5-VL-72B-Instruct` |

> ⚠️ **Note**: Models with 72B+ parameters exceed single-GPU memory even with 4-bit quantization and require multi-GPU environments. In practice, models up to 34B can be loaded on a single Colab GPU (T4 or A100) with 4-bit quantization.

### Task Types

| Type | Description | Parser |
|------|-------------|--------|
| `numeric` | Integer/float extraction | Extracts last number via regex |
| `category` | Classification labels | Strips common prefixes, returns cleaned text |
| `boolean` | Yes/no answers | Normalizes to 1/0 |
| `text` | Free-form responses | Returns cleaned text |

---

## 🚀 Quick Start

UVLM runs entirely in **Google Colab** — no local installation needed.

1. **Open the notebook** in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/perezjoan/UVLM/blob/main/UVLM.ipynb)

2. **Select a GPU runtime**: `Runtime` → `Change runtime type` → `T4 GPU`

3. **Run Block 1**: Select a model from the dropdown, choose a precision mode (4-bit recommended), and click "Load model"

4. **Run Block 2**: Define your analysis tasks using the prompt builder form — specify column names, prompts, task types, and optionally enable consensus validation. Adjust the max-token slider (up to 1,500) if your prompts require longer outputs.

5. **Run Block 3**: Point to an image folder on Google Drive and execute — results are saved as CSV

> ⚠️ **Hugging Face token**: Some models (e.g., LLaMA3-based) require authentication. Enable the "Use Hugging Face token" checkbox in Block 1 and paste your token.

---

## 🔑 Key Features

### Dual-Backend Inference

UVLM automatically detects the model family and routes to the correct pipeline:

- **LLaVA**: `LlavaNextProcessor` → joint tokenization → `model.generate()` → full decode → string-based response cleaning
- **Qwen**: `AutoProcessor` + `process_vision_info()` → separate vision preprocessing → `model.generate(GenerationConfig)` → token trimming → batch decode

### Consensus Validation

Run each task 2–5 times per image, with majority voting to determine the final answer. NA values from failed parses are filtered before voting. Agreement ratio tracks reliability across all runs.

### Reasoning Support

UVLM supports two approaches to chain-of-thought reasoning:

- **User-defined**: Write task prompts that request step-by-step explanations and use the max-token slider (up to 1,500) to provide adequate generation budget. This gives full control over reasoning structure.
- **Built-in reference mode**: Enable per-task to trigger a standardized CoT template. The token budget is automatically set to 1,024. Primarily intended for benchmarking — in practice, users are encouraged to design their own reasoning prompts tailored to their specific tasks.

Both approaches store the reasoning trace in a dedicated `{column}_reasoning` CSV column for inspection.

### Truncation Detection

After every inference call, the exact number of generated tokens (counted directly from the model output tensor) is compared against the token limit. Truncated responses are flagged in per-task `{column}_truncated` CSV columns and trigger console warnings, allowing users to identify insufficient token budgets without post-hoc analysis.

### Resume-Safe Batch Processing

Block 3 detects already-processed images and skips them. New tasks added between runs trigger automatic CSV schema upgrading. Checkpoints saved every 5 images.

---

## 🧪 Benchmark

A benchmark of **120 French streetscape images** across **8 models × 2 inference modes** (16 configurations) is included:

🔗 **[Dataset on Zenodo]** *(link to be added upon publication)*

### Key Findings

- **Qwen2.5-VL-32B with reasoning** achieves the best overall proximity score (88.0%)
- **LLaVA Vicuna 7B standard** is a strong alternative (83.1%) at a fraction of the computation cost
- Model size does not predict performance: LLaVA 34B (62.2%) ranks last
- Qwen models have near-perfect parsing reliability (zero failures in standard mode)
- Advanced reasoning helps Qwen models (+4.7 pp for 32B) but hurts most LLaVA models

Five analysis tasks were benchmarked: sidewalk detection, motor vehicle counting, pedestrian entrance counting, street frontage length estimation, and vegetation type classification.

### Benchmark Notebooks

Two pre-configured notebooks replicate the benchmark without requiring manual prompt setup:

| Notebook | Mode | Max tokens |
|----------|------|------------|
| `bench_1_no_reasoning.ipynb` | Standard | 50 |
| `bench_2_with_reasoning.ipynb` | Advanced reasoning | 1024 |

---

## 📦 Repository Contents

| File | Description |
|------|-------------|
| [`UVLM.ipynb`](UVLM.ipynb) | Main notebook (all three blocks) |
| [`bench_1_no_reasoning.ipynb`](bench_1_no_reasoning.ipynb) | Benchmark notebook: standard mode |
| [`bench_2_with_reasoning.ipynb`](bench_2_with_reasoning.ipynb) | Benchmark notebook: reasoning mode |
| [`Benchmarking_by_human.xlsx`](Benchmarking_by_human.xlsx) | Human ground truth (120 images, 5 tasks) |
| [`UVLM_Benchmark_Prompts.docx`](UVLM_Benchmark_Prompts.docx) | Complete prompt specifications |
| [`UVLM_Benchmark_Report.xlsx`](UVLM_Benchmark_Report.xlsx) | Full benchmark results (16 configurations) |
| [`figure1_architecture.svg`](figure1_architecture.svg) | Architecture diagram (Figure 1) |
| [`UVLM_Project_Complete_Documentation.md`](UVLM_Project_Complete_Documentation.md) | Full technical documentation |
| [`NOTICE.md`](NOTICE.md) | Third-party licenses and attributions |
| [`VERSIONS.txt`](VERSIONS.txt) | Version history |
| [`LICENSE`](LICENSE) | Apache License 2.0 |
| `README.md` | This file |

---

## 📚 Citation

If you use UVLM in your research, please cite:

> Perez, J. and Fusco, G. (2026). *UVLM: A Universal Vision-Language Model Loader for Reproducible Multimodal Benchmarking*. arXiv preprint.

### Related Publications

> Perez, J. and Fusco, G. (2025). *Streetscape Analysis with Generative AI (SAGAI): Vision-Language Assessment and Mapping of Urban Scenes*. Geomatica, 77(2), 100063. Available at: https://www.sciencedirect.com/science/article/pii/S1195103625000199

---

## 🪪 License and Attribution

UVLM is released under the [Apache License 2.0](LICENSE). This allows use, modification, and redistribution in academic, commercial, and open-source contexts.

Third-party components used in UVLM:

- [LLaVA-NeXT](https://github.com/haotian-liu/LLaVA) — Visual instruction tuning models (Apache 2.0)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Vision-language models (Apache 2.0)
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — Model loading and inference (Apache 2.0)
- [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — Quantization library (MIT)
- [CLIP](https://github.com/openai/CLIP) — Vision encoder used in LLaVA (MIT)

See [`NOTICE.md`](NOTICE.md) for complete attribution details.

---

## ✨ Acknowledgments

This research is supported by the [emc2 project](https://emc2-dut.org/) co-funded by **ANR (France)**, **FFG (Austria)**, **MUR (Italy)**, and **Vinnova (Sweden)** under the **Driving Urban Transition Partnership**, which has been co-funded by the European Commission.

## 🏢 Developer

UVLM is developed by [Joan Perez](https://orcid.org/0000-0003-3003-0895), founder of **Urban Geo Analytics** — an independent research and consulting practice focused on geospatial modeling, AI for cities, and open-source urban analytics. 🌐 [urbangeoanalytics.com](https://urbangeoanalytics.com/)

---

## 📫 Feedback and Contributions

Feel free to open an issue or pull request. Contributions and forks are welcome!

🔗 [GitHub Discussions](https://github.com/perezjoan/UVLM/discussions) — Share use cases, ideas, and extensions.
