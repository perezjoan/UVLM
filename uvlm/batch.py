import json
import os
import time
import pandas as pd

from .inference import run_inference
from .parsers import parse_response, parse_advanced_reasoning_response
from .consensus import compute_consensus
from .prompts import ADVANCED_REASONING_MAX_TOKENS
from .utils import set_seed, check_truncation


def run_batch(
    model_ctx: dict,
    task_specs: list,
    image_folder: str,
    output_path: str,
    max_new_tokens: int = 50,
    do_sample: bool = True,
    temperature: float = 0.3,
    top_p: float = 0.9,
    seed=None,
    display_images: bool = False,
    image_extensions: tuple = (".jpg", ".jpeg", ".png"),
    checkpoint_every: int = 3,
) -> pd.DataFrame:
    """
    Process all images in a folder through all configured tasks.
    Supports resume mode, schema upgrade, consensus, and advanced reasoning.
    Returns the final DataFrame.
    """
    from PIL import Image
    from IPython.display import display

    set_seed(seed)

    # Build column headers
    task_columns = [spec["column"] for spec in task_specs]
    consensus_columns = []
    reasoning_columns = []
    truncated_columns = []
    raw_columns = []

    for spec in task_specs:
        col = spec["column"]
        raw_columns.append(f"{col}_raw")

        if spec.get("consensus_enabled", False):
            consensus_columns.append(f"{col}_consensus")
            consensus_columns.append(f"{col}_agreement")
            consensus_columns.append(f"{col}_runs")

        truncated_columns.append(f"{col}_truncated")

        if spec.get("advanced_reasoning", False):
            reasoning_columns.append(f"{col}_reasoning")

    header = (
        ["image_name"]
        + task_columns
        + reasoning_columns
        + truncated_columns
        + consensus_columns
        + raw_columns
    )

    start_time = time.time()

    # Load or create CSV
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        df = pd.read_csv(output_path, dtype=str)
        print(f"Resuming previous run — {len(df)} rows already in CSV.")
    else:
        df = pd.DataFrame(columns=["image_name"])
        print("Starting fresh — no previous CSV found.")

    if "image_name" not in df.columns:
        raise RuntimeError("Existing CSV has no 'image_name' column. Cannot resume safely.")

    df["image_name"] = df["image_name"].astype(str)

    # Schema upgrade
    missing_cols = [c for c in header if c not in df.columns]
    if missing_cols:
        print("Existing CSV is missing columns; upgrading schema:")
        for c in missing_cols:
            print(f"  + adding column: {c}")
            df[c] = "NA"
        print("CSV schema upgraded.\n")

    ordered = header + [c for c in df.columns if c not in header]
    df = df[ordered]

    # Build image list
    image_files = [
        f for f in sorted(os.listdir(image_folder))
        if f.lower().endswith(image_extensions)
    ]

    print(f"Found {len(image_files)} images in folder.")

    consensus_tasks = [s for s in task_specs if s.get("consensus_enabled", False)]
    if consensus_tasks:
        print(f"Consensus enabled for {len(consensus_tasks)} task(s):")
        for ct in consensus_tasks:
            print(f"   - {ct['column']}: {ct['consensus_runs']} runs")

    reasoning_tasks = [s for s in task_specs if s.get("advanced_reasoning", False)]
    if reasoning_tasks:
        print(f"Advanced reasoning enabled for {len(reasoning_tasks)} task(s):")
        for rt in reasoning_tasks:
            print(f"   - {rt['column']} (max_tokens={ADVANCED_REASONING_MAX_TOKENS})")

    print()

    idx_map = {name: i for i, name in enumerate(df["image_name"].tolist())}

    processed_count = 0
    skipped_count = 0
    total_api_calls = 0

    for fname in image_files:
        image_path = os.path.join(image_folder, fname)
        print(f"\nProcessing: {fname}")

        if display_images:
            try:
                display(Image.open(image_path))
            except Exception as e:
                print(f"Could not display image: {e}")

        if fname not in idx_map:
            new_row = {c: "NA" for c in df.columns}
            new_row["image_name"] = fname
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            idx_map[fname] = len(df) - 1

        row_i = idx_map[fname]
        image_processed = False

        for spec in task_specs:
            col = spec["column"]
            task_prompt = spec["prompt"]
            task_type = spec["task_type"]
            raw_col = f"{col}_raw"
            reasoning_col = f"{col}_reasoning"
            truncated_col = f"{col}_truncated"

            advanced_reasoning = spec.get("advanced_reasoning", False)
            consensus_enabled = spec.get("consensus_enabled", False)
            consensus_runs = spec.get("consensus_runs", 1)
            numeric_tolerance = spec.get("numeric_tolerance", 0.0)

            current = df.at[row_i, col] if col in df.columns else "NA"
            current = "" if pd.isna(current) else str(current).strip()

            needs_compute = (
                (current == "")
                or (current.upper() == "NA")
                or current.startswith("ERROR:")
            )

            if not needs_compute:
                print(f"  {col}: already set ({current})")
                skipped_count += 1
                continue

            try:
                if consensus_enabled and task_type != "text":
                    print(f"  {col}: Running {consensus_runs}x for consensus...")

                    raw_responses = []
                    parsed_values = []
                    reasoning_texts = []

                    for run_i in range(consensus_runs):
                        if advanced_reasoning:
                            raw_response, token_count = run_inference(
                                image_path, task_prompt, model_ctx,
                                max_new_tokens=ADVANCED_REASONING_MAX_TOKENS,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_p=top_p,
                            )
                            parsed_result = parse_advanced_reasoning_response(raw_response, task_type)
                            parsed_value = parsed_result["answer"]
                            reasoning_texts.append(parsed_result["reasoning"])
                        else:
                            raw_response, token_count = run_inference(
                                image_path, task_prompt, model_ctx,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_p=top_p,
                            )
                            parsed_value = parse_response(raw_response, task_type)

                        raw_responses.append(raw_response)
                        parsed_values.append(parsed_value)
                        total_api_calls += 1
                        print(f"      Run {run_i + 1}: {parsed_value}")

                    consensus_result = compute_consensus(parsed_values, task_type, numeric_tolerance)

                    df.at[row_i, col] = consensus_result["final_value"]
                    df.at[row_i, f"{col}_consensus"] = "YES" if consensus_result["consensus_reached"] else "NO"
                    df.at[row_i, f"{col}_agreement"] = str(consensus_result["agreement_ratio"])
                    df.at[row_i, f"{col}_runs"] = json.dumps(consensus_result["all_values"])
                    df.at[row_i, raw_col] = raw_responses[0]

                    if advanced_reasoning and reasoning_texts:
                        df.at[row_i, reasoning_col] = reasoning_texts[0]

                    effective_max = ADVANCED_REASONING_MAX_TOKENS if advanced_reasoning else max_new_tokens
                    is_truncated, token_count = check_truncation(token_count, effective_max)
                    df.at[row_i, truncated_col] = "YES" if is_truncated else "NO"
                    if is_truncated:
                        print(f"  {col}: TRUNCATION DETECTED — response used {token_count}/{effective_max} tokens. Increase max_tokens!")

                    consensus_icon = "OK" if consensus_result["consensus_reached"] else "WARN"
                    print(f"  [{consensus_icon}] {col} [{task_type}]: {consensus_result['final_value']} (agreement: {consensus_result['agreement_ratio']:.0%})")

                elif advanced_reasoning and task_type != "text":
                    raw_response, token_count = run_inference(
                        image_path, task_prompt, model_ctx,
                        max_new_tokens=ADVANCED_REASONING_MAX_TOKENS,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    parsed_result = parse_advanced_reasoning_response(raw_response, task_type)
                    total_api_calls += 1

                    df.at[row_i, col] = str(parsed_result["answer"]).strip()
                    df.at[row_i, reasoning_col] = str(parsed_result["reasoning"]).strip()
                    df.at[row_i, raw_col] = str(raw_response).strip()

                    is_truncated, token_count = check_truncation(token_count, ADVANCED_REASONING_MAX_TOKENS)
                    df.at[row_i, truncated_col] = "YES" if is_truncated else "NO"

                    print(f"  [reasoning] {col} [{task_type}]: {df.at[row_i, col]}")
                    if is_truncated:
                        print(f"  {col}: TRUNCATION DETECTED — response used {token_count}/{ADVANCED_REASONING_MAX_TOKENS} tokens. Increase max_tokens!")
                    else:
                        preview = parsed_result["reasoning"][:100] + "..." if len(parsed_result["reasoning"]) > 100 else parsed_result["reasoning"]
                        print(f"      (reasoning: {preview})")

                else:
                    raw_response, token_count = run_inference(
                        image_path, task_prompt, model_ctx,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    parsed_result = parse_response(raw_response, task_type)
                    total_api_calls += 1

                    df.at[row_i, col] = str(parsed_result).strip()
                    df.at[row_i, raw_col] = str(raw_response).strip()

                    is_truncated, token_count = check_truncation(token_count, max_new_tokens)
                    df.at[row_i, truncated_col] = "YES" if is_truncated else "NO"

                    print(f"  {col} [{task_type}]: {df.at[row_i, col]}")
                    if is_truncated:
                        print(f"  {col}: TRUNCATION DETECTED — response used {token_count}/{max_new_tokens} tokens. Increase max_tokens!")
                    elif task_type != "text":
                        raw_preview = raw_response[:80] + "..." if len(raw_response) > 80 else raw_response
                        print(f"      (raw: {raw_preview})")

                image_processed = True

            except Exception as e:
                err = f"ERROR: {e}"
                df.at[row_i, col] = err
                df.at[row_i, raw_col] = err
                print(f"  {col}: {err}")

        if image_processed:
            processed_count += 1

        if processed_count > 0 and processed_count % checkpoint_every == 0:
            df.to_csv(output_path, index=False)
            print(f"  Checkpoint saved ({processed_count} images processed)")

    df.to_csv(output_path, index=False)

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("Analysis completed.")
    print(f"Results saved to: {output_path}")
    print(f"Tasks written: {len(task_specs)}")
    for spec in task_specs:
        flags = []
        if spec.get("advanced_reasoning"):
            flags.append("reasoning")
        if spec.get("consensus_enabled"):
            flags.append(f"consensus={spec['consensus_runs']}x")
        flags_str = f" ({', '.join(flags)})" if flags else ""
        print(f"   - {spec['column']} ({spec['task_type']}){flags_str}")
    print(f"Images processed: {processed_count}")
    print(f"Tasks skipped (already done): {skipped_count}")
    print(f"Total API calls: {total_api_calls}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print("=" * 50)

    return df
