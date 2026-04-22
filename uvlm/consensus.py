from collections import Counter


def compute_consensus(parsed_values: list, task_type: str, numeric_tolerance: float = 0.0) -> dict:
    """
    Compute consensus from multiple parsed values using majority voting.

    UPDATED in v2.2.1: NA values are now properly excluded from consensus.
    When valid values exist, NA is never selected as the final answer.

    Args:
        parsed_values: List of parsed values from multiple runs
        task_type: Type of task (numeric, category, boolean)
        numeric_tolerance: For numeric tasks, values within this % are considered equal

    Returns:
        dict with keys: final_value, consensus_reached, agreement_ratio, all_values
    """
    n_runs = len(parsed_values)

    if n_runs == 0:
        return {
            "final_value": "NA",
            "consensus_reached": False,
            "agreement_ratio": 0.0,
            "all_values": [],
        }

    def is_na_value(v):
        """Check if a value represents NA/missing."""
        if v is None:
            return True
        v_str = str(v).strip().upper()
        return v_str in ("NA", "N/A", "NAN", "NONE", "NULL", "")

    valid_values = [v for v in parsed_values if not is_na_value(v)]

    if len(valid_values) == 0:
        return {
            "final_value": "NA",
            "consensus_reached": False,
            "agreement_ratio": 0.0,
            "all_values": parsed_values,
        }

    if task_type == "numeric" and numeric_tolerance > 0:
        try:
            numeric_vals = [float(v) for v in valid_values]
            groups = []
            used = [False] * len(numeric_vals)

            for i, val in enumerate(numeric_vals):
                if used[i]:
                    continue
                group = [val]
                used[i] = True
                for j, other_val in enumerate(numeric_vals):
                    if not used[j]:
                        if val != 0:
                            diff_pct = abs(other_val - val) / abs(val)
                        else:
                            diff_pct = abs(other_val - val)
                        if diff_pct <= numeric_tolerance:
                            group.append(other_val)
                            used[j] = True
                groups.append(group)

            largest_group = max(groups, key=len)
            final_value = str(round(sum(largest_group) / len(largest_group), 2))
            agreement_count = len(largest_group)

        except (ValueError, ZeroDivisionError):
            counter = Counter(valid_values)
            final_value, agreement_count = counter.most_common(1)[0]
    else:
        counter = Counter(valid_values)
        final_value, agreement_count = counter.most_common(1)[0]

    agreement_ratio = agreement_count / n_runs
    consensus_reached = agreement_ratio > 0.5

    return {
        "final_value": final_value,
        "consensus_reached": consensus_reached,
        "agreement_ratio": round(agreement_ratio, 2),
        "all_values": parsed_values,
    }
