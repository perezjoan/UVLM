TASK_TYPES = {
    "numeric": "Extracts the last number from response (for counting, scores, ratings)",
    "category": "Returns the full text response stripped (for classifications)",
    "boolean": "Extracts yes/no/true/false and normalizes to 1/0",
    "text": "Returns raw response as-is (for descriptions, free-form)",
}

ADVANCED_REASONING_FORMATS = {
    "numeric": (
        "First, describe what you observe and explain your reasoning step by step.\n"
        "Then, on the last line, write only: ANSWER: <integer>"
    ),
    "category": (
        "First, describe what you observe and explain your reasoning step by step.\n"
        "Then, on the last line, write only: ANSWER: <category number>"
    ),
    "boolean": (
        "First, describe what you observe and explain your reasoning step by step.\n"
        "Then, on the last line, write only: ANSWER: <yes or no>"
    ),
}

ADVANCED_REASONING_MAX_TOKENS = 1024


def build_prompt(role: str, task: str, theory: str, format_spec: str) -> str:
    """Assemble a full prompt from the four UVLM prompt components."""
    return f"{role}\n\n{task}\n\n{theory}\n\n{format_spec}".strip()
