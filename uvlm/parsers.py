import re


def parse_numeric(raw: str) -> str:
    """Extract the last number from the response."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", raw)
    return numbers[-1] if numbers else "NA"


def parse_category(raw: str) -> str:
    """
    Return cleaned text response for categorical outputs.
    Strips whitespace, removes common prefixes, and handles multi-line.
    """
    text = raw.strip()

    prefixes_to_remove = [
        "The answer is:",
        "Answer:",
        "Category:",
        "Classification:",
        "This is a",
        "This is an",
        "This appears to be a",
        "This appears to be an",
        "I would classify this as",
        "This can be classified as",
        "Based on the image,",
        "Looking at the image,",
    ]

    text_lower = text.lower()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            text_lower = text.lower()

    text = text.rstrip('.')

    if '\n' in text:
        first_line = text.split('\n')[0].strip()
        if first_line and len(first_line) < 100:
            text = first_line

    return text if text else "NA"


def parse_boolean(raw: str) -> str:
    """
    Extract boolean response and normalize to 1/0.
    """
    text = raw.lower().strip()

    positive = ['yes', 'true', 'y', '1', 'correct', 'affirmative',
                'present', 'visible', 'exists', 'found', 'detected']

    negative = ['no', 'false', 'n', '0', 'incorrect', 'negative',
                'absent', 'not visible', 'not present', 'none',
                'not found', 'not detected', 'cannot', "don't", "doesn't"]

    for neg in negative:
        if neg in text:
            return "0"

    for pos in positive:
        if pos in text:
            return "1"

    return "NA"


def parse_text(raw: str) -> str:
    """Return the raw response with minimal cleaning."""
    text = raw.strip()
    text = ' '.join(text.split())
    return text if text else "NA"


def parse_response(raw: str, task_type: str) -> str:
    """Parse model response based on task type."""
    if task_type == "numeric":
        return parse_numeric(raw)
    elif task_type == "category":
        return parse_category(raw)
    elif task_type == "boolean":
        return parse_boolean(raw)
    elif task_type == "text":
        return parse_text(raw)
    else:
        return parse_text(raw)


def parse_advanced_reasoning_response(raw: str, task_type: str) -> dict:
    """
    Parse response from advanced reasoning mode.
    Extracts both the reasoning and the final answer.

    Returns:
        dict with keys: answer, reasoning, raw
    """
    raw_stripped = raw.strip()

    lines = raw_stripped.split('\n')
    answer = "NA"
    reasoning = raw_stripped

    for i in range(len(lines) - 1, max(len(lines) - 5, -1), -1):
        line = lines[i].strip()
        match = re.search(r'answer\s*:\s*(.+)', line, re.IGNORECASE)
        if match:
            answer_text = match.group(1).strip()
            answer = parse_response(answer_text, task_type)
            reasoning = '\n'.join(lines[:i]).strip()
            break

    if answer == "NA":
        answer = parse_response(raw_stripped, task_type)
        reasoning = raw_stripped

    return {
        "answer": answer,
        "reasoning": reasoning,
        "raw": raw_stripped,
    }
