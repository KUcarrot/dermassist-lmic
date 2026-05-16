"""
json_parser.py
==============
Robust JSON parsing for Gemma 4 LLM responses.

Handles common LLM output edge cases:
1. Complete ```json {...} ``` fenced blocks
2. Opened but unclosed ```json blocks (response truncation)
3. Reasoning text preceding the JSON
4. Multilingual hallucinations at the end of strings
5. Truncated JSON recovery
6. ABCDE double-colon formatting issues

Usage:
    from dermassist.llm.json_parser import parse_gemma_response
    parsed = parse_gemma_response(raw_llm_output, debug=False)
"""

import re
import json
from typing import Dict, Optional


def extract_json_from_response(generated: str) -> Optional[str]:
    """
    Extract the JSON portion from a Gemma response.

    Priority order:
    1. Complete ```json {...} ``` block
    2. Opened ```json {...} block without closing fence
    3. Standalone JSON after reasoning text (no fences)
    4. Fallback: first { to last }
    """
    text = generated.strip()

    # 1. Complete ```json block
    json_fence_match = re.search(
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
        text,
    )
    if json_fence_match:
        return json_fence_match.group(1)

    # 2. Only opening fence (no closing fence, possibly truncated response)
    # Attempt to parse everything after ```json as JSON
    fence_start_match = re.search(r"```(?:json)?\s*(\{[\s\S]*)$", text)
    if fence_start_match:
        candidate = fence_start_match.group(1)
        # Remove trailing ``` if present
        candidate = re.sub(r"```\s*$", "", candidate).strip()
        # Find last }
        last_brace = candidate.rfind("}")
        if last_brace > 0:
            candidate = candidate[:last_brace + 1]
        return candidate

    # 3. Standalone JSON after reasoning text (no fences)
    # Pattern: "thinking text\n{...}"
    standalone_json = re.search(
        r"(?:^|\n)\s*(\{[\s\S]*\})",
        text,
    )
    if standalone_json:
        return standalone_json.group(1)

    # 4. Fallback: first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]

    return None


def fix_truncated_json(json_str: str) -> Optional[str]:
    """
    Recover as much of a truncated JSON as possible.

    Example:
        Input:  '{"a": "value", "b": [1, 2'
        Output: '{"a": "value", "b": [1, 2]}'

    Strategy:
        1. Remove incomplete trailing key-value pair
        2. Close unclosed braces/brackets
    """
    text = json_str.strip()

    # Find the safest endpoint
    in_string = False
    escape = False
    depth_brace = 0
    depth_bracket = 0
    last_complete = -1

    for i, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            depth_brace += 1
        elif char == '}':
            depth_brace -= 1
            if depth_brace == 0:
                # Fully closed outer brace
                return text[:i + 1]
        elif char == '[':
            depth_bracket += 1
        elif char == ']':
            depth_bracket -= 1
        elif char == ',' and depth_brace == 1 and depth_bracket == 0:
            # Outer object key-value separator
            last_complete = i

    # If outer brace is not closed, truncate at last complete comma
    if last_complete > 0:
        truncated = text[:last_complete]
        truncated = truncated.rstrip().rstrip(',')

        # Close any unclosed braces/brackets
        opens = truncated.count('{') - truncated.count('}')
        closes = truncated.count('[') - truncated.count(']')

        if closes > 0:
            truncated += ']' * closes
        if opens > 0:
            truncated += '}' * opens

        return truncated

    return None


def fix_abcde_double_colon(json_str: str) -> str:
    """
    Auto-correct ABCDE field "Key": "Label": "Value" formatting.

    Some LLM outputs incorrectly use double colons in ABCDE fields.
    Convert: "A": "Asymmetry": "value"
    To:      "A": "Asymmetry: value"
    """
    pattern = r'"([A-E])"\s*:\s*"([^"]+)"\s*:\s*"([^"]+)"'
    replacement = r'"\1": "\2: \3"'
    return re.sub(pattern, replacement, json_str)


def remove_hallucinated_multilingual(text: str) -> str:
    """Remove multilingual hallucinations from the tail of a response."""
    if len(text) < 100:
        return text

    last_100 = text[-100:]
    non_ascii_count = sum(1 for c in last_100 if ord(c) > 127)

    if non_ascii_count > 50:
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?\"":
                check_segment = text[max(0, i-50):i]
                segment_ascii_ratio = sum(
                    1 for c in check_segment if ord(c) < 128
                ) / max(len(check_segment), 1)
                if segment_ascii_ratio > 0.9:
                    return text[:i + 1]

    return text


def parse_gemma_response(generated: str, debug: bool = False) -> Dict:
    """
    Robust JSON parsing with multi-stage fallback.

    Args:
        generated: Raw LLM output string
        debug: If True, print parsing stage information

    Returns:
        Parsed dict (or fallback response if all parsing fails)
    """
    # Step 1: Extract JSON
    json_str = extract_json_from_response(generated)
    if json_str is None:
        if debug:
            print("[Parser] JSON extraction failed")
        return _fallback_response(generated)

    # Step 2: Try direct parsing
    try:
        result = json.loads(json_str)
        if debug:
            print("[Parser] Direct parsing succeeded")
        return _clean_response(result)
    except json.JSONDecodeError as e:
        if debug:
            print(f"[Parser] Direct parsing failed: {e}")

    # Step 3: Retry after ABCDE double colon fix
    try:
        fixed = fix_abcde_double_colon(json_str)
        result = json.loads(fixed)
        if debug:
            print("[Parser] ABCDE-fixed parsing succeeded")
        return _clean_response(result)
    except json.JSONDecodeError as e:
        if debug:
            print(f"[Parser] ABCDE-fixed parsing failed: {e}")

    # Step 4: Retry after multilingual hallucination removal
    try:
        cleaned = remove_hallucinated_multilingual(json_str)
        cleaned = fix_abcde_double_colon(cleaned)
        last_brace = cleaned.rfind("}")
        if last_brace > 0:
            cleaned = cleaned[:last_brace + 1]
        result = json.loads(cleaned)
        if debug:
            print("[Parser] Multilingual-cleaned parsing succeeded")
        return _clean_response(result)
    except json.JSONDecodeError as e:
        if debug:
            print(f"[Parser] Multilingual cleanup failed: {e}")

    # Step 5: Truncated JSON recovery
    try:
        truncated = fix_truncated_json(json_str)
        if truncated:
            truncated = fix_abcde_double_colon(truncated)
            result = json.loads(truncated)
            if debug:
                print("[Parser] Truncated JSON recovery succeeded")
            return _clean_response(result)
    except json.JSONDecodeError as e:
        if debug:
            print(f"[Parser] Truncated JSON recovery failed: {e}")

    # Step 6: Force-extract key fields via regex
    try:
        result = _extract_fields_by_regex(json_str)
        if result:
            if debug:
                print("[Parser] Regex extraction succeeded")
            return _clean_response(result)
    except Exception as e:
        if debug:
            print(f"[Parser] Regex extraction failed: {e}")

    return _fallback_response(generated)


def _extract_fields_by_regex(json_str: str) -> Optional[Dict]:
    """Last-resort regex extraction when JSON parsing completely fails."""
    result = {}

    string_fields = [
        "classification_summary", "recommendation",
        "patient_summary", "limitations", "urgency",
    ]
    for field in string_fields:
        pattern = rf'"{field}"\s*:\s*"([^"]+(?:\\.[^"]*)*)"'
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            result[field] = match.group(1).replace("\\\"", "\"")

    list_fields = ["observed_features", "evidence_sources"]
    for field in list_fields:
        pattern = rf'"{field}"\s*:\s*\[(.*?)\]'
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            items = re.findall(r'"([^"]+(?:\\.[^"]*)*)"', match.group(1))
            result[field] = [item.replace("\\\"", "\"") for item in items]

    abcde_match = re.search(
        r'"abcde_analysis"\s*:\s*\{([^{}]*)\}', json_str, re.DOTALL,
    )
    if abcde_match:
        abcde = {}
        for letter in "ABCDE":
            patterns = [
                rf'"{letter}"\s*:\s*"([^"]+)"\s*:\s*"([^"]+)"',
                rf'"{letter}"\s*:\s*"([^"]+)"',
            ]
            for pattern in patterns:
                m = re.search(pattern, abcde_match.group(1))
                if m:
                    if len(m.groups()) == 2:
                        abcde[letter] = f"{m.group(1)}: {m.group(2)}"
                    else:
                        abcde[letter] = m.group(1)
                    break
        if abcde:
            result["abcde_analysis"] = abcde

    if len(result) >= 4:
        return result
    return None


def _clean_response(response: Dict) -> Dict:
    """Clean response by removing hallucinations from text fields."""
    if not isinstance(response, dict):
        return response

    text_fields = [
        "classification_summary", "recommendation",
        "patient_summary", "limitations",
    ]
    for field in text_fields:
        if field in response and isinstance(response[field], str):
            response[field] = remove_hallucinated_multilingual(response[field])

    return response


def _fallback_response(generated: str) -> Dict:
    """English fallback response when all parsing strategies fail."""
    return {
        "observed_features": ["JSON parsing failed - see raw output"],
        "abcde_analysis": {k: "Parsing failed" for k in "ABCDE"},
        "classification_summary": generated[:300],
        "evidence_sources": [],
        "recommendation": (
            "Generated response format error - manual review required"
        ),
        "urgency": "soon",
        "patient_summary": (
            "A system error occurred during analysis. "
            "Please consult a dermatology specialist directly for evaluation."
        ),
        "limitations": (
            "This analysis result is unreliable due to a parsing error. "
            "Do not rely on this output for clinical decisions."
        ),
        "_raw_output": generated,
    }


# ============================================================
# Unit tests (based on patterns observed in production)
# ============================================================
if __name__ == "__main__":
    # Test 1: ```json block without closing fence
    test1 = """The patient context is detailed but the patient summary is plain English.)```json
{
  "observed_features": [
    "Vision Classifier: Predicted benign nevus (nv) with 81.3% confidence",
    "Grad-CAM activation: Even"
  ],
  "abcde_analysis": {
    "A": "Symmetry good",
    "B": "Border regular",
    "C": "Color uniform",
    "D": "Diameter unknown",
    "E": "Evolution stable"
  },
  "classification_summary": "Benign nevus",
  "evidence_sources": ["Vision Classifier"],
  "recommendation": "Routine follow-up",
  "urgency": "routine",
  "patient_summary": "Looks benign.",
  "limitations": "AI screening only."
}"""
    print("=" * 60)
    print("Test 1: ```json block without closing fence")
    print("=" * 60)
    result1 = parse_gemma_response(test1, debug=True)
    print(f"Result: {list(result1.keys()) if 'observed_features' in result1 else 'FAILED'}\n")

    # Test 2: reasoning + truncated JSON
    test2 = """6.  **Final JSON Generation.** (Proceeding with generating the response.)```json
{
  "observed_features": [
    "Vision Classifier confidence: 59.9%",
    "Grad-CAM: borderline"
  ],
  "abcde_analysis": {
    "A": "Asymmetry - Further"""
    print("=" * 60)
    print("Test 2: reasoning + truncated JSON")
    print("=" * 60)
    result2 = parse_gemma_response(test2, debug=True)
    print(f"Result: {list(result2.keys()) if 'observed_features' in result2 else 'FAILED'}")
    if 'abcde_analysis' in result2:
        print(f"  ABCDE: {result2.get('abcde_analysis')}")
    print()

    # Test 3: heavily truncated case
    test3 = """{
  "observed_features": [
    "Test feature"
  ],
  "abcde_analysis": {
    "A": "Asymmetry - Further"""
    print("=" * 60)
    print("Test 3: last key-value truncated")
    print("=" * 60)
    result3 = parse_gemma_response(test3, debug=True)
    print(f"Result: {list(result3.keys())}")
    print()
