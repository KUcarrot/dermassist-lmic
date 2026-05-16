"""
robust_json_parser.py (v2 - 강화 버전)
======================================
이전 버전 대비 개선 사항:
1. 종료 펜스 없는 ```json 블록 처리 (응답 잘림 대응)
2. reasoning 텍스트가 JSON 앞에 있는 경우 처리
3. 응답 잘림 시 JSON 부분 복구 시도

처리 가능한 신규 케이스:
- "...reasoning text...```json\n{...}" (종료 펜스 없음)
- "...thinking process...{...}" (펜스 없이 JSON 시작)
- 응답 끝이 잘린 경우 가능한 만큼 복구
"""

import re
import json
from typing import Dict, Optional


def extract_json_from_response(generated: str) -> Optional[str]:
    """
    Gemma 응답에서 JSON 부분만 추출 (강화 버전).

    처리 우선순위:
    1. 완전한 ```json {...} ``` 블록
    2. 시작만 있는 ```json {...} (종료 펜스 없음)
    3. reasoning 후 단순 {...}
    4. 첫 { 부터 마지막 } 까지 (폴백)
    """
    text = generated.strip()

    # 1. 완전한 ```json 블록
    json_fence_match = re.search(
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
        text,
    )
    if json_fence_match:
        return json_fence_match.group(1)

    # 2. 시작 펜스만 있는 경우 (종료 펜스 없음, 응답 잘림 가능)
    # ```json 다음에 오는 모든 내용을 JSON으로 시도
    fence_start_match = re.search(r"```(?:json)?\s*(\{[\s\S]*)$", text)
    if fence_start_match:
        candidate = fence_start_match.group(1)
        # 마지막 ```가 있으면 제거
        candidate = re.sub(r"```\s*$", "", candidate).strip()
        # 마지막 } 찾기
        last_brace = candidate.rfind("}")
        if last_brace > 0:
            candidate = candidate[:last_brace + 1]
        return candidate

    # 3. reasoning 후 단순 JSON 시작 (펜스 없음)
    # "thinking text\n{...}" 형식
    # 첫 번째 { 가 줄바꿈 후에 오는 패턴
    standalone_json = re.search(
        r"(?:^|\n)\s*(\{[\s\S]*\})",
        text,
    )
    if standalone_json:
        return standalone_json.group(1)

    # 4. 폴백: 첫 { 부터 마지막 } 까지
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]

    return None


def fix_truncated_json(json_str: str) -> Optional[str]:
    """
    잘린 JSON을 가능한 만큼 복구.

    예시:
      Input:  '{"a": "value", "b": [1, 2'
      Output: '{"a": "value", "b": [1, 2]}'

    전략:
      1. 미완성된 마지막 키-값 쌍 제거
      2. 미닫힌 brace/bracket 닫기
    """
    text = json_str.strip()

    # 마지막 완성된 ", " 또는 "}, "에서 자르기
    # 가장 안전한 끝점 찾기
    last_safe_end = -1

    # ", \n  "X" : 패턴 또는 "}, \n" 패턴 찾기
    # 즉 새 키가 시작되기 직전의 콤마 위치
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
                # 완전히 닫힌 외부 brace
                return text[:i + 1]
        elif char == '[':
            depth_bracket += 1
        elif char == ']':
            depth_bracket -= 1
        elif char == ',' and depth_brace == 1 and depth_bracket == 0:
            # 외부 객체의 키-값 구분 콤마
            last_complete = i

    # 외부 brace 닫히지 않은 경우 마지막 완성 콤마에서 자르기
    if last_complete > 0:
        truncated = text[:last_complete]
        # 미닫힌 괄호 닫기
        # 추가 분석 필요시 자동으로
        truncated = truncated.rstrip().rstrip(',')

        # 미닫힌 } 추가
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
    ABCDE 필드의 "Key": "Label": "Value" 형식 자동 교정.
    """
    pattern = r'"([A-E])"\s*:\s*"([^"]+)"\s*:\s*"([^"]+)"'
    replacement = r'"\1": "\2: \3"'
    return re.sub(pattern, replacement, json_str)


def remove_hallucinated_multilingual(text: str) -> str:
    """다국어 환각 제거."""
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
    """강화된 JSON 파싱."""
    # Step 1: JSON 추출
    json_str = extract_json_from_response(generated)
    if json_str is None:
        if debug:
            print("[Parser] JSON 추출 실패")
        return _fallback_response(generated)

    # Step 2: 직접 파싱 시도
    try:
        result = json.loads(json_str)
        if debug:
            print("[Parser] 직접 파싱 성공")
        return _clean_response(result)
    except json.JSONDecodeError as e:
        if debug:
            print(f"[Parser] 직접 파싱 실패: {e}")

    # Step 3: ABCDE double colon 교정 후 재시도
    try:
        fixed = fix_abcde_double_colon(json_str)
        result = json.loads(fixed)
        if debug:
            print("[Parser] ABCDE 교정 후 파싱 성공")
        return _clean_response(result)
    except json.JSONDecodeError as e:
        if debug:
            print(f"[Parser] ABCDE 교정 후도 실패: {e}")

    # Step 4: 다국어 환각 제거 후 재시도
    try:
        cleaned = remove_hallucinated_multilingual(json_str)
        cleaned = fix_abcde_double_colon(cleaned)
        last_brace = cleaned.rfind("}")
        if last_brace > 0:
            cleaned = cleaned[:last_brace + 1]
        result = json.loads(cleaned)
        if debug:
            print("[Parser] 다국어 환각 제거 후 파싱 성공")
        return _clean_response(result)
    except json.JSONDecodeError as e:
        if debug:
            print(f"[Parser] 다국어 제거 후도 실패: {e}")

    # Step 5: NEW - 잘린 JSON 복구 시도
    try:
        truncated = fix_truncated_json(json_str)
        if truncated:
            truncated = fix_abcde_double_colon(truncated)
            result = json.loads(truncated)
            if debug:
                print("[Parser] 잘린 JSON 복구 후 파싱 성공")
            return _clean_response(result)
    except json.JSONDecodeError as e:
        if debug:
            print(f"[Parser] 잘린 JSON 복구도 실패: {e}")

    # Step 6: 정규식으로 핵심 필드만 강제 추출
    try:
        result = _extract_fields_by_regex(json_str)
        if result:
            if debug:
                print("[Parser] 정규식 추출 성공")
            return _clean_response(result)
    except Exception as e:
        if debug:
            print(f"[Parser] 정규식 추출 실패: {e}")

    return _fallback_response(generated)


def _extract_fields_by_regex(json_str: str) -> Optional[Dict]:
    """JSON 파싱 실패 시 정규식으로 핵심 필드 추출."""
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
    """응답에서 환각 제거 및 정리."""
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
    """파싱 실패 시 영어 폴백 응답."""
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
# 단위 테스트 (실제 발견된 패턴 기반)
# ============================================================
if __name__ == "__main__":
    # Test 1: reasoning 후 JSON (종료 펜스 없음)
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
    print("Test 1: 종료 펜스 없는 ```json 블록")
    print("=" * 60)
    result1 = parse_gemma_response(test1, debug=True)
    print(f"Result: {list(result1.keys()) if 'observed_features' in result1 else 'FAILED'}\n")

    # Test 2: reasoning + 잘린 JSON
    test2 = """6.  **Final JSON Generation.** (Proceeding with generating the response.)```json
{
  "observed_features": [
    "Vision Classifier confidence: 59.9%",
    "Grad-CAM: borderline"
  ],
  "abcde_analysis": {
    "A": "Asymmetry - Further"""
    print("=" * 60)
    print("Test 2: reasoning + 잘린 JSON")
    print("=" * 60)
    result2 = parse_gemma_response(test2, debug=True)
    print(f"Result: {list(result2.keys()) if 'observed_features' in result2 else 'FAILED'}")
    if 'abcde_analysis' in result2:
        print(f"  ABCDE: {result2.get('abcde_analysis')}")
    print()

    # Test 3: 더 심하게 잘린 케이스 (실제 ISIC_0024800 패턴)
    test3 = """{
  "observed_features": [
    "Test feature"
  ],
  "abcde_analysis": {
    "A": "Asymmetry - Further"""
    print("=" * 60)
    print("Test 3: 마지막 키-값 잘림")
    print("=" * 60)
    result3 = parse_gemma_response(test3, debug=True)
    print(f"Result: {list(result3.keys())}")
    print()