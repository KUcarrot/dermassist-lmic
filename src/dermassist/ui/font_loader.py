"""
font_loader.py
==============
Noto Sans 폰트를 base64로 인코딩하여 CSS에 임베드.

사용:
  from font_loader import build_font_css

  custom_css = build_font_css() + "..."

요구 파일:
  fonts/NotoSans-Regular.woff2  (font-weight: 400)
  fonts/NotoSans-Medium.woff2   (font-weight: 500)
  fonts/NotoSans-Bold.woff2     (font-weight: 700)

폰트 파일이 없으면 빈 CSS 반환 (시스템 폰트로 폴백).
"""

import base64
from pathlib import Path
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=None)
def _encode_font_base64(font_path: str) -> Optional[str]:
    """폰트 파일을 base64로 인코딩 (캐시됨)."""
    path = Path(font_path)
    if not path.exists():
        print(f"[Font] 파일 없음: {path}")
        return None

    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        print(f"[Font] 로드 완료: {path.name} ({len(encoded)/1024:.0f} KB base64)")
        return encoded
    except Exception as e:
        print(f"[Font] 로드 실패: {path.name}: {e}")
        return None


def build_font_css(fonts_dir: Optional[Path] = None) -> str:
    """
    Noto Sans 폰트를 base64 임베드한 CSS 생성.

    Args:
        fonts_dir: 폰트 디렉터리. None이면 현재 파일 위치의 fonts/ 사용.

    Returns:
        @font-face 정의 + body font-family CSS.
        폰트 로드 실패 시 빈 문자열 (시스템 폰트로 폴백).
    """
    if fonts_dir is None:
        fonts_dir = Path(__file__).resolve().parent.parent.parent.parent / "fonts"
    else:
        fonts_dir = Path(fonts_dir)

    if not fonts_dir.exists():
        print(f"[Font] fonts/ 디렉터리 없음: {fonts_dir}")
        return ""

    # 3가지 weight 로드
    regular = _encode_font_base64(fonts_dir / "NotoSans-Regular.woff2")
    medium = _encode_font_base64(fonts_dir / "NotoSans-Medium.woff2")
    bold = _encode_font_base64(fonts_dir / "NotoSans-Bold.woff2")

    if not regular:
        print("[Font] Regular 폰트 없음 — 시스템 폰트로 폴백")
        return ""

    # CSS 빌드
    css_parts = []

    # Regular (font-weight: 400)
    css_parts.append(f"""
    @font-face {{
      font-family: 'Noto Sans';
      src: url(data:font/woff2;base64,{regular}) format('woff2');
      font-weight: 400;
      font-style: normal;
      font-display: block;
    }}
    """)

    # Medium (font-weight: 500)
    if medium:
        css_parts.append(f"""
        @font-face {{
          font-family: 'Noto Sans';
          src: url(data:font/woff2;base64,{medium}) format('woff2');
          font-weight: 500;
          font-style: normal;
          font-display: block;
        }}
        """)

    # Bold (font-weight: 700)
    if bold:
        css_parts.append(f"""
        @font-face {{
          font-family: 'Noto Sans';
          src: url(data:font/woff2;base64,{bold}) format('woff2');
          font-weight: 700;
          font-style: normal;
          font-display: block;
        }}
        """)

    # Apply to all elements
    css_parts.append("""
    .gradio-container,
    .gradio-container * {
      font-family: 'Noto Sans', system-ui, -apple-system, sans-serif !important;
    }

    /* Monospace 요소는 예외 처리 */
    .gradio-container code,
    .gradio-container pre,
    .gradio-container [data-testid="textbox"] textarea {
      font-family: 'Consolas', 'Monaco', monospace !important;
    }
    """)

    return "\n".join(css_parts)


# ============================================================
# 단위 테스트
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Font Loader 테스트")
    print("=" * 60)

    css = build_font_css()
    if css:
        print(f"\n[성공] CSS 생성 완료 ({len(css)/1024:.0f} KB)")
        print(f"\n[CSS 시작 부분 200자]")
        print(css[:200])
    else:
        print("\n[실패] CSS 생성 실패 — fonts/ 디렉터리 확인")
