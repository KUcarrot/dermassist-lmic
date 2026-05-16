"""
font_loader.py
==============
Embed Noto Sans fonts as base64 into CSS for Gradio UI.

Usage:
    from dermassist.ui.font_loader import build_font_css
    custom_css = build_font_css() + "..."

Required files (relative to project root):
    fonts/NotoSans-Regular.woff2  (font-weight: 400)
    fonts/NotoSans-Medium.woff2   (font-weight: 500)
    fonts/NotoSans-Bold.woff2     (font-weight: 700)

Falls back to system fonts if font files are missing (empty CSS returned).
"""

import base64
from pathlib import Path
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=None)
def _encode_font_base64(font_path: str) -> Optional[str]:
    """Encode font file as base64 (cached)."""
    path = Path(font_path)
    if not path.exists():
        print(f"[Font] File not found: {path}")
        return None

    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        print(f"[Font] Loaded: {path.name} ({len(encoded)/1024:.0f} KB base64)")
        return encoded
    except Exception as e:
        print(f"[Font] Load failed: {path.name}: {e}")
        return None


def build_font_css(fonts_dir: Optional[Path] = None) -> str:
    """
    Build CSS with Noto Sans fonts embedded as base64.

    Args:
        fonts_dir: Font directory path. If None, uses project_root/fonts/.

    Returns:
        CSS string with @font-face definitions and body font-family rules.
        Returns empty string if font loading fails (falls back to system fonts).
    """
    if fonts_dir is None:
        # This file: src/dermassist/ui/font_loader.py
        # Project root: 4 levels up
        fonts_dir = Path(__file__).resolve().parent.parent.parent.parent / "fonts"
    else:
        fonts_dir = Path(fonts_dir)

    if not fonts_dir.exists():
        print(f"[Font] fonts/ directory not found: {fonts_dir}")
        return ""

    # Load 3 font weights
    regular = _encode_font_base64(fonts_dir / "NotoSans-Regular.woff2")
    medium = _encode_font_base64(fonts_dir / "NotoSans-Medium.woff2")
    bold = _encode_font_base64(fonts_dir / "NotoSans-Bold.woff2")

    if not regular:
        print("[Font] Regular weight not found - falling back to system fonts")
        return ""

    # Build CSS
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

    /* Monospace elements: preserve fixed-width fonts */
    .gradio-container code,
    .gradio-container pre,
    .gradio-container [data-testid="textbox"] textarea {
      font-family: 'Consolas', 'Monaco', monospace !important;
    }
    """)

    return "\n".join(css_parts)


# ============================================================
# Unit test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Font Loader Test")
    print("=" * 60)

    css = build_font_css()
    if css:
        print(f"\n[Success] CSS generated ({len(css)/1024:.0f} KB)")
        print(f"\n[CSS preview - first 200 chars]")
        print(css[:200])
    else:
        print("\n[Failed] CSS generation failed - check fonts/ directory")
