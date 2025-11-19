#!/usr/bin/env python3
"""Convert images of handwritten notes into Obsidian Markdown files.

Usage examples:
    python ocr_to_obsidian.py --input note.jpg
    python ocr_to_obsidian.py --input ./scans --output ./obsidian_notes --use-llm
"""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import cv2
from skimage import filters

from PIL import Image, ImageEnhance, ImageFilter
import google.generativeai as genai

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}

# Lazy load spellchecker
_spellchecker = None


def get_spellchecker():
    """Lazy load spellchecker to avoid import cost if not needed."""
    global _spellchecker
    if _spellchecker is None:
        try:
            from spellchecker import SpellChecker

            _spellchecker = SpellChecker()
        except ImportError:
            print(
                "Warning: pyspellchecker not installed. Install with: pip install pyspellchecker"
            )
            print("Spell checking disabled.")
            _spellchecker = False
    return _spellchecker if _spellchecker is not False else None


def enhance_math_formatting(text: str, verbose: bool = False) -> str:
    """Enhance mathematical equation formatting for better Obsidian rendering.

    Ensures equations are properly wrapped in LaTeX delimiters and common
    patterns are converted to LaTeX notation.
    """
    if verbose:
        print(f"  Enhancing math formatting...")

    import re

    lines = text.split("\n")
    enhanced_lines = []

    # Common math patterns that should be in LaTeX
    patterns = [
        # Fractions: a/b -> \frac{a}{b}
        (r"\b(\w+)/(\w+)\b", r"$\\frac{\1}{\2}$"),
        # Exponents: x^2 -> x^{2} (if not already wrapped)
        (r"(?<!\$)([a-zA-Z])\^([0-9]+)(?!\$)", r"$\1^{\2}$"),
        # Subscripts: x_i -> x_{i}
        (r"(?<!\$)([a-zA-Z])_([a-zA-Z0-9]+)(?!\$)", r"$\1_{\2}$"),
        # Common operators
        (r"\s*<=\s*", r" $\\leq$ "),
        (r"\s*>=\s*", r" $\\geq$ "),
        (r"\s*!=\s*", r" $\\neq$ "),
        (r"\s*~=\s*", r" $\\approx$ "),
    ]

    for line in lines:
        enhanced = line

        # Only apply to lines that look like they contain math
        if any(char in line for char in ["/", "^", "_", "=", "<", ">", "∫", "∑", "√"]):
            for pattern, replacement in patterns:
                enhanced = re.sub(pattern, replacement, enhanced)

        enhanced_lines.append(enhanced)

    result = "\n".join(enhanced_lines)

    if verbose and result != text:
        print(f"  Enhanced mathematical notation")

    return result


def fix_spelling(text: str, verbose: bool = False) -> str:
    """Fix common spelling mistakes using local dictionary.

    Fast, offline spell checking without API calls.
    Only corrects clear mistakes, preserves proper nouns and technical terms.
    """
    checker = get_spellchecker()
    if not checker:
        return text

    if verbose:
        print(f"  Running spell check...")

    lines = text.split("\n")
    corrected_lines = []
    corrections_made = 0

    for line in lines:
        words = line.split()
        corrected_words = []

        for word in words:
            # Preserve punctuation
            stripped = word.strip(".,!?;:\"''\"()[]{}").lower()

            # Skip short words, numbers, and words with capitals (likely proper nouns)
            if (
                len(stripped) <= 2
                or stripped.isdigit()
                or any(c.isupper() for c in word)
            ):
                corrected_words.append(word)
                continue

            # Check if misspelled
            if stripped not in checker:
                # Get correction
                correction = checker.correction(stripped)
                if correction and correction != stripped:
                    # Preserve original case pattern
                    if word[0].isupper():
                        correction = correction.capitalize()
                    # Replace in original word (preserve punctuation)
                    corrected = word.replace(stripped, correction)
                    corrected_words.append(corrected)
                    corrections_made += 1
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        corrected_lines.append(" ".join(corrected_words))

    if verbose and corrections_made > 0:
        print(f"  Fixed {corrections_made} spelling error(s)")

    return "\n".join(corrected_lines)


def preprocess_image(
    img: Image.Image,
    save_path: Path = None,
    verbose: bool = True,
    window_size: int = 31,
    k_param: float = 0.2,
) -> Image.Image:
    """Preprocess image for better OCR accuracy on black pen on white paper.

    Uses advanced pipeline with illumination correction and Sauvola thresholding.
    Optimized for handwritten notes with black/blue pen on white paper.

    Pipeline:
    1. Bilateral filter (denoise while preserving edges)
    2. Illumination correction (handle uneven lighting)
    3. Sauvola adaptive thresholding
    4. Morphological opening (remove small noise)
    """
    if verbose:
        print(f"  Preprocessing: Original size {img.size}, mode {img.mode}")

    # Convert PIL to numpy array
    if img.mode == "RGB":
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img_array = np.array(img.convert("RGB"))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Step 1: Grayscale + bilateral filter (denoise while preserving edges)
    if verbose:
        print(f"  Step 1/4: Converting to grayscale + bilateral filter...")
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    clean = cv2.bilateralFilter(gray, 9, 75, 75)

    # Step 2: Illumination correction (handle uneven lighting/shadows)
    if verbose:
        print(f"  Step 2/4: Correcting illumination...")
    bg = cv2.medianBlur(clean, 31)
    norm = cv2.divide(clean, bg, scale=255)

    # Step 3: Sauvola adaptive thresholding
    if verbose:
        print(
            f"  Step 3/4: Applying Sauvola thresholding (window={window_size}, k={k_param})..."
        )
    thresh = filters.threshold_sauvola(norm, window_size=window_size, k=k_param)
    binary = (norm > thresh).astype("uint8") * 255

    # Step 4: Small noise removal (morphological opening)
    if verbose:
        print(f"  Step 4/4: Removing small noise...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Convert back to PIL Image
    img = Image.fromarray(clean_binary)

    if verbose:
        print(f"  Preprocessing complete - clean binary image created")

    # Save preprocessed image for inspection
    if save_path:
        debug_dir = save_path.parent / "debug_preprocessed"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"{save_path.stem}_preprocessed.png"
        img.save(debug_path)
        if verbose:
            print(f"  Saved preprocessed image: {debug_path}")

    return img


def ocr_image(
    path: Path,
    max_retries: int = 3,
    verbose: bool = True,
    skip_preprocess: bool = False,
    window_size: int = 31,
    k_param: float = 0.2,
) -> str:
    """Return OCR'd text for a single image path using Gemini Vision API.

    Uses Google Gemini's vision capabilities to extract text from images.
    Includes image preprocessing and retry logic for rate limit errors.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Set it in .env file or export GOOGLE_API_KEY='your-key'"
        )

    if verbose:
        print(f"  Loading image...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Load and optionally preprocess image
    img = Image.open(path)

    if skip_preprocess:
        if verbose:
            print(f"  Skipping preprocessing (using raw image)")
    else:
        img = preprocess_image(
            img,
            save_path=path,
            verbose=verbose,
            window_size=window_size,
            k_param=k_param,
        )

    # Prompt optimized for handwritten notes including math equations
    prompt = """Extract all visible text from this image exactly as written.

Rules:
- Output ONLY the text you see, preserving line breaks
- For mathematical equations and expressions, use LaTeX notation:
  - Fractions: \frac{a}{b}
  - Superscripts: x^2, e^{-x}
  - Subscripts: x_i, a_{n+1}
  - Greek letters: \alpha, \beta, \theta, etc.
  - Common operators: \times, \div, \leq, \geq, \approx, \neq
  - Roots: \sqrt{x}, \sqrt[n]{x}
  - Integrals: \int, \sum, \prod
  - Wrap inline math in $...$ and display math in $$...$$
- If handwriting is unclear, use your best interpretation
- Maintain the original structure and spacing
- For regular text, do not add markdown formatting unless clearly written
"""

    if verbose:
        print(f"  Sending to Gemini API for OCR...")

    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, img])
            text = response.text.strip()
            if verbose:
                print(f"  Extracted {len(text)} characters")
            return text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) * 5  # 5s, 10s, 20s
                    print(f"  Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Rate limit exceeded after {max_retries} attempts. "
                        "Try again later or use gemini-1.5-pro with paid quota."
                    ) from e
            else:
                raise


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def format_filename_for_obsidian(path: Path) -> str:
    # Use stem as title, fallback to timestamp if empty
    title = path.stem
    if not title:
        title = path.name
    return title


def process_file(
    path: Path,
    output_dir: Path,
    use_llm: bool,
    verbose: bool = True,
    skip_preprocess: bool = False,
    window_size: int = 31,
    k_param: float = 0.2,
    spell_check: bool = False,
    enhance_math: bool = False,
) -> Path:
    if verbose:
        print(f"  Step 1/4: Running OCR...")
    raw = ocr_image(
        path,
        verbose=verbose,
        skip_preprocess=skip_preprocess,
        window_size=window_size,
        k_param=k_param,
    )

    # Apply math enhancement if requested
    if enhance_math:
        if verbose:
            print(f"  Step 2/4: Enhancing math formatting...")
        raw = enhance_math_formatting(raw, verbose=verbose)

    # Apply spell checking if requested (after math to avoid breaking LaTeX)
    if spell_check:
        step_num = "3" if enhance_math else "2"
        if verbose:
            print(f"  Step {step_num}/4: Checking spelling...")
        raw = fix_spelling(raw, verbose=verbose)

    # Calculate final step number
    steps_taken = 1 + (1 if enhance_math else 0) + (1 if spell_check else 0)
    if verbose:
        print(
            f"  Step {steps_taken + 1}/{steps_taken + 1}: Creating Obsidian markdown file..."
        )

    # Create Obsidian-formatted markdown
    title = format_filename_for_obsidian(path)
    from datetime import datetime

    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Format as Obsidian markdown with frontmatter
    markdown_content = f"""---
title: {title}
date: {date_str}
time: {time_str}
source: {path.name}
---

# {title}

**Date:** {date_str}  
**Time:** {time_str}

---

{raw}
"""

    # Save as .md file for Obsidian
    out_path = output_dir / (path.stem + ".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown_content, encoding="utf-8")
    return out_path


def gather_images(input_path: Path):
    if input_path.is_dir():
        for p in sorted(input_path.iterdir()):
            if p.is_file() and is_image_file(p):
                yield p
    elif input_path.is_file() and is_image_file(input_path):
        yield input_path
    else:
        return


def main():
    p = argparse.ArgumentParser(
        description="Convert images of handwritten notes to Obsidian Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s -i note.jpg
  %(prog)s -i equation.png --enhance-math
  %(prog)s -i ./scans -o ./vault/notes --spell-check --enhance-math
  %(prog)s -i image.png --no-preprocess
  %(prog)s -i folder/ --window-size 25 -k 0.15
        """,
    )
    p.add_argument(
        "--input", "-i", required=True, help="Image file or directory containing images"
    )
    p.add_argument(
        "--output",
        "-o",
        default="./obsidian_notes",
        help="Output directory for generated .md files (default: ./obsidian_notes)",
    )
    p.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip image preprocessing (use raw image for OCR)",
    )
    p.add_argument(
        "--spell-check",
        action="store_true",
        help="Fix spelling mistakes in extracted text (fast, offline)",
    )
    p.add_argument(
        "--enhance-math",
        action="store_true",
        help="Enhance mathematical equation formatting with LaTeX",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=31,
        help="Sauvola window size for thresholding (default: 31)",
    )
    p.add_argument(
        "-k",
        type=float,
        default=0.2,
        help="Sauvola k parameter (default: 0.2)",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other images if one fails",
    )
    args = p.parse_args()

    verbose = not args.quiet

    verbose = not args.quiet

    # Validate inputs
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found.")
        print("Set it with: export GOOGLE_API_KEY='your-key'")
        print("Or create a .env file with: GOOGLE_API_KEY=your-key")
        return 1

    images = list(gather_images(input_path))
    if not images:
        print(f"Error: No image files found at: {input_path}")
        print(f"Supported formats: {', '.join(IMAGE_EXTS)}")
        return 1

    if verbose:
        print(f"Processing {len(images)} image(s). Output dir: {output_dir}")
        print()

    success_count = 0
    failed_images = []

    for img in images:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {img.name}")
            print(f"{'='*60}")
        try:
            out = process_file(
                img,
                output_dir,
                use_llm=False,
                verbose=verbose,
                skip_preprocess=args.no_preprocess,
                window_size=args.window_size,
                k_param=args.k,
                spell_check=args.spell_check,
                enhance_math=args.enhance_math,
            )
            if verbose:
                print(f"\n✓ Success! Output: {out}")
            success_count += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            error_msg = f"Error processing {img.name}: {e}"
            print(f"\n✗ {error_msg}")
            failed_images.append((img.name, str(e)))

            if not args.continue_on_error:
                print(
                    "\nStopping due to error. Use --continue-on-error to process remaining images."
                )
                break

    if verbose:
        print(f"\n{'='*60}")
        print(f"Complete: {success_count}/{len(images)} files processed successfully")
        if failed_images:
            print(f"\nFailed images:")
            for name, error in failed_images:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")

    return 0 if success_count == len(images) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
