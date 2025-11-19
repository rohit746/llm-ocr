# LLM-OCR Project Instructions

## Project Overview

Handwritten notes OCR system that converts images to Obsidian-compatible markdown using Google Gemini Vision API. Main pipeline: image preprocessing → OCR → optional postprocessing (spell check, math formatting) → Obsidian markdown output.

## Architecture

- **`ocr_to_obsidian.py`**: Main CLI orchestrator with preprocessing pipeline and batch processing
- **`llm_client.py`**: Standalone LLM formatter (currently unused in main workflow, but available for text postprocessing)
- Two-module design allows independent use: direct OCR or OCR + LLM cleanup

## Environment Setup

```bash
# This project uses uv for package management (note uv.lock presence)
uv sync                          # Install dependencies
cp .env.example .env             # Configure API key
export GOOGLE_API_KEY='your-key' # Or set in .env
```

Python 3.13+ required (see `pyproject.toml` and `.python-version`).

## Critical Preprocessing Pipeline

The OCR quality depends on a 4-step preprocessing pipeline in `preprocess_image()`:

1. **Bilateral filter** (denoise, preserve edges)
2. **Illumination correction** (handle uneven lighting with median blur + normalize)
3. **Sauvola adaptive thresholding** (window_size=31, k=0.2 are tuned defaults)
4. **Morphological opening** (remove small noise)

Tuning parameters: `--window-size` and `-k` flags adjust Sauvola thresholding. Use `--no-preprocess` to bypass entirely and send raw images to Gemini.

## Development Workflows

```bash
# Basic OCR (single image)
python ocr_to_obsidian.py -i note.jpg

# Batch with all features
python ocr_to_obsidian.py -i ./scans -o ./vault --spell-check --enhance-math

# Debug preprocessing (outputs to debug_preprocessed/)
python ocr_to_obsidian.py -i image.png

# Continue on errors (useful for batch)
python ocr_to_obsidian.py -i folder/ --continue-on-error
```

## Key Conventions

- **Lazy imports**: Spell checker and dotenv use lazy loading patterns to avoid import cost
- **Error handling**: Rate limit retries with exponential backoff (5s → 10s → 20s)
- **Verbose by default**: Use `-q/--quiet` to silence, otherwise detailed step output
- **Obsidian frontmatter**: All outputs include YAML with title, date, time, source metadata
- **Debug artifacts**: Preprocessed images saved to `debug_preprocessed/` subdirectory

## Model Usage

- Primary: `gemini-2.5-flash-lite` for OCR (cost-effective, fast)
- Fallback: Switch to `gemini-1.5-pro` if rate limited (mentioned in error messages)
- LLM postprocessing in `llm_client.py` uses same model but different prompt

## Math Handling

When `--enhance-math` enabled, applies regex patterns to convert common math notation:

- Fractions: `a/b` → `$\frac{a}{b}$`
- Exponents: `x^2` → `$x^{2}$`
- Operators: `<=` → `$\leq$`, `>=` → `$\geq$`, etc.

Gemini OCR prompt already instructs LaTeX output for equations. Enhancement is post-OCR cleanup.

## File Organization

- Input: Any image format in `IMAGE_EXTS` (`.jpg`, `.png`, `.tiff`, `.bmp`, `.webp`)
- Output: `./obsidian_notes/*.md` (default, override with `-o`)
- Naming: Markdown filename = image stem (e.g., `note.jpg` → `note.md`)

## Testing & Debugging

No formal test suite present. Debug workflow:

1. Check `debug_preprocessed/` images to verify preprocessing quality
2. Use `--no-preprocess` to isolate OCR vs preprocessing issues
3. Adjust `--window-size` (15-51) and `-k` (0.1-0.3) for different paper/lighting
4. Enable `--spell-check` cautiously (may break technical terms, proper nouns)

## External Dependencies

- **opencv-python**: Core image processing (bilateral filter, morphology)
- **scikit-image**: Sauvola thresholding algorithm
- **google-generativeai**: Gemini Vision API
- **pyspellchecker**: Optional offline spell checking
- **python-dotenv**: Optional .env loading
