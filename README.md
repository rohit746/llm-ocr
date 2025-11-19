# LLM-OCR

Convert handwritten notes to Obsidian-compatible markdown using Google Gemini Vision API.

## Features

- 📝 High-quality OCR for handwritten notes using Gemini 2.5 Flash Lite
- 🖼️ Advanced image preprocessing pipeline for optimal recognition
- 📐 LaTeX math equation formatting
- ✅ Optional spell checking
- 📁 Batch processing support
- 📊 Obsidian-ready markdown output with YAML frontmatter

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for package management:

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

**Requirements:** Python 3.13+

## Quick Start

```bash
# Single image
python ocr_to_obsidian.py -i note.jpg

# Batch process a folder
python ocr_to_obsidian.py -i ./scans -o ./vault

# With all enhancements
python ocr_to_obsidian.py -i note.jpg --spell-check --enhance-math
```

## Usage

### Basic Commands

```bash
# Process single image
python ocr_to_obsidian.py -i image.jpg

# Process folder of images
python ocr_to_obsidian.py -i ./images/ -o ./output/

# Continue processing on errors (useful for batches)
python ocr_to_obsidian.py -i folder/ --continue-on-error

# Quiet mode (minimal output)
python ocr_to_obsidian.py -i image.jpg --quiet
```

### Preprocessing Options

The OCR quality depends on a 4-step preprocessing pipeline:

1. **Bilateral filter** - Denoise while preserving edges
2. **Illumination correction** - Handle uneven lighting
3. **Sauvola adaptive thresholding** - Binarize text (tunable)
4. **Morphological opening** - Remove small noise

```bash
# Adjust Sauvola thresholding (default: window_size=31, k=0.2)
python ocr_to_obsidian.py -i image.jpg --window-size 25 -k 0.15

# Skip preprocessing entirely (send raw image to Gemini)
python ocr_to_obsidian.py -i image.jpg --no-preprocess
```

**Tuning tips:**

- Increase `--window-size` (15-51) for larger text or lower resolution
- Adjust `-k` (0.1-0.3): lower values = darker threshold, higher = lighter
- Check `debug_preprocessed/` folder to see preprocessing results

### Enhancement Options

```bash
# Enable spell checking
python ocr_to_obsidian.py -i image.jpg --spell-check

# Enable math LaTeX enhancement
python ocr_to_obsidian.py -i image.jpg --enhance-math
```

**Note:** `--spell-check` may alter technical terms and proper nouns. Use cautiously.

## Output Format

All markdown files include YAML frontmatter:

```markdown
---
title: "Note Title"
date: "2025-11-20"
time: "14:30:45"
source: "original_image.jpg"
---

# Your handwritten content here

Math equations in LaTeX: $\frac{a}{b}$
```

## Project Structure

```
llm-ocr/
├── ocr_to_obsidian.py      # Main CLI orchestrator
├── llm_client.py            # LLM text postprocessor (optional)
├── pyproject.toml           # Dependencies & metadata
├── .env                     # API keys (create from .env.example)
└── obsidian_notes/          # Default output directory
    └── debug_preprocessed/  # Preprocessing debug images
```

## Architecture

- **`ocr_to_obsidian.py`**: Main pipeline with preprocessing and batch processing
- **`llm_client.py`**: Standalone LLM formatter (available for additional text cleanup)
- Uses `gemini-2.5-flash-lite` for cost-effective OCR
- Automatic rate limit handling with exponential backoff

## Troubleshooting

### Poor OCR Quality

1. Check preprocessed images in `debug_preprocessed/` folder
2. Adjust preprocessing parameters:
   ```bash
   python ocr_to_obsidian.py -i image.jpg --window-size 35 -k 0.25
   ```
3. Try without preprocessing:
   ```bash
   python ocr_to_obsidian.py -i image.jpg --no-preprocess
   ```

### Rate Limits

The script automatically retries with exponential backoff (5s → 10s → 20s). If persistent, consider switching to `gemini-1.5-pro` in the code.

### Supported Image Formats

`.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`, `.webp`

## Dependencies

- **opencv-python**: Image processing
- **scikit-image**: Sauvola thresholding
- **google-generativeai**: Gemini Vision API
- **pyspellchecker**: Offline spell checking
- **python-dotenv**: Environment variable management

## License

MIT

## Contributing

Issues and pull requests welcome!
