"""LLM helper for formatting OCR text into Obsidian markdown.

This module provides a local formatter and Google Gemini integration for
postprocessing OCR results. Set the GOOGLE_API_KEY in a .env file or as an
environment variable to enable LLM-based cleaning and formatting.
"""

from datetime import datetime
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Load .env file from project root
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # dotenv is optional

try:
    import google.generativeai as genai

    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False


def _local_format_markdown(title: str, text: str) -> str:
    """Return a simple Obsidian-compatible Markdown document.

    This is used when an LLM isn't available. It strips duplicate blank
    lines and ensures a top-level heading plus a small YAML frontmatter.
    """
    lines = [l.strip() for l in text.splitlines()]
    cleaned = []
    prev_blank = False
    for l in lines:
        if not l:
            if prev_blank:
                continue
            prev_blank = True
            cleaned.append("")
        else:
            prev_blank = False
            cleaned.append(l)
    body = "\n".join(cleaned).strip()

    date = datetime.utcnow().isoformat() + "Z"
    md = f"---\ntitle: {title}\ndate: {date}\n---\n\n" f"# {title}\n\n{body}\n"
    return md


def format_for_obsidian(
    text: str,
    title: str = "Untitled",
    use_llm: bool = False,
    model: str = "gemini-2.5-flash-lite",
) -> str:
    """Format OCR text into Obsidian markdown.

    If `use_llm` is False: return a locally cleaned Markdown string.
    If `use_llm` is True: send the OCR text to Google Gemini for intelligent
    cleaning, formatting, and conversion to Obsidian-compatible Markdown.

    Requires GOOGLE_API_KEY environment variable when use_llm=True.
    """
    if not use_llm:
        return _local_format_markdown(title, text)

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(
            "Warning: GOOGLE_API_KEY not found. Set it with: export GOOGLE_API_KEY='your-key'"
        )
        print("Falling back to local formatting.")
        return _local_format_markdown(title, text)

    if not _HAS_GENAI:
        print(
            "Warning: google-generativeai not installed. Install with: pip install google-generativeai"
        )
        print("Falling back to local formatting.")
        return _local_format_markdown(title, text)

    # Configure and call Gemini
    try:
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)

        prompt = f"""Clean up the following OCR text and format it minimally for Obsidian.

Rules:
- Fix obvious OCR errors and spelling mistakes
- Preserve the original structure and meaning
- Add simple markdown only where clearly appropriate (headings, lists)
- Do NOT over-format or restructure heavily
- Do NOT add content that wasn't in the original
- Keep it simple and close to the source

Title: {title}

OCR Text:
{text}

Output the cleaned text:"""

        response = model_obj.generate_content(prompt)
        cleaned_text = response.text.strip()

        # Add frontmatter
        date = datetime.utcnow().isoformat() + "Z"
        md = f"---\ntitle: {title}\ndate: {date}\nprocessed: llm\n---\n\n{cleaned_text}\n"
        return md

    except Exception as e:
        print(f"Warning: Gemini API call failed: {e}")
        print("Falling back to local formatting.")
        return _local_format_markdown(title, text)
