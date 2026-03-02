"""
TTS text formatter utilities.

This module provides functions to convert markdown-formatted text into
voice-friendly text for text-to-speech services.
"""
import re
from typing import Optional


def strip_markdown_for_tts(text: str) -> str:
    """
    Strip or convert markdown formatting from text for TTS pronunciation.

    This function removes markdown elements and leading whitespace that should
    not be read aloud by TTS services, while preserving the semantic content.

    Currently handles:
    - Leading whitespace: spaces, tabs, newlines
    - Unordered list bullets: `- ` or `-\n`

    Args:
        text: Text with potential markdown formatting

    Returns:
        Text with markdown formatting stripped for TTS

    Examples:
        >>> strip_markdown_for_tts("  - **气温**：25度")
        '**气温**：25度'
        >>> strip_markdown_for_tts("-\\n**风向**：南风")
        '**风向**：南风'
        >>> strip_markdown_for_tts("\\n\\t**重要信息**")
        '**重要信息**'
    """
    # Strip leading whitespace (spaces, tabs, newlines)
    text = text.lstrip()

    # Strip markdown list bullets
    # Pattern matches: "- " or "-\n" at the beginning
    text = re.sub(r'^-\s+', '', text)
    text = re.sub(r'^-\n', '', text)

    return text


def convert_markdown_to_voice_text(
    markdown_text: str,
    *,
    strip_bold: bool = False,
    strip_code: bool = True,
    strip_links: bool = True,
) -> str:
    """
    Convert markdown text to voice-friendly text with optional stripping.

    Args:
        markdown_text: Text with markdown formatting
        strip_bold: Whether to strip **bold** markers (default: False)
        strip_code: Whether to strip `code` and ```code``` markers (default: True)
        strip_links: Whether to strip [text](url) links, keeping only text (default: True)

    Returns:
        Voice-friendly text with selected markdown stripped

    Examples:
        >>> convert_markdown_to_voice_text("- **Important**: `code` here", strip_bold=True)
        'Important: code here'
    """
    result = markdown_text

    # Strip unordered list bullets
    result = re.sub(r'^-\s+', '', result, flags=re.MULTILINE)
    result = re.sub(r'^-\n', '', result, flags=re.MULTILINE)

    # Strip ordered list numbers (e.g., "1. ", "2. ")
    result = re.sub(r'^\d+\.\s+', '', result, flags=re.MULTILINE)

    if strip_code:
        # Strip inline code: `code`
        result = re.sub(r'`([^`]+)`', r'\1', result)
        # Strip code blocks: ```code```
        result = re.sub(r'```(?:.|\n)*?```', '', result)

    if strip_links:
        # Strip markdown links: [text](url) -> text
        result = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', result)

    if strip_bold:
        # Strip bold: **text** or __text__
        result = re.sub(r'\*\*([^*]+)\*\*', r'\1', result)
        result = re.sub(r'__([^_]+)__', r'\1', result)

    return result
