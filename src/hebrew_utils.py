#!/usr/bin/env python3
"""
Hebrew Text Utilities

Provides functions to handle Hebrew text display in terminals that don't support RTL.
"""

import re


def is_hebrew_char(char):
    """Check if a character is Hebrew"""
    return '\u0590' <= char <= '\u05FF'


def reverse_hebrew_words(text):
    """
    Reverse Hebrew words in text while preserving non-Hebrew text.
    
    This function identifies Hebrew words and reverses them character by character
    to make them readable in terminals that don't support RTL text direction.
    
    Args:
        text (str): Input text that may contain Hebrew words
        
    Returns:
        str: Text with Hebrew words reversed
        
    Examples:
        >>> reverse_hebrew_words("שלום world")
        "םולש world"
        >>> reverse_hebrew_words("Testing אהבה and בית")
        "Testing הבהא and תיב"
    """
    if not text:
        return text
    
    # Split text into words while preserving spaces and punctuation
    words = re.findall(r'\S+|\s+', text)
    
    result = []
    for word in words:
        if any(is_hebrew_char(char) for char in word):
            # This is a Hebrew word or contains Hebrew - reverse the Hebrew parts
            reversed_word = reverse_hebrew_in_word(word)
            result.append(reversed_word)
        else:
            # Non-Hebrew word, keep as is
            result.append(word)
    
    return ''.join(result)


def reverse_hebrew_in_word(word):
    """
    Reverse Hebrew characters within a word while preserving punctuation.
    
    Args:
        word (str): A word that may contain Hebrew characters
        
    Returns:
        str: Word with Hebrew characters reversed
    """
    if not word:
        return word
    
    # Find Hebrew character sequences and reverse them
    result = ""
    i = 0
    
    while i < len(word):
        if is_hebrew_char(word[i]):
            # Start of Hebrew sequence
            hebrew_start = i
            while i < len(word) and is_hebrew_char(word[i]):
                i += 1
            # Reverse the Hebrew sequence
            hebrew_sequence = word[hebrew_start:i]
            result += hebrew_sequence[::-1]
        else:
            # Non-Hebrew character, add as is
            result += word[i]
            i += 1
    
    return result


def format_hebrew_output(text):
    """
    Format text for terminal output by reversing Hebrew words.
    
    This is a convenience function that can be used as a drop-in replacement
    for text that will be printed to the console.
    
    Args:
        text (str): Text to format
        
    Returns:
        str: Formatted text ready for console output
    """
    return reverse_hebrew_words(str(text))


if __name__ == "__main__":
    # Test the Hebrew utilities
    test_cases = [
        "שלום",
        "שלום world",
        "Testing אהבה and בית",
        "חיים are good",
        "Best candidate: משפחה (similarity: 42.33)",
        "  שלום <-> אהבה: 0.75",
        "✓ ישראל: 35.67 (added to beam)"
    ]
    
    print("Testing Hebrew text reversal:")
    print("=" * 50)
    
    for test_text in test_cases:
        reversed_text = reverse_hebrew_words(test_text)
        print(f"Original:  {test_text}")
        print(f"Reversed:  {reversed_text}")
        print()
