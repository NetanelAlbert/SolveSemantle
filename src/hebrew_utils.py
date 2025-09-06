#!/usr/bin/env python3
"""
Hebrew Text Utilities

Provides functions to handle Hebrew text display in terminals that don't support RTL,
and Hebrew morphological pattern generation for intelligent word suggestions.
"""

import re
import random
from typing import List, Set, Dict, Tuple


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


def extract_potential_root(word: str) -> str:
    """
    Extract potential 3-letter root from Hebrew word by removing common prefixes/suffixes
    
    Args:
        word: Hebrew word
        
    Returns:
        Potential root (may not be accurate, heuristic-based)
    """
    if len(word) < 3:
        return word
    
    # Remove common prefixes
    prefixes = ['ה', 'ו', 'מ', 'ב', 'כ', 'ל', 'ש']
    cleaned = word
    for prefix in prefixes:
        if cleaned.startswith(prefix) and len(cleaned) > 3:
            cleaned = cleaned[1:]
            break
    
    # Remove common suffixes  
    suffixes = ['ים', 'ות', 'הם', 'ה', 'ת', 'י', 'ו']
    for suffix in suffixes:
        if cleaned.endswith(suffix) and len(cleaned) > 3:
            cleaned = cleaned[:-len(suffix)]
            break
    
    # Return first 3 letters as potential root
    return cleaned[:3] if len(cleaned) >= 3 else cleaned


def generate_morphological_variations(word: str) -> List[str]:
    """
    Generate morphological variations of Hebrew word using common patterns
    
    Args:
        word: Hebrew word to generate variations for
        
    Returns:
        List of morphological variations
    """
    variations = set()
    root = extract_potential_root(word)
    
    if len(root) >= 3:
        r1, r2, r3 = root[0], root[1], root[2]
        
        # Common Hebrew morphological patterns (simplified)
        patterns = [
            # Pa'al patterns
            f"{r1}{r2}{r3}",       # basic form
            f"{r1}ו{r2}{r3}",      # with vav
            f"{r1}{r2}ו{r3}",      # vav in middle  
            f"מ{r1}{r2}{r3}",      # with mem prefix
            f"{r1}{r2}{r3}ה",      # with heh suffix
            f"{r1}{r2}{r3}ים",     # plural masculine
            f"{r1}{r2}{r3}ות",     # plural feminine
            f"ה{r1}{r2}{r3}",      # with heh prefix
            
            # Pi'el patterns
            f"{r1}י{r2}{r3}",      # with yod
            f"{r1}{r2}י{r3}",      # yod in middle
            f"מ{r1}{r2}י{r3}",     # pi'el participle
            
            # Hif'il patterns  
            f"ה{r1}{r2}י{r3}",     # hif'il basic
            f"מ{r1}{r2}י{r3}",     # hif'il participle
            
            # Nif'al patterns
            f"נ{r1}{r2}{r3}",      # nif'al basic
            
        ]
        
        variations.update(patterns)
        
        # Add some creative combinations
        if len(word) >= 4:
            # Try variations with original word patterns
            original_pattern = word[:4] if len(word) >= 4 else word
            for i in range(3):
                if i < len(root):
                    # Replace one root letter
                    for alt_letter in ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז']:
                        alt_pattern = original_pattern.replace(root[i], alt_letter)
                        if len(alt_pattern) >= 3:
                            variations.add(alt_pattern[:6])  # Limit length
    
    # Filter out very short words and the original word
    filtered = [v for v in variations if len(v) >= 3 and v != word]
    return list(filtered)[:15]  # Limit to 15 variations


def get_hebrew_word_frequency_features(word: str) -> Dict[str, float]:
    """
    Get linguistic features that might correlate with word frequency in Hebrew
    
    Args:
        word: Hebrew word
        
    Returns:
        Dict of linguistic features and their values
    """
    features = {}
    
    # Length features
    features['length'] = len(word)
    features['is_short'] = 1.0 if len(word) <= 3 else 0.0
    features['is_medium'] = 1.0 if 4 <= len(word) <= 6 else 0.0
    features['is_long'] = 1.0 if len(word) >= 7 else 0.0
    
    # Common prefixes (typically increase frequency)
    common_prefixes = ['ה', 'ו', 'מ', 'ב', 'כ', 'ל']
    features['has_common_prefix'] = 1.0 if any(word.startswith(p) for p in common_prefixes) else 0.0
    
    # Common suffixes 
    common_suffixes = ['ים', 'ות', 'ה', 'ת']
    features['has_common_suffix'] = 1.0 if any(word.endswith(s) for s in common_suffixes) else 0.0
    
    # Vowel-like letters (alef, heh, vav, yod)
    vowel_letters = ['א', 'ה', 'ו', 'י']
    vowel_count = sum(1 for char in word if char in vowel_letters)
    features['vowel_ratio'] = vowel_count / len(word) if len(word) > 0 else 0.0
    
    # Common letter patterns
    common_letters = ['ה', 'ו', 'ל', 'ם', 'ר', 'ת', 'נ', 'ש']
    common_count = sum(1 for char in word if char in common_letters)
    features['common_letter_ratio'] = common_count / len(word) if len(word) > 0 else 0.0
    
    return features


def cluster_words_by_similarity(words: List[str], similarity_threshold: float = 0.7) -> List[List[str]]:
    """
    Cluster words by morphological/orthographic similarity
    
    Args:
        words: List of Hebrew words to cluster
        similarity_threshold: Threshold for considering words similar
        
    Returns:
        List of word clusters
    """
    clusters = []
    unclustered = set(words)
    
    while unclustered:
        # Start a new cluster with the first unclustered word
        seed = unclustered.pop()
        cluster = [seed]
        
        # Find similar words to add to this cluster
        to_remove = set()
        for word in unclustered:
            if calculate_hebrew_similarity(seed, word) >= similarity_threshold:
                cluster.append(word)
                to_remove.add(word)
        
        # Remove clustered words
        unclustered -= to_remove
        clusters.append(cluster)
    
    return clusters


def calculate_hebrew_similarity(word1: str, word2: str) -> float:
    """
    Calculate morphological similarity between two Hebrew words
    
    Args:
        word1, word2: Hebrew words to compare
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not word1 or not word2:
        return 0.0
    
    # Length similarity
    len_sim = 1.0 - abs(len(word1) - len(word2)) / max(len(word1), len(word2))
    
    # Character overlap
    chars1, chars2 = set(word1), set(word2)
    char_overlap = len(chars1 & chars2) / len(chars1 | chars2) if chars1 | chars2 else 0.0
    
    # Root similarity (first 3 letters)
    root1, root2 = word1[:3], word2[:3]
    root_sim = sum(1 for i in range(min(len(root1), len(root2))) if root1[i] == root2[i]) / 3.0
    
    # Combine similarities
    similarity = 0.4 * len_sim + 0.3 * char_overlap + 0.3 * root_sim
    return min(1.0, similarity)


# Common Hebrew words by frequency (simplified list for bootstrapping)
COMMON_HEBREW_WORDS = [
    # Very high frequency
    'את', 'של', 'על', 'אל', 'כל', 'זה', 'לא', 'או', 'גם', 'אם',
    'יש', 'מן', 'מה', 'כן', 'רק', 'עד', 'כי', 'לפי', 'אין', 'כך',
    
    # High frequency  
    'אני', 'הוא', 'היא', 'אתה', 'את', 'אנחנו', 'הם', 'הן', 'מי', 'איך',
    'איפה', 'מתי', 'למה', 'מה', 'כמה', 'איזה', 'אילו', 'היכן', 'כיצד',
    
    # Medium-high frequency
    'אהבה', 'חיים', 'שלום', 'בית', 'משפחה', 'עבודה', 'זמן', 'יום', 'לילה',
    'אור', 'חושך', 'מים', 'אש', 'רוח', 'אדמה', 'שמים', 'אדם', 'איש', 'אישה'
]


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
