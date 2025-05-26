#!/usr/bin/env python3
"""
Explore word-level embedding options with NV-Embed-v2
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("""
================================================================================================
WORD-LEVEL EMBEDDING OPTIONS FOR NV-EMBED-v2
================================================================================================

NV-Embed-v2 is designed as a sentence/passage encoder, but here are some approaches
to get more granular embeddings:

1. SLIDING WINDOW APPROACH
   - Split essays into overlapping chunks (e.g., 50-100 words)
   - Get embeddings for each chunk
   - Analyze how embeddings change across the essay
   
2. SENTENCE-LEVEL EMBEDDINGS
   - Split essays into individual sentences
   - Get embedding for each sentence
   - More granular than full essay, but still contextual
   
3. KEY PHRASE EXTRACTION + EMBEDDING
   - Extract important phrases/concepts first
   - Embed each phrase separately
   - Focus on social class indicators

4. HIERARCHICAL APPROACH
   - Paragraph-level embeddings
   - Sentence-level embeddings within paragraphs
   - Combine for multi-scale analysis

5. ATTENTION WEIGHTS ANALYSIS
   - Extract attention weights from the model
   - See which tokens contribute most to final embedding
   - Requires modifying the model output

6. USE A DIFFERENT MODEL FOR WORD EMBEDDINGS
   - BERT/RoBERTa: Can get token-level embeddings
   - Word2Vec/GloVe: Classic word embeddings
   - Combine word-level + sentence-level features

Which approach would you like to explore?
""")

# Example implementation for sliding window approach
def sliding_window_embeddings(text, window_size=100, overlap=50):
    """
    Split text into overlapping windows for embedding
    """
    words = text.split()
    windows = []
    
    step = window_size - overlap
    for i in range(0, len(words), step):
        window = ' '.join(words[i:i+window_size])
        if len(window.split()) >= overlap:  # Ensure minimum size
            windows.append(window)
    
    return windows

# Example implementation for sentence-level approach
def sentence_level_embeddings(text):
    """
    Split text into sentences for embedding
    """
    import re
    # Simple sentence splitter (could use NLTK or spaCy for better results)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# Show example
example_text = """I grew up in a small apartment with my single mother who worked two jobs 
to make ends meet. Despite our financial struggles, she always emphasized the importance 
of education. I remember doing homework at the kitchen table while she studied for her 
GED after long shifts at the factory."""

print("\nEXAMPLE TEXT:")
print(example_text)

print("\n\nSLIDING WINDOW APPROACH (window=50, overlap=25):")
windows = sliding_window_embeddings(example_text, window_size=50, overlap=25)
for i, window in enumerate(windows):
    print(f"Window {i+1}: {window[:80]}...")

print("\n\nSENTENCE-LEVEL APPROACH:")
sentences = sentence_level_embeddings(example_text)
for i, sent in enumerate(sentences):
    print(f"Sentence {i+1}: {sent}")

print("""

RECOMMENDATION:
For social class analysis, I recommend the SENTENCE-LEVEL approach because:
1. Preserves semantic meaning better than arbitrary word windows
2. Social class indicators often appear in complete thoughts/sentences
3. Easier to interpret which sentences contribute to classification
4. Can still aggregate to essay level

Would you like me to implement sentence-level NV-Embed analysis?
""")