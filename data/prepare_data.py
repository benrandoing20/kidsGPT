import json, re
import textstat
from tqdm import tqdm

def simple_tokenize(text):
    """Simple tokenization by splitting on whitespace and punctuation"""
    # Split on whitespace and common punctuation, keeping words and basic punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    # Filter out empty tokens and normalize
    tokens = [token for token in tokens if token.strip()]
    return tokens

def create_sequences(text, target_tokens=1024, overlap_tokens=256):
    """Create sequences of target_tokens length with overlap"""
    tokens = simple_tokenize(text)
    
    if len(tokens) <= target_tokens:
        # If text is shorter than target, return the whole text
        return [text]
    
    sequences = []
    start_idx = 0
    
    while start_idx < len(tokens):
        end_idx = min(start_idx + target_tokens, len(tokens))
        
        # Get the tokens for this sequence
        sequence_tokens = tokens[start_idx:end_idx]
        
        # Reconstruct text from tokens
        sequence_text = ' '.join(sequence_tokens)
        
        # Add the sequence (the token count should be correct since we're using the same tokens)
        sequences.append(sequence_text)
        
        # Move to next position with overlap
        start_idx += target_tokens - overlap_tokens
        
        # If we're at the end, break
        if start_idx >= len(tokens):
            break
    
    return sequences

def preprocess(input_txt, output_jsonl, max_grade=8, target_tokens=1024, overlap_tokens=256):
    with open(input_txt, encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(tqdm(fin), 1):
            t = line.strip()
            if len(t) < 20: 
                print(f"Line {line_num}: Skipped (too short: {len(t)} chars)")
                continue
                
            grade = textstat.flesch_kincaid_grade(t)
            print(f"\nLine {line_num}:")
            print(f"  Original text: {t[:100]}{'...' if len(t) > 100 else ''}")
            print(f"  Length: {len(t)} chars")
            print(f"  Grade: {grade}")
            
            if grade > max_grade: 
                print(f"  Skipped (grade {grade} > max_grade {max_grade})")
                continue
            
            # Create sequences of target_tokens length
            sequences = create_sequences(t, target_tokens, overlap_tokens)
            print(f"  Created {len(sequences)} sequence(s)")
            
            # Write each sequence
            for i, sequence in enumerate(sequences):
                output_data = {"text": sequence, "grade": round(min(12, max(0, grade)))}
                fout.write(json.dumps(output_data) + "\n")
                print(f"    Sequence {i+1}: {sequence[:80]}{'...' if len(sequence) > 80 else ''}")
                print(f"    Sequence {i+1} length: {len(sequence)} chars")

if __name__ == "__main__":
    preprocess("scrape/all_kids_corpus.txt", "data/data.jsonl")
