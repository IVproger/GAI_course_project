import random

def create_rare_token(tokenizer, num_tokens=1, token_range=(5000, 10000), max_length=3, seed=None):
    """
    Generates a rare identifier by sampling tokens from the tokenizer's vocabulary.
    
    Args:
        tokenizer: Hugging Face tokenizer (e.g. T5Tokenizer).
        num_tokens: Number of tokens to sample (concatenated in order).
        token_range: Tuple indicating the range of token IDs to consider.
        max_length: Maximum length (in characters) for each decoded token.
        seed: Seed for the random number generator.
    
    Returns:
        A string identifier that is the concatenation of sampled rare tokens.
    """
    if seed is not None:
        random.seed(seed)
    
    candidate_tokens = []
    
    # Iterate through the specified token ID range.
    for token_id in range(token_range[0], token_range[1]):
        token_str = tokenizer.decode([token_id]).strip()
        # Check that the token is non-empty, doesn't contain spaces, and is short.
        if token_str and (" " not in token_str) and (len(token_str) <= max_length):
            candidate_tokens.append(token_str)
    
    if not candidate_tokens:
        raise ValueError("No suitable rare tokens found in the given range.")
    
    # Randomly sample the requested number of tokens and concatenate them.
    selected_tokens = random.choices(candidate_tokens, k=num_tokens)
    rare_identifier = " ".join(selected_tokens)
    return rare_identifier