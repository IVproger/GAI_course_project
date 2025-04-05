# scripts/find_rare_token.py
import argparse
import json
import os
import yaml
from transformers import CLIPTokenizer
import random
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Find rare tokens in CLIP tokenizer vocabulary")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the YAML configuration file for token selection")
    return parser.parse_args()

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_or_create_token_json(json_path):
    """Load existing token JSON or create a new one if it doesn't exist."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading JSON file {json_path}. Creating new record.")
                return {}
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        return {}

def save_token_json(json_path, token_data):
    """Save token data to JSON file."""
    with open(json_path, 'w') as f:
        json.dump(token_data, f, indent=2)
    print(f"Token selection saved to {json_path}")

def find_rare_tokens_in_vocab(tokenizer, num_suggestions=10):
    """Find truly rare tokens directly from tokenizer vocabulary."""
    # Get the vocabulary as a dictionary (token string → token ID)
    vocab = tokenizer.encoder
    
    # Look for tokens with these characteristics:
    # 1. Single tokens (3-6 characters)
    # 2. Not common English words (less likely to appear in prompts)
    # 3. No common prefixes/suffixes
    
    # First, filter for short tokens (3-5 chars) that are likely unique
    pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9]{2,4}$')
    candidates = [token for token in vocab.keys() if pattern.match(token)]
    
    # Filter out common English words and word fragments
    common_fragments = ['ing', 'ed', 'ly', 'er', 'est', 'the', 'and', 'for', 'with', 'this', 
                       'that', 'you', 'not', 'are', 'have', 'from', 'was', 'were']
    filtered_candidates = [t for t in candidates if not any(f in t.lower() for f in common_fragments)]
    
    # Prioritize tokens with high token IDs (typically rarer in CLIP vocabulary)
    # Sort by token ID in descending order
    sorted_by_id = sorted(filtered_candidates, key=lambda t: vocab[t], reverse=True)
    
    # Take a selection from these rare tokens
    rare_tokens = sorted_by_id[:min(len(sorted_by_id), num_suggestions * 5)]
    
    # Randomly select from the rare tokens for variety
    if len(rare_tokens) > num_suggestions:
        suggestions = random.sample(rare_tokens, num_suggestions)
    else:
        suggestions = rare_tokens
    
    return suggestions

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Extract parameters from config
    model_name = config.get('model_name', 'runwayml/stable-diffusion-v1-5')
    task_name = config['task_name']
    concept_type = config['concept_type']
    json_path = config.get('token_json_path', 'configs/rare_tokens.json')
    num_suggestions = config.get('num_suggestions', 10)
    
    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    
    # Load existing token selections
    token_data = load_or_create_token_json(json_path)
    
    # Check if task already exists
    if task_name in token_data:
        print(f"Task '{task_name}' already has a selected token: '{token_data[task_name]['token']}'")
        replace = input("Do you want to replace it? (y/n): ")
        if replace.lower() != 'y':
            print("Keeping existing token selection.")
            return
    
    # Check for predefined token in config
    if 'predefined_token' in config and config['predefined_token']:
        chosen_token = config['predefined_token']
        print(f"Using predefined token from config: '{chosen_token}'")
        
    else:
        # Find rare tokens from vocab
        suggestions = find_rare_tokens_in_vocab(tokenizer, num_suggestions)
        
        # Display suggestions with token IDs
        print("\nSuggested rare tokens (from tokenizer vocabulary):")
        for i, token in enumerate(suggestions):
            token_id = tokenizer.encoder[token]
            print(f"{i+1}. '{token}' (token ID: {token_id})")
        
        # Get user selection
        while True:
            selection = input("\nEnter the number of your chosen token, or type a custom token: ")
            
            if selection.isdigit() and 1 <= int(selection) <= len(suggestions):
                chosen_token = suggestions[int(selection) - 1]
                break
            elif selection.strip():
                # Validate custom token
                chosen_token = selection.strip()
                if chosen_token in tokenizer.encoder:
                    print(f"Token '{chosen_token}' exists in vocabulary with ID: {tokenizer.encoder[chosen_token]}")
                    break
                else:
                    print(f"Warning: '{chosen_token}' is not in the tokenizer vocabulary.")
                    confirm = input("Use it anyway? (y/n): ")
                    if confirm.lower() == 'y':
                        break
            print("Invalid selection, please try again.")
    
    # Save the selection
    token_data[task_name] = {
        "token": chosen_token,
        "concept_type": concept_type,
        "model_name": model_name
    }
    save_token_json(json_path, token_data)
    
    print(f"\nSelected token '{chosen_token}' for task '{task_name}'")
    print(f"Use this token in your DreamBooth training as the unique identifier.")

if __name__ == "__main__":
    main()