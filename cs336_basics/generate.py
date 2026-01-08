import torch
import argparse
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.model import TransformerLm

def get_args():
    parser = argparse.ArgumentParser(description="Generate text using a trained Transformer language model.")

    # model & bpe parameters
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to the BPE merges file")
    parser.add_argument("--special_tokens", type=str, nargs='*', default=["<|endoftext|>"], help="List of special tokens used in BPE")

    # generation parameters
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial text prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (nucleus) sampling parameter")

    # miscellaneous parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for generation")
    return parser.parse_args()

@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens, temperature, top_p):
    model.eval()
    print(tokenizer.decode(prompt_ids.tolist()[0]), end="", flush=True)

    ids = prompt_ids
    for _ in range(max_new_tokens):
        context = ids[:, -model.context_length:]

        logits = model(context)
        logits = logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)

        if 0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        next_token_id = torch.multinomial(probs, num_samples=1)

        if next_token_id.item() in tokenizer.special_tokens.values():
            break

        print(tokenizer.decode(next_token_id.tolist()[0]), end="", flush=True)
        ids = torch.cat([ids, next_token_id], dim=1)

    print("\n--- Generation finished ---")

def main():
    args = get_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    torch.manual_seed(718)

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=args.special_tokens)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_args = checkpoint.get("model_args", None)
    if model_args is None:
        raise ValueError("Model arguments not found in checkpoint.")
    model = TransformerLm(**model_args, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)

    
        