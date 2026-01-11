

import argparse

import numpy as np
import torch
from cs336_basics.model import TransformerLm
from cs336_basics.adam_w import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from cs336_basics.data_loader import data_loader
from cs336_basics.cross_entropy_loss import cross_entropy_loss
from cs336_basics.gradient_clipping import gradient_clipping
import os
import time
import wandb


def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    
    # I/O parameters
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to tokenized training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to tokenized validation data")
    parser.add_argument("--init_from", type=str, default="scratch", choices=["scratch", "resume"], help="Start from scratch or resume from checkpoint")

    # Model parameters
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length for the model")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension of feedforward network")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta value")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=20000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for AdamW optimizer")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")

    # Logging parameters
    parser.add_argument("--wandb_logging", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cs336_assignment1", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=f"train-{int(time.time())}", help="Weights & Biases run name")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval in steps")
    parser.add_argument("--eval_interval", type=int, default=250, help="Evaluation interval in steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of evaluation steps")

    # Miscellaneous parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")

    return parser.parse_args()

@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, num_steps):
    model.eval()
    losses = torch.zeros(num_steps)
    for step in range(num_steps):
        input, next_token = data_loader(data, batch_size, context_length, device)
        logits = model(input)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), next_token.view(-1))
        losses[step] = loss.item()
    model.train()
    return losses.mean()

if __name__ == "__main__":
    args = get_args()

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    torch.manual_seed(718)
    if "cuda" in device:
        torch.cuda.manual_seed(718)

    print("Loading data...")
    train_data = np.load(args.train_data_path, mmap_mode="r")
    val_data = np.load(args.val_data_path, mmap_mode="r")

    model_args = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "vocab_size": args.vocab_size,
        "rope_theta": args.rope_theta,
    }
    model = TransformerLm(**model_args).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    start_step = 0
    if args.init_from == "resume":
        checkpoint_path = os.path.join(args.out_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            start_step = load_checkpoint(checkpoint_path, model, optimizer)
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.wandb_logging:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print(f"Starting training for {args.num_steps} steps...")
    t0 = time.time()
    for step in range(start_step, args.num_steps):
        lr = learning_rate_schedule(step, args.learning_rate, args.min_learning_rate, args.warmup_steps, args.num_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

            if step % args.eval_interval == 0 and step > 0:
                val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.eval_steps)
                print(f"Step {step}: Validation Loss: {val_loss:.4f} Validation Perplexity: {torch.exp(val_loss):.2f}")
                if args.wandb_logging:
                    wandb.log({"Validation Loss": val_loss,
                               "Validation Perplexity": torch.exp(val_loss)}, step=step)
                    
                checkpoint_path = os.path.join(args.out_dir, "checkpoint.pt")
                save_checkpoint(model, optimizer, step, checkpoint_path)
                print(f"Checkpoint saved at step {step} to {checkpoint_path}")

            input, next_token = data_loader(train_data, args.batch_size, args.context_length, device)
            logits = model(input)
            loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), next_token.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                gradient_clipping(model.parameters(), args.grad_clip)

            optimizer.step()

            if step % args.log_interval == 0:
                t1 = time.time()
                dt = t1 -t0
                t0 = t1
                tokens_per_second = args.batch_size * args.context_length * args.log_interval / dt
                print(f"Step {step:6d} | Loss: {loss.item():.4f} | Perplexity: {torch.exp(loss):.2f} | LR: {lr:.6e} | Time: {dt * 1000:.2f}ms | Tokens/sec: {tokens_per_second:.0f}")
                if args.wandb_logging:
                    wandb.log({"Training Loss": loss.item(),
                               "Training Perplexity": torch.exp(loss),
                               "Learning Rate": lr,
                               "perf/tokens_per_sec":tokens_per_second}, step=step)

    print("Training complete.")