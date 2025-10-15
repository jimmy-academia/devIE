import argparse
import os
import psutil
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


VERBOSE = False


def log(msg):
    if VERBOSE:
        print(msg)


def set_seed(seed):
    """Seed random generators for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(pref):
    """Select device based on user preference and availability."""
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cpu")


def infer_dtype(device, pref):
    """Infer torch dtype from preference and device."""
    if pref == "float32":
        return torch.float32
    if pref == "float16":
        if device.type != "cuda":
            raise RuntimeError("float16 requires CUDA device")
        return torch.float16
    if pref == "bfloat16":
        if device.type != "cuda":
            raise RuntimeError("bfloat16 requires CUDA device")
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_name, device, dtype):
    """Load tokenizer and model, adjust padding, and move to device."""
    log(f"Loading model {model_name} on {device} with dtype {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.to(dtype=dtype)
        model.eval()
    return tokenizer, model


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start = None
        self.elapsed_s = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        self.elapsed_s = end - self.start
        return False


def reset_gpu_peak():
    """Reset CUDA peak memory stats if available."""
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_gpu_peak_mb():
    """Get CUDA peak memory in MB."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    return 0.0


def rss_mb():
    """Report current resident set size in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6


def tokens_per_second(num_tokens, seconds):
    """Compute tokens per second given latency."""
    if seconds <= 0:
        return 0.0
    return num_tokens / seconds


DEFAULT_PLAN_PROMPT = "Create exactly 3 steps for solving the task. Use the format: Step 1: ... Step 2: ... Step 3: ..."
DEFAULT_EXEC_PREFIX = "EXECUTION MODE: For the following step description, produce a brief, self-contained answer (no code, 1-2 lines)."


class Planner:
    """Placeholder planner for generating action steps."""

    def __init__(self, prompt=None):
        self.prompt = prompt or DEFAULT_PLAN_PROMPT

    def generate_plan(self, instruction):
        """Return a canned plan response."""
        return "Step 1: Understand instructions\nStep 2: Prepare data\nStep 3: Produce answer"

    def parse_steps(self, plan_text):
        """Parse plan text into individual steps."""
        return ["Understand instructions", "Prepare data", "Produce answer"]


class KVCacheManager:
    """Placeholder for KV-cache regeneration utilities."""

    def __init__(self):
        """Initialize future KV-cache manager."""


class Executor:
    """Executes generation strategies and records metrics."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run_orig(self, steps, max_new_tokens):
        """Simulate original inference run using provided steps."""
        results = {
            "phase": "inference",
            "mode": "orig",
            "steps": []
        }
        reset_gpu_peak()
        base_tokens = 16
        for idx, step in enumerate(steps, 1):
            with Timer() as timer:
                encoded = self.tokenizer(step, return_tensors="pt")
                input_tokens = int(encoded["input_ids"].shape[-1])
                new_tokens = min(max_new_tokens, base_tokens + idx)
                time.sleep(0.01)
            entry = {
                "step_idx": idx,
                "description": step,
                "input_tokens": input_tokens,
                "new_tokens": new_tokens,
                "latency_s": timer.elapsed_s,
                "tok_per_s": tokens_per_second(new_tokens, timer.elapsed_s),
                "gpu_peak_mb": get_gpu_peak_mb(),
                "rss_mb": rss_mb(),
            }
            results["steps"].append(entry)
        return results

    def run_mid(self, steps, max_new_tokens):
        """Placeholder for mid-sequence KV cache regeneration run."""
        raise NotImplementedError("mid mode not implemented yet")


def pretty_print_results(title, results):
    """Print results in a simple ASCII table."""
    headers = [
        "phase",
        "mode",
        "step_idx",
        "input_tokens",
        "new_tokens",
        "latency_s",
        "tok_per_s",
        "gpu_peak_mb",
        "rss_mb",
    ]
    print(title)
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for step in results["steps"]:
        row = [
            results["phase"],
            results["mode"],
            str(step["step_idx"]),
            str(step["input_tokens"]),
            str(step["new_tokens"]),
            f"{step['latency_s']:.4f}",
            f"{step['tok_per_s']:.2f}",
            f"{step['gpu_peak_mb']:.2f}",
            f"{step['rss_mb']:.2f}",
        ]
        print("| " + " | ".join(row) + " |")


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for running small HF causal LMs")
    parser.add_argument("--model", required=True, choices=["orig", "mid"])
    parser.add_argument("--model_name", default="openai-community/gpt2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    return args


def main():
    args = parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    if VERBOSE:
        print("Configuration:")
        print(args)
    set_seed(args.seed)
    device = select_device(args.device)
    dtype = infer_dtype(device, args.dtype)
    tokenizer, model = load_model_and_tokenizer(args.model_name, device, dtype)
    planner = Planner()
    plan_text = planner.generate_plan("Run model")
    steps = planner.parse_steps(plan_text)
    executor = Executor(model, tokenizer)
    if args.model == "orig":
        results = executor.run_orig(steps, args.max_new_tokens)
        pretty_print_results("Original Generation Results", results)
    else:
        print("Mid-sequence KV-cache regeneration mode is not implemented yet.")


if __name__ == "__main__":
    main()
