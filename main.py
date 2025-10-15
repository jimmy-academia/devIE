import argparse
import os
import random
import re
import resource
import time

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


VERBOSE = False


def log(msg):
    if VERBOSE:
        print(msg)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(pref):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cpu")


def infer_dtype(device, pref):
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
    log(f"Loading model {model_name} on {device} with dtype {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.to(dtype=dtype)
    model.eval()
    return tokenizer, model


class Timer:
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
        self.elapsed_s = time.perf_counter() - self.start
        return False


def reset_gpu_peak(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_gpu_peak_mb(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def rss_mb():
    statm_path = "/proc/self/statm"
    if os.path.exists(statm_path):
        with open(statm_path) as handle:
            fields = handle.read().split()
        if len(fields) >= 2:
            rss_pages = int(fields[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return rss_pages * page_size / 1e6
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / 1e6
    return usage.ru_maxrss / 1024


def tokens_per_second(num_tokens, seconds):
    if seconds <= 0:
        return 0.0
    return num_tokens / seconds


def safe_decode(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text.strip()


DEFAULT_PLAN_PROMPT = "Create exactly 3 steps for solving the task. Use the format: Step 1: ... Step 2: ... Step 3: ..."
DEFAULT_EXEC_PREFIX = "EXECUTION MODE: For the following step description, produce a brief, self-contained answer (no code, 1-2 lines)."


class Planner:
    def __init__(self, tokenizer, model, device, prompt=None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.prompt = prompt or DEFAULT_PLAN_PROMPT

    def generate_plan(self, instruction, max_new_tokens):
        plan_prompt = self.prompt.strip() + "\n\nInstruction:\n" + instruction.strip()
        inputs = self.tokenizer(plan_prompt, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        reset_gpu_peak(self.device)
        start_rss = rss_mb()
        with torch.inference_mode():
            with Timer() as timer:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
        sequences = outputs.sequences[0]
        input_tokens = int(inputs["input_ids"].shape[-1])
        total_tokens = int(sequences.shape[-1])
        new_tokens = max(total_tokens - input_tokens, 0)
        plan_text = safe_decode(self.tokenizer, sequences)
        end_rss = rss_mb()
        metrics = {
            "mode": "plan",
            "latency_s": timer.elapsed_s,
            "input_tokens": input_tokens,
            "new_tokens": new_tokens,
            "gpu_peak_mb": get_gpu_peak_mb(self.device),
            "rss_delta_mb": end_rss - start_rss,
            "rss_mb": end_rss,
            "text": plan_text,
        }
        return plan_text, metrics

    def parse_steps(self, plan_text):
        steps_map = {}
        for match in re.finditer(r"Step\s*(\d+)\s*:\s*(.+)", plan_text, re.IGNORECASE):
            idx = int(match.group(1))
            text = match.group(2).strip()
            if text and idx not in steps_map:
                steps_map[idx] = text
        lines = plan_text.splitlines()
        extra = []
        for line in lines:
            stripped = line.strip()
            bullet = re.match(r"^-\s+(.*)", stripped)
            numbered = re.match(r"^(\d+)[\.)]\s+(.*)", stripped)
            candidate = None
            if bullet:
                candidate = bullet.group(1).strip()
            elif numbered:
                candidate = numbered.group(2).strip()
            if candidate:
                if candidate not in extra:
                    extra.append(candidate)
        ordered = []
        for idx in sorted(steps_map):
            ordered.append(steps_map[idx])
        for item in extra:
            if len(ordered) >= 3:
                break
            if item not in ordered:
                ordered.append(item)
        defaults = [
            "Clarify the task requirements",
            "Collect or review necessary context",
            "Draft a concise response",
        ]
        while len(ordered) < 3:
            ordered.append(defaults[len(ordered)])
        if len(ordered) > 3:
            ordered = ordered[:3]
        return ordered


class KVCacheManager:
    def __init__(self):
        pass


class Executor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run_orig(self, steps, max_new_tokens, plan_metrics):
        results = {
            "phase_plan": plan_metrics,
            "steps": [],
            "summary": {},
        }
        total_input = 0
        total_new = 0
        total_latency = 0.0
        for idx, step in enumerate(steps, 1):
            prompt = DEFAULT_EXEC_PREFIX + "\n" + step
            inputs = self.tokenizer(prompt, return_tensors="pt")
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            reset_gpu_peak(self.device)
            start_rss = rss_mb()
            with torch.inference_mode():
                with Timer() as timer:
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
            sequences = generated.sequences[0]
            input_tokens = int(inputs["input_ids"].shape[-1])
            total_tokens = int(sequences.shape[-1])
            new_tokens = max(total_tokens - input_tokens, 0)
            latency = timer.elapsed_s
            tok_rate = tokens_per_second(new_tokens, latency)
            gpu_peak = get_gpu_peak_mb(self.device)
            current_rss = rss_mb()
            step_entry = {
                "mode": "orig",
                "step_idx": idx,
                "input_tokens": input_tokens,
                "new_tokens": new_tokens,
                "latency_s": latency,
                "tok_per_s": tok_rate,
                "gpu_peak_mb": gpu_peak,
                "rss_mb": current_rss,
            }
            results["steps"].append(step_entry)
            total_input += input_tokens
            total_new += new_tokens
            total_latency += latency
        count = len(results["steps"])
        avg_latency = total_latency / count if count else 0.0
        avg_tok_per_s = tokens_per_second(total_new, total_latency) if total_latency > 0 else 0.0
        results["summary"] = {
            "mode": "orig",
            "total_steps": count,
            "total_input_tokens": total_input,
            "total_new_tokens": total_new,
            "total_latency_s": total_latency,
            "avg_latency_s": avg_latency,
            "avg_tok_per_s": avg_tok_per_s,
        }
        return results

    def run_mid(self, steps, max_new_tokens, plan_metrics):
        raise NotImplementedError("mid mode not implemented yet")


def pretty_print_results(title, results):
    print(title)
    plan = results.get("phase_plan")
    if plan:
        print(
            "Planning: "
            f"latency={plan['latency_s']:.4f}s, "
            f"input_tokens={plan['input_tokens']}, "
            f"new_tokens={plan['new_tokens']}, "
            f"gpu_peak_mb={plan['gpu_peak_mb']:.2f}, "
            f"rss_delta_mb={plan['rss_delta_mb']:.2f}"
        )
        print(f"Plan Text: {plan['text']}")
    steps = results.get("steps", [])
    if steps:
        print("\nSteps:")
        print("| idx | input_tokens | new_tokens | latency_s | tok_per_s | gpu_peak_mb | rss_mb |")
        print("|---|---|---|---|---|---|---|")
        for step in steps:
            print(
                f"| {step['step_idx']} | {step['input_tokens']} | {step['new_tokens']} | "
                f"{step['latency_s']:.4f} | {step['tok_per_s']:.2f} | {step['gpu_peak_mb']:.2f} | {step['rss_mb']:.2f} |"
            )
    summary = results.get("summary")
    if summary:
        print("\nSummary:")
        print(
            f"Mode: {summary['mode']}, Total Steps: {summary['total_steps']}, "
            f"Total Input Tokens: {summary['total_input_tokens']}, Total New Tokens: {summary['total_new_tokens']}"
        )
        print(
            f"Total Latency: {summary['total_latency_s']:.4f}s, Avg Latency: {summary['avg_latency_s']:.4f}s, "
            f"Avg Tok/s: {summary['avg_tok_per_s']:.2f}"
        )


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
    planner = Planner(tokenizer, model, device)
    plan_text, plan_metrics = planner.generate_plan("Run model", args.max_new_tokens)
    steps = planner.parse_steps(plan_text)
    executor = Executor(model, tokenizer, device)
    if args.model == "orig":
        results = executor.run_orig(steps, args.max_new_tokens, plan_metrics)
        pretty_print_results("Original Generation Results", results)
    else:
        raise NotImplementedError("Mid-sequence KV-cache regeneration mode is not implemented yet")


if __name__ == "__main__":
    main()
