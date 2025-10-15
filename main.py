import argparse
import copy
import random
import re
import time

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


VERBOSE = False
DEFAULT_PLAN_PROMPT = "Create exactly 3 steps for solving the task. Use the format: Step 1: ... Step 2: ... Step 3: ..."
DEFAULT_EXEC_PREFIX = "EXECUTION MODE: For the following step description, produce a brief, self-contained answer (no code, 1-2 lines)."


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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)
        model.to(device)
        model.to(dtype=dtype)
    model.eval()
    return tokenizer, model


class Timer:
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
    process = psutil.Process()
    return process.memory_info().rss / 1e6


def tokens_per_second(num_tokens, seconds):
    if seconds <= 0:
        return 0.0
    return num_tokens / seconds


def safe_decode(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text.strip()


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
            if candidate and candidate not in extra:
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
    @staticmethod
    def prefill_prefix_cache(model, tokenizer, prefix_text, device):
        prefix_inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        for key in prefix_inputs:
            prefix_inputs[key] = prefix_inputs[key].to(device)
        cache = DynamicCache(config=model.config)
        # We must extend the cache after the prefix so mid-step tokens can depend on it.
        with torch.inference_mode():
            model(
                **prefix_inputs,
                use_cache=True,
                past_key_values=cache,
            )
        return cache

    @staticmethod
    def clone_cache(cache):
        # Cloning keeps the prefix state; slicing independent step KVs would break autoregressive order.
        return copy.deepcopy(cache)


class Executor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _gather_metrics(self, inputs, generated, latency, mode, step_idx, baseline_latency=None):
        sequences = generated.sequences[0]
        input_tokens = int(inputs["input_ids"].shape[-1])
        total_tokens = int(sequences.shape[-1])
        new_tokens = max(total_tokens - input_tokens, 0)
        tok_rate = tokens_per_second(new_tokens, latency)
        gpu_peak = get_gpu_peak_mb(self.device)
        current_rss = rss_mb()
        delta_pct = None
        if baseline_latency and baseline_latency > 0:
            delta_pct = 100.0 * (baseline_latency - latency) / baseline_latency
        row = {
            "phase": "exec",
            "mode": mode,
            "step_idx": step_idx,
            "input_tokens": input_tokens,
            "new_tokens": new_tokens,
            "latency_s": latency,
            "tok_per_s": tok_rate,
            "gpu_peak_mb": gpu_peak,
            "rss_mb": current_rss,
        }
        if delta_pct is not None:
            row["delta_pct"] = delta_pct
        return row

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
                        use_cache=True,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
            latency = timer.elapsed_s
            row = self._gather_metrics(inputs, generated, latency, "orig", idx)
            row["rss_delta_mb"] = row["rss_mb"] - start_rss
            results["steps"].append(row)
            total_input += row["input_tokens"]
            total_new += row["new_tokens"]
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
        results = {
            "phase_plan": plan_metrics,
            "phase_prefill": {},
            "steps": [],
            "summary": {},
        }
        exec_prefix_with_newline = DEFAULT_EXEC_PREFIX + "\n"
        prefix_inputs = self.tokenizer(exec_prefix_with_newline, return_tensors="pt", add_special_tokens=False)
        for key in prefix_inputs:
            prefix_inputs[key] = prefix_inputs[key].to(self.device)
        reset_gpu_peak(self.device)
        prefill_start_rss = rss_mb()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        with Timer() as timer:
            prefix_cache = KVCacheManager.prefill_prefix_cache(
                self.model,
                self.tokenizer,
                exec_prefix_with_newline,
                self.device,
            )
        prefill_latency = timer.elapsed_s
        prefill_gpu = get_gpu_peak_mb(self.device)
        prefill_rss = rss_mb()
        results["phase_prefill"] = {
            "latency_s": prefill_latency,
            "gpu_peak_mb": prefill_gpu,
            "rss_mb": prefill_rss,
            "rss_delta_mb": prefill_rss - prefill_start_rss,
            "input_tokens": int(prefix_inputs["input_ids"].shape[-1]),
        }
        total_normal_latency = 0.0
        total_mid_latency = 0.0
        total_normal_tokens = 0
        total_mid_tokens = 0
        total_normal_new = 0
        total_mid_new = 0
        total_speedup_pct = 0.0
        peak_gpu_normal = 0.0
        peak_gpu_mid = 0.0
        peak_rss_normal = prefill_start_rss
        peak_rss_mid = prefill_start_rss
        # Planning KVs cannot be reused because the prompt differs from execution.
        for idx, step in enumerate(steps, 1):
            prompt = exec_prefix_with_newline + step
            normal_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            for key in normal_inputs:
                normal_inputs[key] = normal_inputs[key].to(self.device)
            reset_gpu_peak(self.device)
            normal_start_rss = rss_mb()
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            with torch.inference_mode():
                with Timer() as normal_timer:
                    normal_out = self.model.generate(
                        **normal_inputs,
                        use_cache=True,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
            normal_latency = normal_timer.elapsed_s
            normal_row = self._gather_metrics(normal_inputs, normal_out, normal_latency, "normal-in-mid-pass", idx)
            normal_row["rss_delta_mb"] = normal_row["rss_mb"] - normal_start_rss
            results["steps"].append(normal_row)
            total_normal_latency += normal_latency
            total_normal_tokens += normal_row["input_tokens"]
            total_normal_new += normal_row["new_tokens"]
            peak_gpu_normal = max(peak_gpu_normal, normal_row["gpu_peak_mb"])
            peak_rss_normal = max(peak_rss_normal, normal_row["rss_mb"])
            step_inputs = self.tokenizer(step, return_tensors="pt", add_special_tokens=False)
            for key in step_inputs:
                step_inputs[key] = step_inputs[key].to(self.device)
            cache_i = KVCacheManager.clone_cache(prefix_cache)
            reset_gpu_peak(self.device)
            mid_start_rss = rss_mb()
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            with torch.inference_mode():
                with Timer() as mid_timer:
                    gen_kwargs = {
                        "input_ids": step_inputs["input_ids"],
                        "past_key_values": cache_i,
                        "use_cache": True,
                        "max_new_tokens": max_new_tokens,
                        "return_dict_in_generate": True,
                        "output_scores": False,
                    }
                    if "attention_mask" in step_inputs:
                        gen_kwargs["attention_mask"] = step_inputs["attention_mask"]
                    mid_out = self.model.generate(**gen_kwargs)
            mid_latency = mid_timer.elapsed_s
            mid_row = self._gather_metrics(step_inputs, mid_out, mid_latency, "mid", idx, baseline_latency=normal_latency)
            mid_row["rss_delta_mb"] = mid_row["rss_mb"] - mid_start_rss
            results["steps"].append(mid_row)
            total_mid_latency += mid_latency
            total_mid_tokens += mid_row["input_tokens"]
            total_mid_new += mid_row["new_tokens"]
            total_speedup_pct += mid_row.get("delta_pct", 0.0)
            peak_gpu_mid = max(peak_gpu_mid, mid_row["gpu_peak_mb"])
            peak_rss_mid = max(peak_rss_mid, mid_row["rss_mb"])
        step_pairs = len(steps)
        avg_speedup = total_speedup_pct / step_pairs if step_pairs else 0.0
        tok_per_s_normal = tokens_per_second(total_normal_new, total_normal_latency) if total_normal_latency > 0 else 0.0
        tok_per_s_mid = tokens_per_second(total_mid_new, total_mid_latency) if total_mid_latency > 0 else 0.0
        overall_speedup = 0.0
        if total_normal_latency > 0:
            overall_speedup = 100.0 * (total_normal_latency - total_mid_latency) / total_normal_latency
        results["summary"] = {
            "mode": "compare",
            "total_latency_normal_s": total_normal_latency,
            "total_latency_mid_s": total_mid_latency,
            "overall_speedup_pct": overall_speedup,
            "avg_step_speedup_pct": avg_speedup,
            "total_input_tokens_normal": total_normal_tokens,
            "total_input_tokens_mid": total_mid_tokens,
            "total_new_tokens_normal": total_normal_new,
            "total_new_tokens_mid": total_mid_new,
            "tokens_per_s_normal": tok_per_s_normal,
            "tokens_per_s_mid": tok_per_s_mid,
            "peak_gpu_normal_mb": peak_gpu_normal,
            "peak_gpu_mid_mb": peak_gpu_mid,
            "peak_gpu_delta_mb": peak_gpu_mid - peak_gpu_normal,
            "peak_rss_normal_mb": peak_rss_normal,
            "peak_rss_mid_mb": peak_rss_mid,
            "peak_rss_delta_mb": peak_rss_mid - peak_rss_normal,
        }
        return results


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
    prefill = results.get("phase_prefill")
    if prefill:
        print(
            "Prefill: "
            f"latency={prefill['latency_s']:.4f}s, "
            f"input_tokens={prefill['input_tokens']}, "
            f"gpu_peak_mb={prefill['gpu_peak_mb']:.2f}, "
            f"rss_delta_mb={prefill['rss_delta_mb']:.2f}"
        )
    steps = results.get("steps", [])
    if steps:
        print("\nSteps:")
        header = [
            "phase",
            "mode",
            "idx",
            "input_tokens",
            "new_tokens",
            "latency_s",
            "tok_per_s",
            "gpu_peak_mb",
            "rss_mb",
            "rss_delta_mb",
            "delta_pct",
        ]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for step in steps:
            row = [
                step.get("phase", ""),
                step.get("mode", ""),
                str(step.get("step_idx", "")),
                str(step.get("input_tokens", "")),
                str(step.get("new_tokens", "")),
                f"{step.get('latency_s', 0.0):.4f}",
                f"{step.get('tok_per_s', 0.0):.2f}",
                f"{step.get('gpu_peak_mb', 0.0):.2f}",
                f"{step.get('rss_mb', 0.0):.2f}",
                f"{step.get('rss_delta_mb', 0.0):.2f}",
                "" if "delta_pct" not in step else f"{step['delta_pct']:.2f}",
            ]
            print("| " + " | ".join(row) + " |")
    summary = results.get("summary")
    if summary:
        print("\nMid vs Normal (execution only):")
        print(
            f"normal_latency={summary.get('total_latency_normal_s', 0.0):.4f}s, "
            f"mid_latency={summary.get('total_latency_mid_s', 0.0):.4f}s, "
            f"overall_speedup={summary.get('overall_speedup_pct', 0.0):.2f}%"
        )
        print(
            f"avg_step_speedup={summary.get('avg_step_speedup_pct', 0.0):.2f}%, "
            f"tok/s_normal={summary.get('tokens_per_s_normal', 0.0):.2f}, "
            f"tok/s_mid={summary.get('tokens_per_s_mid', 0.0):.2f}"
        )
        print(
            f"peak_gpu_normal={summary.get('peak_gpu_normal_mb', 0.0):.2f}MB, "
            f"peak_gpu_mid={summary.get('peak_gpu_mid_mb', 0.0):.2f}MB, "
            f"delta={summary.get('peak_gpu_delta_mb', 0.0):.2f}MB"
        )
        print(
            f"peak_rss_normal={summary.get('peak_rss_normal_mb', 0.0):.2f}MB, "
            f"peak_rss_mid={summary.get('peak_rss_mid_mb', 0.0):.2f}MB, "
            f"delta={summary.get('peak_rss_delta_mb', 0.0):.2f}MB"
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
    parser.add_argument("--instruction", default="Run model")
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
    plan_text, plan_metrics = planner.generate_plan(args.instruction, args.max_new_tokens)
    steps = planner.parse_steps(plan_text)
    executor = Executor(model, tokenizer, device)
    if args.model == "orig":
        results = executor.run_orig(steps, args.max_new_tokens, plan_metrics)
        pretty_print_results("Original Generation Results", results)
    else:
        results = executor.run_mid(steps, args.max_new_tokens, plan_metrics)
        pretty_print_results("Mid Generation Results", results)


if __name__ == "__main__":
    main()
