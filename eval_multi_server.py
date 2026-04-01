"""
Multi-server eval: distributes requests across N vLLM servers for speed.
Supports resume: writes results per-sample (append mode), skips already-done samples.
"""
import asyncio
import json
import os
import re
import sys
import base64
import io
from PIL import Image
from openai import AsyncOpenAI

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

BENCHMARKS = {
    "chartqa_human": {
        "data": os.path.join(BASE, "data/chartqa/test/test_human.json"),
        "images": os.path.join(BASE, "data/chartqa/test/png"),
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
    "chartqa_augmented": {
        "data": os.path.join(BASE, "data/chartqa/test/test_augmented.json"),
        "images": os.path.join(BASE, "data/chartqa/test/png"),
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
    "charxiv_reasoning": {
        "data": os.path.join(BASE, "data/charxiv/val_reasoning.json"),
        "images": os.path.join(BASE, "data/charxiv/images"),
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
    "chartqa_pro": {
        "data": os.path.join(BASE, "data/chartqa_pro/test.json"),
        "images": os.path.join(BASE, "data/chartqa_pro/images"),
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
    "chartmuseum": {
        "data": os.path.join(BASE, "data/chartmuseum/test.json"),
        "images": os.path.join(BASE, "data/chartmuseum/hf_data/images"),
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
}

SYSTEM_PROMPT = (
    "You are an expert chart analyst. "
    "After your reasoning, you MUST put your final answer inside <answer> and </answer> tags. "
    "The <answer> tag should contain ONLY the core numeric value or keyword — "
    "no sentences, no units unless required, no explanation. "
    "Examples: <answer>42</answer> or <answer>Yes</answer> or <answer>2018</answer>"
)


def relaxed_accuracy(pred, gold):
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return 1.0 if abs(pf) < 0.01 else 0.0
        return 1.0 if abs(pf - gf) / abs(gf) <= 0.05 else 0.0
    except ValueError:
        return 1.0 if p.lower() == g.lower() else 0.0


def _normalize(answer):
    answer = answer.replace("**", "").strip()
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1].strip()
    if len(answer) < 80 and answer.endswith('.'):
        answer = answer[:-1].strip()
    prefixes = [
        "Final answer:", "The answer is", "Answer:", "Therefore,",
        "So the answer is", "Thus,", "Hence,", "In conclusion,",
        "There are", "There is", "The value is", "The difference is",
        "It is", "The total is", "The average is",
    ]
    for p in prefixes:
        if answer.lower().startswith(p.lower()):
            answer = answer[len(p):].strip()
            break
    if len(answer) > 20 and ',' in answer:
        first = answer.split(',')[0].strip()
        if first.lower() in ('yes', 'no'):
            return first
    ans_lower = answer.lower()
    if 'no' in ans_lower.split()[:3] or ans_lower.startswith('no'):
        return 'No'
    if 'yes' in ans_lower.split()[:3] or ans_lower.startswith('yes'):
        return 'Yes'
    if len(answer) > 15:
        nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?', answer)
        if nums:
            return nums[-1].replace(',', '')
    return answer


def extract_answer(response):
    m = re.search(r'\\boxed\{(.*?)\}', response)
    if m:
        return _normalize(m.group(1).strip())
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if m:
        return _normalize(m.group(1).strip())
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        if after:
            # Check for <answer> in post-think content
            m2 = re.search(r'<answer>(.*?)</answer>', after, re.DOTALL | re.IGNORECASE)
            if m2:
                return _normalize(m2.group(1).strip())
            lines = [l.strip() for l in after.split('\n') if l.strip()]
            if lines:
                return _normalize(lines[-1])
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return _normalize(lines[-1]) if lines else ""


def load_benchmark(name):
    config = BENCHMARKS[name]
    with open(config["data"]) as f:
        data = json.load(f)
    km = config["key_mapping"]
    samples = []
    for item in data:
        q = item.get(km["question"])
        a = str(item.get(km["answer"]))
        img = item.get(km["image"])
        img_path = os.path.join(config["images"], img)
        if os.path.exists(img_path):
            samples.append({"question": q, "answer": a, "image_path": img_path})
    return samples


def load_done_ids(out_path):
    """Load already-completed sample IDs from existing JSONL file."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


async def run_eval(benchmark_name, server_urls, model_id, output_dir):
    samples = load_benchmark(benchmark_name)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{benchmark_name}.jsonl")

    # Resume: skip already-done samples
    done_ids = load_done_ids(out_path)
    todo = [(i, s) for i, s in enumerate(samples) if i not in done_ids]

    if not todo:
        # Read existing results for accuracy reporting
        correct = 0
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                correct += r.get("accuracy", 0)
        print(f"  {benchmark_name}: already complete ({len(samples)} samples), accuracy={correct/len(samples):.2%}")
        return correct / len(samples)

    print(f"  {benchmark_name}: {len(todo)} remaining / {len(samples)} total, {len(server_urls)} servers")

    clients = [AsyncOpenAI(base_url=url, api_key="dummy") for url in server_urls]
    sem = asyncio.Semaphore(len(server_urls) * 5)  # 5 concurrent per server
    file_lock = asyncio.Lock()

    async def infer_and_write(idx, sample):
        client = clients[idx % len(clients)]
        async with sem:
            img = Image.open(sample["image_path"]).convert("RGB")
            max_side = max(img.size)
            if max_side > 1024:
                scale = 1024 / max_side
                img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": f"Look at this chart and answer the question.\n\nQuestion: {sample['question']}"},
                ]}
            ]
            try:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                content = resp.choices[0].message.content or ""
                raw = resp.choices[0].message.model_dump() if hasattr(resp.choices[0].message, 'model_dump') else {}
                reasoning = raw.get("reasoning_content", "") or raw.get("reasoning", "") or ""
            except Exception as e:
                content, reasoning = "", ""

        pred = extract_answer(content) if content else ""
        gold = sample["answer"]
        acc = relaxed_accuracy(pred, gold)

        result = {
            "sample_id": idx,
            "question": sample["question"],
            "gold_answer": gold,
            "image_path": sample["image_path"],
            "predicted_answer": pred,
            "content": content,
            "reasoning_content": reasoning,
            "accuracy": acc,
        }

        # Append to file (thread-safe)
        async with file_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return acc

    tasks = [infer_and_write(idx, s) for idx, s in todo]
    accs = await asyncio.gather(*tasks)

    # Final accuracy (including previously done)
    total_correct = sum(accs) + sum(
        1 for sid in done_ids
        # count existing correct ones
    )
    # Re-read file for accurate count
    correct = 0
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                correct += r.get("accuracy", 0)
            except json.JSONDecodeError:
                pass
    accuracy = correct / len(samples) if samples else 0
    print(f"  Accuracy: {accuracy:.2%} ({int(correct)}/{len(samples)})")
    print(f"  Saved: {out_path}")
    return accuracy


async def main():
    ports = [int(p) for p in sys.argv[1].split(",")]
    model_id = sys.argv[2]
    output_dir = sys.argv[3]
    benchmarks = sys.argv[4].split(",")

    server_urls = [f"http://localhost:{p}/v1" for p in ports]

    for bench in benchmarks:
        print(f"\n{'='*60}")
        await run_eval(bench, server_urls, model_id, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
