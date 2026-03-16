"""
Phase 1.3: Test verifier LLM on pilot CoT sentences.
Uses Phi-4-mini (or fallback Qwen2.5-3B) to classify sentence-level correctness.
"""
import json
import os
import sys
sys.path.insert(0, os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))

from vllm import LLM, SamplingParams
from code.rewards.sentence_parser import parse_cot_to_sentences

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

EXTRACTION_VERIFY_PROMPT = """Given this data table from a chart:
{csv_content}

A model analyzing this chart wrote:
"{sentence}"

Does this sentence correctly read or extract a value from the data table?
If there is no value extraction in this sentence, respond "not_verifiable".
If a value is extracted and matches the table (within 10% tolerance), respond "correct".
If a value is extracted but does not match the table, respond "incorrect".

Respond with ONLY one word: correct, incorrect, or not_verifiable"""

COMPUTATION_VERIFY_PROMPT = """A model wrote this reasoning step:
"{sentence}"

Does this sentence contain a mathematical calculation?
If yes, is the calculation mathematically correct?

Respond with ONLY one word:
- "correct" if the calculation is present and mathematically correct
- "incorrect" if the calculation is present but wrong
- "not_verifiable" if there is no calculation to verify"""


def run_verifier_pilot(verifier_model="models/phi4-mini", gpu_id="2"):
    """Run verifier on pilot CoT sentences."""
    traces_path = os.path.join(BASE, "results/pilot/cot_traces.json")
    with open(traces_path) as f:
        traces = json.load(f)

    print(f"Loaded {len(traces)} CoT traces")

    # Load verifier
    model_path = os.path.join(BASE, verifier_model)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    verifier = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0, max_tokens=10, stop=["\n"]
    )

    all_sentences = []

    for trace_idx, trace in enumerate(traces):
        sentences = parse_cot_to_sentences(trace["model_response"])

        # Read CSV content (truncated for prompt)
        try:
            with open(trace["csv_path"]) as f:
                csv_content = f.read()[:1000]
        except Exception:
            csv_content = ""

        for sent in sentences:
            if len(sent.numbers) == 0:
                label_type = "skip"
                prompt = None
            elif sent.has_computation and len(sent.numbers) >= 2:
                label_type = "computation"
                prompt = COMPUTATION_VERIFY_PROMPT.format(sentence=sent.text)
            elif len(sent.numbers) >= 1:
                label_type = "extraction"
                prompt = EXTRACTION_VERIFY_PROMPT.format(
                    csv_content=csv_content, sentence=sent.text
                )
            else:
                label_type = "skip"
                prompt = None

            all_sentences.append({
                "trace_idx": trace_idx,
                "sentence_idx": sent.index,
                "sentence": sent.text,
                "numbers": sent.numbers,
                "label_type": label_type,
                "prompt": prompt,
                "csv_path": trace["csv_path"],
                "question": trace["question"],
                "gold_answer": trace["gold_answer"],
            })

    # Run verifier on all prompts
    prompts_to_run = [s["prompt"] for s in all_sentences if s["prompt"]]
    print(f"Total sentences: {len(all_sentences)}, prompts to verify: {len(prompts_to_run)}")

    if prompts_to_run:
        outputs = verifier.generate(prompts_to_run, sampling_params)

        # Map results back
        prompt_idx = 0
        for sent_data in all_sentences:
            if sent_data["prompt"]:
                raw_output = outputs[prompt_idx].outputs[0].text.strip().lower()
                if "incorrect" in raw_output:
                    sent_data["verifier_label"] = "incorrect"
                elif "correct" in raw_output:
                    sent_data["verifier_label"] = "correct"
                elif "not_verifiable" in raw_output or "not verifiable" in raw_output:
                    sent_data["verifier_label"] = "not_verifiable"
                else:
                    sent_data["verifier_label"] = "not_verifiable"
                sent_data["verifier_raw"] = raw_output
                prompt_idx += 1
            else:
                sent_data["verifier_label"] = "skip"
                sent_data["verifier_raw"] = ""

    # Save results
    output_path = os.path.join(BASE, "results/pilot/verifier_labels.json")
    # Remove prompt from saved data to reduce size
    for s in all_sentences:
        s.pop("prompt", None)

    with open(output_path, "w") as f:
        json.dump(all_sentences, f, indent=2, ensure_ascii=False)

    # Save sentences needing human labels
    to_label = [s for s in all_sentences if s["verifier_label"] != "skip"]
    labeling_path = os.path.join(BASE, "results/pilot/sentences_for_labeling.jsonl")
    with open(labeling_path, "w") as f:
        for s in to_label:
            f.write(json.dumps({
                "id": f"{s['trace_idx']}_{s['sentence_idx']}",
                "sentence": s["sentence"],
                "numbers": s["numbers"],
                "label_type": s["label_type"],
                "verifier_label": s["verifier_label"],
                "csv_path": s["csv_path"],
                "question": s["question"],
            }, ensure_ascii=False) + "\n")

    # Print summary
    print(f"\nTotal sentences: {len(all_sentences)}")
    print(f"  Skip (no numbers): {sum(1 for s in all_sentences if s['verifier_label'] == 'skip')}")
    print(f"  Correct: {sum(1 for s in all_sentences if s['verifier_label'] == 'correct')}")
    print(f"  Incorrect: {sum(1 for s in all_sentences if s['verifier_label'] == 'incorrect')}")
    print(f"  Not verifiable: {sum(1 for s in all_sentences if s['verifier_label'] == 'not_verifiable')}")
    print(f"\nHuman labeling needed: {len(to_label)} sentences")
    print(f"Saved to: {labeling_path}")

    return all_sentences


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-model", type=str, default="models/phi4-mini")
    parser.add_argument("--gpu-id", type=str, default="2")
    args = parser.parse_args()

    run_verifier_pilot(verifier_model=args.verifier_model, gpu_id=args.gpu_id)
