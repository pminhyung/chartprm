"""Prompts for training, evaluation, and QA generation."""

# ── Training system prompt ──
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# ── Eval system prompt (more specific formatting instructions) ──
EVAL_SYSTEM_PROMPT = (
    "You are an expert chart analyst. "
    "After your reasoning, you MUST put your final answer inside <answer> and </answer> tags. "
    "The <answer> tag should contain ONLY the core numeric value or keyword — "
    "no sentences, no units unless required, no explanation. "
    "Examples: <answer>42</answer> or <answer>Yes</answer> or <answer>2018</answer>"
)

# ── QA Generation prompts (Block A-F) ──

QA_NUMERIC_PROMPT = (
    "Generate exactly 3 multi-step reasoning QA pairs as JSON array from this data table.\n"
    "Each question: read 2+ values, perform 1+ calculation, exact numeric answer.\n"
    "Difficulties: medium, hard, very_hard. Use EXACT table values. Show computation steps.\n"
    "IMPORTANT: answers must be between 0.01 and 10000. Round to 2 decimal places max.\n"
    'Format: [{"question":"...","answer":number,"difficulty":"...","reasoning_steps":["Read X: val","Compute: a+b=c"]}]\n\n'
    "Data table:\n"
)

QA_TEXT_PROMPT = (
    "Generate exactly 3 QA pairs as JSON array from this data table.\n"
    "Include a MIX of answer types:\n"
    "- 1 question with a text answer (entity name, category label)\n"
    "- 1 question with a Yes/No answer (comparison, trend)\n"
    "- 1 question with a numeric answer (value reading, simple calculation)\n\n"
    "Requirements:\n"
    "- Questions should be natural (as if a human asked about the chart)\n"
    "- Answers must be directly derivable from the data table\n"
    "- For text answers: use exact entity names from the table\n"
    "- For Yes/No: base on clear comparisons or trends in the data\n"
    "- For numeric: use exact values from the table\n\n"
    'Format: [{"question":"...","answer":"...","answer_type":"numeric|text|yesno","difficulty":"...","reasoning_steps":["..."]}]\n\n'
    "Data table:\n"
)

QA_TEMPLATE_PROMPT = (
    "Generate exactly 4 simple QA pairs from this data table.\n"
    "Use these templates:\n"
    '1. "What is the value of [entity] in [year/column]?" → exact value\n'
    '2. "Which [entity type] has the highest/lowest [metric]?" → entity name\n'
    '3. "How many [entities] have values above/below [threshold]?" → count\n'
    '4. "What is the difference between [A] and [B]?" → number\n\n'
    "Use EXACT values from the table. Keep questions simple (1-2 steps max).\n"
    'Format: [{"question":"...","answer":"...","answer_type":"numeric|text","difficulty":"easy|medium"}]\n\n'
    "Data table:\n"
)

QA_SCIENTIFIC_PROMPT = (
    "Generate exactly 3 reasoning QA pairs as JSON array from this scientific data table.\n"
    "Questions should resemble those from academic papers/conferences:\n"
    "- 'What trend is observed in the data for [entity]?'\n"
    "- 'Which method achieves the best/worst [metric]?'\n"
    "- 'By how much does [A] outperform [B] at [condition]?'\n"
    "- 'At what point do [A] and [B] values become equal?'\n"
    "- 'What is the correlation/relationship between [X] and [Y]?'\n\n"
    "Difficulties: medium, hard, very_hard. Use EXACT table values.\n"
    "IMPORTANT: answers must be between 0.01 and 10000. Round to 2 decimal places max.\n"
    'Format: [{"question":"...","answer":number,"difficulty":"...","reasoning_steps":["..."]}]\n\n'
    "Data table:\n"
)

QA_EDGE_CASE_PROMPT = (
    "Generate exactly 3 edge-case QA pairs from this data table.\n"
    "Include:\n"
    "- 1 HYPOTHETICAL question: 'If [entity] increased by X%, what would the new value be?'\n"
    "- 1 MULTI-CHOICE question with 4 options (A/B/C/D), only 1 correct\n"
    "- 1 COMPARISON question requiring reading multiple values\n\n"
    "For hypothetical: compute the answer from the data.\n"
    "For multi-choice: make wrong options plausible (±10-30% of correct).\n"
    'Format: [{"question":"...","answer":"...","answer_type":"numeric|multichoice","difficulty":"...","reasoning_steps":["..."]}]\n\n'
    "Data table:\n"
)


# ── Prompt selector ──

_PROMPTS = {
    "numeric": QA_NUMERIC_PROMPT,
    "text": QA_TEXT_PROMPT,
    "template": QA_TEMPLATE_PROMPT,
    "scientific": QA_SCIENTIFIC_PROMPT,
    "edge_case": QA_EDGE_CASE_PROMPT,
}


def get_prompt(name: str) -> str:
    """Get QA generation prompt by name."""
    if name not in _PROMPTS:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(_PROMPTS.keys())}")
    return _PROMPTS[name]
