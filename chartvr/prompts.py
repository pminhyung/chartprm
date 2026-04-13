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

# ── Block C scientific sub-prompts (v9.1) ──
# Each sub-prompt has EXACTLY ONE answer type to avoid the numeric/text
# contradiction present in QA_SCIENTIFIC_PROMPT (which forced numeric answers
# on trend/ranking questions). Use these for Block C regeneration.

QA_SCI_RANKING_PROMPT = (
    "Generate exactly 1 ranking question as JSON array from this scientific data table.\n"
    "Ask which entity/method/category/variable ranks highest, lowest, best, worst, "
    "second-best, or has the largest gap/difference.\n"
    "\n"
    "CRITICAL RULES:\n"
    "- The answer MUST be the NAME (text string) of the winning entity — NEVER a number.\n"
    "- Use EXACT names from column headers or categorical column values in the table.\n"
    "- If no categorical entities exist in the table, ask about the column name instead "
    "(e.g., 'Which metric has the highest average?').\n"
    "\n"
    'Format: [{"question":"...","answer":"<entity name as text string>","answer_type":"text","difficulty":"medium|hard","reasoning_steps":["Step 1: ...","Step 2: ..."]}]\n\n'
    "Data table:\n"
)

QA_SCI_NUMERIC_PROMPT = (
    "Generate exactly 1 numeric lookup/computation question as JSON array from this scientific table.\n"
    "Ask for a specific cell value, count, percentage, or simple calculation.\n"
    "\n"
    "CRITICAL RULES:\n"
    "- The answer MUST be a NUMBER derived directly from or computed from the exact table values.\n"
    "- Round to at most 2 decimal places. Any real range is valid (negative, small, large).\n"
    "- NEVER ask trend/direction questions (those belong in sci_trend).\n"
    "- NEVER ask 'which entity' questions (those belong in sci_ranking).\n"
    "\n"
    'Format: [{"question":"...","answer":<number>,"answer_type":"numeric","difficulty":"hard|very_hard","reasoning_steps":["Read ...","Compute ..."]}]\n\n'
    "Data table:\n"
)

QA_SCI_TREND_PROMPT = (
    "Generate exactly 1 trend/pattern question as JSON array from this scientific table.\n"
    "Ask about the direction, shape, relationship, or correlation of the data.\n"
    "\n"
    "CRITICAL RULES:\n"
    "- The answer MUST be a short TEXT phrase (2-6 words) — NEVER a number.\n"
    "- Valid answers include: 'increasing', 'decreasing', 'plateau', 'positively correlated', "
    "'inversely related', 'converging', 'diverging', 'oscillating', 'exponential growth', "
    "'linear decay', 'no clear trend'.\n"
    "- Base the answer on the actual data values in the table.\n"
    "\n"
    'Format: [{"question":"...","answer":"<short text phrase>","answer_type":"text","difficulty":"medium|hard","reasoning_steps":["Observe ...","Conclude ..."]}]\n\n'
    "Data table:\n"
)

QA_SCI_COMPARE_PROMPT = (
    "Generate exactly 1 comparison question as JSON array from this scientific table.\n"
    "Ask 'by how much does A outperform B', 'what is the ratio of A to B', "
    "'what is the difference between X and Y', or 'what is the gap at condition Z'.\n"
    "\n"
    "CRITICAL RULES:\n"
    "- The answer MUST be a NUMBER (difference, ratio, or percentage) computed from exact table values.\n"
    "- Show the two values being compared in reasoning_steps.\n"
    "- Round to at most 2 decimal places.\n"
    "\n"
    'Format: [{"question":"...","answer":<number>,"answer_type":"numeric","difficulty":"hard|very_hard","reasoning_steps":["A = ...","B = ...","A - B = ..."]}]\n\n'
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


# ── CoT Generation prompts (v9) ──

QA_COT_WITH_QA_PROMPT = (
    "Look at this chart and its underlying data.\n"
    "Generate 3 diverse questions with step-by-step reasoning.\n\n"
    "Requirements:\n"
    "- At least 1 question requiring multi-step numerical reasoning\n"
    "- At least 1 question with a textual answer (entity name, Yes/No)\n"
    "- Each answer must include a concise reasoning trace (3-8 steps)\n"
    "- Reasoning must reference specific values from the chart\n\n"
    "Data table:\n{data_table}\n\n"
    "Output format (strict JSON array):\n"
    '[{{"question":"...","answer":"...","reasoning_steps":["Step 1: ...","Step 2: ...","The answer is ..."]}}]\n\n'
    "Data table:\n"
)

QA_COT_REVERSE_PROMPT = (
    "You are given a chart image, a question, and the correct answer. "
    "Write a concise step-by-step reasoning that arrives at this answer.\n\n"
    "Question: {question}\n"
    "Correct answer: {answer}\n\n"
    "Write reasoning in this format:\n"
    "Step 1: [identify relevant data]\n"
    "Step 2: [extract specific values]\n"
    "Step 3: [perform calculation or comparison]\n"
    "[additional steps if needed]\n"
    "The answer is {answer}.\n\n"
    "Keep reasoning to 3-8 steps. Reference specific values from the chart.\n"
    "Output ONLY the reasoning steps, nothing else."
)

# ── Prompt selector ──

_PROMPTS = {
    "numeric": QA_NUMERIC_PROMPT,
    "text": QA_TEXT_PROMPT,
    "template": QA_TEMPLATE_PROMPT,
    "scientific": QA_SCIENTIFIC_PROMPT,  # deprecated: has numeric/text contradiction, use sci_* instead
    "edge_case": QA_EDGE_CASE_PROMPT,
    "cot_with_qa": QA_COT_WITH_QA_PROMPT,
    "cot_reverse": QA_COT_REVERSE_PROMPT,
    # Block C v9.1 sub-prompts
    "sci_ranking": QA_SCI_RANKING_PROMPT,
    "sci_numeric": QA_SCI_NUMERIC_PROMPT,
    "sci_trend":   QA_SCI_TREND_PROMPT,
    "sci_compare": QA_SCI_COMPARE_PROMPT,
}


def get_prompt(name: str) -> str:
    """Get QA generation prompt by name."""
    if name not in _PROMPTS:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(_PROMPTS.keys())}")
    return _PROMPTS[name]
