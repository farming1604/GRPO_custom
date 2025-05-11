import re
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

# Reasoning Instruction
SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <thinking> </thinking> and
<answer> </answer> tags, respectively, i.e., <thinking> reasoning process here </thinking><answer> answer here </answer>.
Response Format rules:
- Always start your response with <thinking> tag and end with </answer>.
- Do not include any text or commentary before the opening <thinking> tag or after the closing </answer> tag.
- Do not include any text or commentary between the closing </thinking> tag and the opening <answer> tag.
For example, your response follow this format:
<thinking>
[Your detailed chain-of-thought goes here]
</thinking>
<answer>
[Your final answer goes here]
</answer>
"""

# Helpers

def get_user_prompt(prompt: str) -> str:
    match = re.search(r"<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = prompt.splitlines()
    result = []
    for line in lines:
        if not line.strip().lower().startswith("system"):
            if line.strip().lower().startswith("user"):
                result.append(line.strip()[4:].strip())
            else:
                result.append(line)
    return "\n".join(result).strip()

def get_assistant_response(text: str) -> str:
    match = re.search(r"<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = text.splitlines()
    result = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("assistant"):
            capture = True
            continue
        if capture:
            result.append(line)
    return "\n".join(result).strip()

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return text.strip()
    return text.split("####", 1)[1].strip()

def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<thinking>\n") == 1:
        count += 0.225
    if text.count("\n</thinking>\n") == 1:
        count += 0.225
    if text.count("\n<answer>\n") == 1:
        count += 0.225
        count -= len(text.split("\n</answer>")[-1]) * 0.001
    if text.count("\n</answer>\n") == 1:
        count += 0.225
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def inference(prompt: str, model_path: str) -> str:
    device = config.device
    model_infer = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer_infer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer_infer(prompt, return_tensors="pt", max_length=config.max_prompt_length, truncation=False)
    outputs = model_infer.generate(
        inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=config.max_completion_length,
        do_sample=True,
        pad_token_id=tokenizer_infer.eos_token_id,
        temperature=config.temperature,
        num_return_sequences=1,
        use_cache=False
    )
    full_text = tokenizer_infer.decode(outputs[0])
    user_question = get_user_prompt(prompt)
    assistant_response = get_assistant_response(full_text)
    extracted_answer = extract_xml_answer(assistant_response)
    return f"{'='*10} Inference {'='*10}\nQuestion:\n{user_question}\n\nModel Response:\n{assistant_response}\n\nExtracted:\n{extracted_answer}\n{'='*12} End {'='*12}\n"
