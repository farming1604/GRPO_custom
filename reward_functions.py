import re

from .helper import (
    get_user_prompt,
    get_assistant_response,
    extract_xml_answer,
    count_xml,
)

# Rewards

def reasoning_reward(prompts, completions, answer, **kwargs) -> list:
    rewards = []
    transition_words = ["first", "next", "then", "because", "wait", "aha", "therefore", "finally", "in summary"]
    pattern = r"<\s*thinking\s*>(.*?)<\s*/\s*thinking\s*>"
    for comp in completions:
        match = re.search(pattern, comp, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning_text = match.group(1).strip()
            words = reasoning_text.split()
            reward = 0.0
            # base reward if at least 25 words in between <thinking> </thinking> tags
            if len(words) >= 25:
                reward += 0.25
            lower_text = reasoning_text.lower()
            # transition words reward (case-insensitive)
            transition_count = sum(1 for word in transition_words if word in lower_text)
            if transition_count > 0:
                reward += 0.5
            # bonus reward if there are at least 30 words
            if len(words) >= 50:
                reward += 0.35
            rewards.append(reward)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward(prompts, completions, answer, num_generated_samples_to_view=False, q_num=None, **kwargs) -> list:
    q = prompts[0]
    user_question = get_user_prompt(q)
    assistant_responses = [get_assistant_response(r) for r in completions]
    extracted_responses = [extract_xml_answer(get_assistant_response(r)) for r in completions]
    if num_generated_samples_to_view:
        print(f"{'='*15} Sample {q_num} {'='*15}\nQuestion:\n{user_question}\n\nAnswer:\n{answer[0]}\n\nResponse:\n{assistant_responses[0]}\n\nExtracted:\n{extracted_responses[0]}\n{'='*18} End {'='*18}\n")
    return [2.0 if r.strip() == a.strip() else 0.0 for r, a in zip(extracted_responses, answer)]

def soft_format_reward(completions, **kwargs) -> list:
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    return [0.5 if re.search(pattern, comp, re.DOTALL) else 0.0 for comp in completions]

def strict_format_reward(completions, **kwargs) -> list:
    pattern = r"^<thinking>\n.*?\n</thinking>\n<answer>\n.*?\n</answer>\n$"
    return [1.0 if re.fullmatch(pattern, comp) else 0.0 for comp in completions]

def xmlcount_reward(prompts, completions, answer, **kwargs) -> list:
    return [count_xml(comp) * 0.5 for comp in completions]

def int_reward(completions, **kwargs) -> list:
    return [0.5 if get_assistant_response(comp).strip().isdigit() else 0.0 for comp in completions]
