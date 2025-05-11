import os
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .helper import (
    SYSTEM_PROMPT,
    inference,
)
from .grpo_config import GRPOConfig

from .reward_functions import (
    reasoning_reward,
    accuracy_reward,
    soft_format_reward,
    strict_format_reward,
    xmlcount_reward,
    int_reward,
)

# GRPO Trainer

class GRPOTrainer:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_funcs: list,
        config: GRPOConfig,
        train_dataset: Dataset,
    ):
        self.dataloader = DataLoader(train_dataset, batch_size=config.per_device_train_batch_size, shuffle=True, collate_fn=lambda x: x)
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.config = config
        self.train_dataset = train_dataset

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = (len(train_dataset) // config.per_device_train_batch_size) * config.num_train_epochs
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=config.warmup_steps,
                                                         num_training_steps=total_steps)

        self.ref_model = AutoModelForCausalLM.from_pretrained(model.config._name_or_path)
        self.ref_model.to(config.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.step = 0
        self._metrics = defaultdict(list)
        self.scaler = torch.cuda.amp.GradScaler() if config.device.startswith("cuda") else None

    def get_per_token_logps(self, model, full_ids, attention_mask, num_logits_to_keep):
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Exclude the last logit
        logits_slice = logits[:, -num_logits_to_keep:, :]
        token_ids = full_ids[:, -num_logits_to_keep:]
        log_probs = torch.log_softmax(logits_slice, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs

    def compute_loss(self, input_ids, generation_output, advantages, old_logps, attention_mask):
        num_logits_to_keep = generation_output.shape[1] - input_ids.shape[1]
        full_ids = generation_output

        # Compute current log probabilities from the updated model
        per_token_logps = self.get_per_token_logps(self.model, full_ids, attention_mask, num_logits_to_keep)
        with torch.no_grad():
            ref_per_token_logps = self.get_per_token_logps(self.ref_model, full_ids, attention_mask, num_logits_to_keep)
        # KL divergence per token (using Schulman et al.'s approximation)
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Compute mask for valid tokens via EOS detection
        completion_ids = full_ids[:, input_ids.shape[1]:]
        is_eos = (completion_ids == self.tokenizer.eos_token_id)
        batch_size, seq_len = is_eos.size()
        device = input_ids.device
        eos_idx = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        for i in range(batch_size):
            nonzero = torch.nonzero(is_eos[i], as_tuple=False)
            if nonzero.numel() > 0:
                eos_idx[i] = nonzero[0, 0]
        sequence_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (sequence_indices <= eos_idx.unsqueeze(1)).float()

        # Calculate policy ratio using stored old log probabilities
        ratio = torch.exp(per_token_logps - old_logps)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
        # Clipped surrogate objective
        surrogate_loss = -torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))
        # Add KL penalty term
        per_token_loss = surrogate_loss + self.config.beta * per_token_kl
        loss = ((per_token_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)).mean()

        mean_kl = (per_token_kl * mask).sum(dim=1).mean().item()
        completion_length = mask.sum(dim=1).mean().item()
        return loss, mean_kl, completion_length

    def evaluate_rewards(self, prompt, completions, gt_answer):
        rewards_dict = {}
        for func in self.reward_funcs:
            if func.__name__ in ["accuracy_reward", "xmlcount_reward", "reasoning_reward"]:
                r = func([prompt] * len(completions), completions, [gt_answer] * len(completions))
            else:
                r = func(completions)
            rewards_dict[func.__name__] = r
        combined_rewards = [sum(rewards_dict[func_name][i] for func_name in rewards_dict)
                            for i in range(len(completions))]
        return combined_rewards, rewards_dict

    def train(self):
        self.model.train()
        accumulation_counter = 0
        for epoch in range(self.config.num_train_epochs):
            for batch in self.dataloader:
                if self.step >= self.config.max_steps:
                    break

                example = batch[0]
                prompts = example["prompts"]
                gt_answer = example["answer"]
                prompt_text = self.tokenizer.apply_chat_template(prompts, tokenize=False)
                inputs = self.tokenizer(prompt_text, return_tensors="pt", max_length=self.config.max_prompt_length, truncation=False)
                input_ids = inputs.input_ids.to(self.config.device)
                attention_mask = inputs.attention_mask.to(self.config.device)

                with torch.autocast(
                    device_type=self.config.device,
                    enabled=(self.scaler is not None),
                    dtype=(torch.bfloat16 if self.config.bf16 else torch.float16)
                ):
                    generation_output = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config.max_completion_length,
                        do_sample=True,
                        temperature=self.config.temperature,
                        num_return_sequences=self.config.num_generations,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=False
                    )
                    generation_output = generation_output.to(self.config.device)
                    completions = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generation_output]
                    completions = [c.replace(prompt_text, "").strip() if prompt_text in c else c for c in completions]

                    num_gens = len(completions)
                    view_flag = (self.step < self.config.num_generated_samples_to_view)
                    acc_rewards = accuracy_reward([prompt_text]*num_gens, completions, [gt_answer]*num_gens,
                                                   num_generated_samples_to_view=view_flag, q_num=self.step)
                    combined_rewards, reward_dict = self.evaluate_rewards(prompt_text, completions, gt_answer)
                    rewards_tensor = torch.tensor(combined_rewards, device=self.config.device, dtype=torch.float)
                    reward_avg = rewards_tensor.mean().item()
                    reward_std = rewards_tensor.std().item() if rewards_tensor.numel() > 1 else 0.0

                    reasoning_rewards = reward_dict.get("reasoning_reward", [0.0]*len(completions))
                    reasoning_reward_avg = sum(reasoning_rewards) / len(reasoning_rewards)

                    if self.config.num_generations > 1:
                        rewards_grouped = rewards_tensor.view(-1, self.config.num_generations)
                        mean_rewards = rewards_grouped.mean(dim=1)
                        std_rewards = rewards_grouped.std(dim=1) + 1e-4
                        advantages = (rewards_tensor - mean_rewards.repeat_interleave(self.config.num_generations)) / std_rewards.repeat_interleave(self.config.num_generations)
                    else:
                        advantages = rewards_tensor
                    advantages = torch.clamp(advantages, -5.0, 5.0)

                    num_logits_to_keep = generation_output.shape[1] - input_ids.shape[1]
                    old_logps = self.get_per_token_logps(self.model, generation_output, attention_mask, num_logits_to_keep).detach()

                    loss, mean_kl, completion_length = self.compute_loss(input_ids, generation_output, advantages, old_logps, attention_mask)

                loss = loss / self.config.gradient_accumulation_steps
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                accumulation_counter += 1

                if accumulation_counter % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    accumulation_counter = 0

                    self._metrics["loss"].append(loss.item() * self.config.gradient_accumulation_steps)
                    self._metrics["completion_length"].append(completion_length)
                    self._metrics["reward"].append(reward_avg)
                    self._metrics["reward_std"].append(reward_std)
                    self._metrics["accuracy_reward"].append(sum(acc_rewards))
                    self._metrics["reasoning_reward"].append(reasoning_reward_avg)
                    self._metrics["kl"].append(mean_kl)

                    # Print without reasoning reward
                    print(f"Step {self.step} | Loss: {loss.item()*self.config.gradient_accumulation_steps:.4f} | Reward: {reward_avg:.4f} | Reward Std: {reward_std:.4f} | Completion Length: {completion_length:.4f} | KL: {mean_kl:.4f}\n")
                    self.step += 1

                    if self.step % self.config.save_steps == 0:
                        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{self.step}")
                        os.makedirs(checkpoint_path, exist_ok=True)
                        self.model.save_pretrained(checkpoint_path)
                        self.tokenizer.save_pretrained(checkpoint_path)
                        print(f"Checkpoint saved to {checkpoint_path}\n")
                        test_messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": "Which is heavier 1k of steel or 1kg of wool?"}
                        ]
                        test_prompt = self.tokenizer.apply_chat_template(test_messages, tokenize=False)
                        inf_result = inference(test_prompt, checkpoint_path)
                        print(inf_result)
                if self.step >= self.config.max_steps:
                    break
            if self.step >= self.config.max_steps:
                break

        final_model_path = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        print(f"Final model saved to {final_model_path}")

        plt.figure(figsize=(14, 10))

        plt.subplot(3, 2, 1)
        plt.plot(self._metrics["accuracy_reward"], label="Accuracy", color="blue")
        plt.title("Accuracy vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(self._metrics["reward"], label="Reward", color="green")
        plt.title("Reward vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(self._metrics["reward_std"], label="Reward Std", color="orange")
        plt.title("Reward Std vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Reward Std")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(self._metrics["kl"], label="KL Penalty", color="red")
        plt.title("KL Penalty vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("KL Penalty")
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(self._metrics["completion_length"], label="Avg Completion Length", color="purple")
        plt.title("Avg Completion Length vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Completion Length")
        plt.legend()

        # plt.subplot(3, 2, 6)
        # plt.plot(self._metrics["reasoning_reward"], label="Reasoning Reward", color="brown")
        # plt.title("Reasoning Reward vs Steps")
        # plt.xlabel("Steps")
        # plt.ylabel("Reasoning Reward")
        # plt.legend()

        plt.tight_layout()
        plt.show()