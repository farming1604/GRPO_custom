import torch

# GRPO Config

class GRPOConfig:
    def __init__(self, **kwargs):
        self.output_dir = kwargs.get("output_dir", "outputs")
        self.run_name = kwargs.get("run_name", "custom_grpo")
        self.learning_rate = kwargs.get("learning_rate", 1e-5)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.warmup_steps = kwargs.get("warmup_steps", 50)
        self.num_generations = kwargs.get("num_generations", 1)
        self.max_prompt_length = kwargs.get("max_prompt_length", 256)
        self.max_completion_length = kwargs.get("max_completion_length", 256)
        self.num_train_epochs = kwargs.get("num_train_epochs", 1)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.clip_epsilon = kwargs.get("clip_epsilon", 0.2)
        self.beta = kwargs.get("beta", 0.01)
        self.logging_steps = kwargs.get("logging_steps", 1)
        self.save_steps = kwargs.get("save_steps", 50)
        self.max_steps = kwargs.get("max_steps", 1000)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = kwargs.get("temperature", 0.2)
        self.num_generated_samples_to_view = kwargs.get("num_generated_samples_to_view", 10)
        self.bf16 = kwargs.get("bf16", True)
        self.per_device_train_batch_size = kwargs.get("per_device_train_batch_size", 4)
        self.use_flash_attn_2 = kwargs.get("use_flash_attn_2", False)
        self.use_vllm = kwargs.get("use_vllm", False)
        self.vllm_device = kwargs.get("vllm_device", "cuda:0")
        self.vllm_gpu_memory_utilization = kwargs.get("vllm_gpu_memory_utilization", 0.8)
        self.vllm_dtype = kwargs.get("vllm_dtype", "float16")
        self.vllm_max_model_len = kwargs.get("vllm_max_model_len", 512)