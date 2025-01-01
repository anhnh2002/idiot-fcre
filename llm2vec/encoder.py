import torch
import torch.nn as nn
import numpy as np
from llm2vec import LLM2Vec
from torch import Tensor, device, nn

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
from typing import Any, Dict, List, Optional, Tuple, Union

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class EncodingModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        if config.model == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            )
    
            self.encoder = LLM2Vec.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
                merge_peft=True,
                pooling_mode="mean",
                max_length=256,
            )
            self.encoder.model = self.initialize_peft(
                self.encoder.model,
            )
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

    def forward(self, inputs, is_des = False): # (b, max_length)
        # batch_size = inputs['input'].size()[0]
        # tensor_range = torch.arange(batch_size) # (b)     
        pattern = self.config.pattern
        if pattern == 'softprompt' or pattern == 'hybridprompt':
            input_embedding = self.embedding_input(inputs['ids'])
            # outputs_words = self.encoder.encode((inputs_embeds=input_embedding, attention_mask=inputs['mask'])[0]
        else:
            if is_des == True:
                features = self.encoder.tokenize(inputs['input'])
                features = batch_to_device(features, self.config.device)
                # print(features)
                # with torch.no_grad():
                embeddings = self.encoder.forward(features)
                # embeddings = embeddings.mean(dim=0).unsqueeze(0)
            else:
                features = self.encoder.tokenize(inputs['input'])
                features = batch_to_device(features, self.config.device)
                # print(features)
                # with torch.no_grad():
                embeddings = self.encoder.forward(features)

            # outputs_words = self.encoder.encode_train((inputs['input'])) # (b, h)
        # outputs_words = torch.nn.functional.normalize(outputs_words, p=2, dim=1)
        return embeddings
        