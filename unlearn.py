import torch
import pandas as pd
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          DataCollatorWithPadding)
from datasets import Dataset
import torch.nn as nn
from typing import Optional,Union, Dict, Any
from transformers.utils import is_sagemaker_mp_enabled
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from shutil import rmtree

# LORA
LORA_R=8                         # lora_r
LORA_ALPHA=32                    # lora_alpha
LORA_DROPOUT=0.0                 # lora_dropout
LORA_TARGET_MODULES="q_proj,k_proj,q_attn,v_proj,o_proj" # lora_target_modules

def concat_input_output(input, output):
  """It concatenates the input and the LLM output"""
  text = []
  for i,o in zip(input, output):
    text.append(f'{i}\n  {o}')
  return text

def read_data(path, name):
    data = pd.read_parquet(f'{path}/{name}', engine='pyarrow')
    data = data[['id','input', 'output']]
    data['text'] = concat_input_output(data.input.values, data.output.values)
    return Dataset.from_pandas(data)

def get_model(path):
    quantizationConfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        low_cpu_mem_usage=True
    )
    # Set up lora
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES.split(","),
    )

    olmo = AutoModelForCausalLM.from_pretrained(path, quantization_config=quantizationConfig)
    olmo = prepare_model_for_kbit_training(olmo)
    return get_peft_model(olmo, peft_config)

class Unlearner(SFTTrainer):

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch:Optional[int]=None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss = -loss

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        return loss.detach()

def do_gradient_ascent(model,tokenizer,dataset,path_checkpoints):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = SFTConfig(
        max_seq_length=512,
        report_to='none',
        output_dir="/tmp",
        dataset_text_field="text",
        packing=True,
    )

    trainer = Unlearner(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(f'{path_checkpoints}/Gradient_Ascent_forget')

def do_fine_tune(model,tokenizer,dataset,path_checkpoints):
    training_args = SFTConfig(
        max_seq_length=512,
        report_to='none',
        output_dir="/tmp",
        dataset_text_field="text",
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(f"{path_checkpoints}/fine_tune_Gradient_Ascent")

def unlearn(path_forget, 
            path_retain, 
            path_model, 
            path_checkpoints):
    '''
    Unlearning sensitive content (forget dataset) from Large Language Models

    Args:
        path_forget (str): Path for the private forget dataset (jsonl/parquet files) 
        path_retain (str): Path for the private retain dataset.
        path_model (str): Path to the fine tuned model path (includes the tokenizer).
        path_checkpoints (str): Path to the output directory to store the unlearned checkpoints.
    '''
    forget_df = read_data(path_forget, 'forget_train-00000-of-00001.parquet')
    retain_df = read_data(path_retain, 'retain_train-00000-of-00001.parquet')
    model = get_model(path_model)
    tokenizer = AutoTokenizer.from_pretrained(path_model)
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    do_gradient_ascent(model,tokenizer,forget_df, path_checkpoints)

    model_forget = get_model(f'{path_checkpoints}/Gradient_Ascent_forget')
    do_fine_tune(model_forget,tokenizer,retain_df, path_checkpoints)

    rmtree(f'{path_checkpoints}/Gradient_Ascent_forget')
