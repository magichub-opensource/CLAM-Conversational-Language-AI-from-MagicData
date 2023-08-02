import os,sys,argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import re
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# modelpath = 'models/Chinese-llama2-CLAM-7b' # local path
modelpath = 'MagicHub/Chinese-llama2-CLAM-7b' # huggingface repo 

print(f'model path: {modelpath}')
model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="cuda:0", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)

prompt = "歌剧和京剧的区别是什么？\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(
        inputs.input_ids, do_sample=True, max_new_tokens=1024, top_k=10, top_p=0.1, temperature=0.5, repetition_penalty=1.18, 
        eos_token_id=2, bos_token_id=1, pad_token_id=0, typical_p=1.0,encoder_repetition_penalty=1,
        )
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
cleaned_response = re.sub('^'+prompt,'', response)
print(f'输入：\n{prompt}\n')
print(f"输出：\n{cleaned_response}\n")
