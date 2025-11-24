#
#   用于模型推理
#

import torch
import json
import reranker

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

#加载微调模型
model_path = "/root/autodl-fs/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
peft_model = PeftModel.from_pretrained(model, "/root/autodl-fs/LLaMA-Factory/saves/LLaMA3-8B/lora/train_2024-06-27-00-22-25")
with open('/root/autodl-fs/dataset/test.json', 'r') as f:
    dataset=json.load(f)


list=[]
for item in dataset:

    # 进行重排
    key_contexts,keys=reranker.rerank(item['context'],item['answer'])

    # 按照不同类型的问题选用不同的提示词
    if item['answer']=='yes' or item['answer']=='no':
        prompt="[INST]Ask a multi-hop question.[/INST]Context:"+key_contexts+"""\n Answer:"""+item['answer']+'.\n Question:'
    else:
        prompt="[INST]Ask a multi-hop question, reference to those keys in context:"+keys+"[/INST]Context:"+key_contexts+"""\n Answer:"""+item['answer']+'.\n Question:'

    # prompt的一个历史版本
    # prompt2='Context:'+key_contexts+"""#######################
    # According to the above content, ask a question to """+item['answer']+'.Question:'

    # 生成问题，需要将输出长度控制在48tokens以内
    model_inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=48
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 收集生成的文本
    list.append({"_id":item["_id"],"question":response})

    print(response)

# 保存预测结果
with open ('results.json','w') as f:
    json.dump(list,f)