#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QF2模型推理演示脚本

使用方法:
1. 修改main()函数中的model_path变量，指向你的模型目录
2. 运行脚本: python qf2_a5_inference_demo.py

示例:
model_path = "./output/model2507081350"  # 替换为你的模型路径

支持的模型路径格式:
- 本地训练模型: "./output/model{timestamp}/"
- HuggingFace模型: "Qwen/Qwen2.5-1.5B-Instruct"
- 其他本地路径: "/path/to/your/model/"
"""

# model_path = "./output/model2507091653/"  # 示例路径

model_path = "./output/qwen2_1_5B_qft_beta1000/"  # 示例路径
# model_path = "./output/qwen2_1_5B_qft_beta10/"  # 示例路径
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
import numpy as np
import os

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer"""
    if not model_path:
        raise ValueError("请指定model_path参数")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 检查是否有GPU可用
    if torch.cuda.is_available():
        print("使用GPU加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print("使用CPU加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float32,  # CPU上使用float32
            device_map="auto"
        )
    
    return model, tokenizer

def generate_response(model, tokenizer, messages):
    """Generate response"""
    # Build conversation text
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n"
    
    # Encode input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            step=None  # 推理时明确传递step=None
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取assistant响应 - 只返回user之后第一行不为空的内容
    lines = generated_text.split('\n')
    assistant_response = ""
    
    # 查找user: 的位置
    user_found = False
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('user:'):
            user_found = True
            # 从user: 后面开始查找第一行不为空的内容
            for j in range(i+1, len(lines)):
                line_content = lines[j].strip()
                if line_content and line_content.lower().startswith('assistant:'):
                    # 去掉"assistant:"前缀
                    assistant_response = line_content[len('assistant:'):].strip()
                    break
            break
    
    # 如果没有找到user:，尝试查找assistant:
    if not user_found:
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('assistant:'):
                # 从assistant: 后面开始查找第一行不为空的内容
                for j in range(i+1, len(lines)):
                    line_content = lines[j].strip()
                    if line_content:
                        assistant_response = line_content
                        break
                break
    
    # 如果还是没有找到，返回整个生成的内容（去掉输入部分）
    if not assistant_response:
        # 找到输入文本的结束位置
        input_text = ""
        for msg in messages:
            input_text += f"{msg['role']}: {msg['content']}\n"
        
        # 如果生成文本包含输入文本，只返回后面的部分
        if input_text in generated_text:
            remaining_text = generated_text.split(input_text, 1)[1].strip()
            # 取第一行不为空的内容
            remaining_lines = remaining_text.split('\n')
            for line in remaining_lines:
                if line.strip():
                    # 如果这行以"assistant:"开头，去掉前缀
                    line_content = line.strip()
                    if line_content.lower().startswith('assistant:'):
                        assistant_response = line_content[len('assistant:'):].strip()
                    else:
                        assistant_response = line_content
                    break
        else:
            # 否则返回整个生成文本的第一行不为空内容
            for line in lines:
                if line.strip():
                    # 如果这行以"assistant:"开头，去掉前缀
                    line_content = line.strip()
                    if line_content.lower().startswith('assistant:'):
                        assistant_response = line_content[len('assistant:'):].strip()
                    else:
                        assistant_response = line_content
                    break
    
    return assistant_response

def main():
    set_seed(42)
    

    
    try:
        # Load model once for all tests
        model, tokenizer = load_model_and_tokenizer(model_path)

        print(f"\n--- Testing with validate_6.jsonl ---")
        with open("./validate_6.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                sample_data = json.loads(line.strip())
                messages = [sample_data["messages"][0],sample_data["messages"][1]]
                
                # 提取system、user、assistant内容
                system_content = ""
                user_content = ""
                assistant_content = ""
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_content = msg["content"]
                    elif msg["role"] == "user":
                        user_content = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_content = msg["content"]
                
                # 生成模型回答
                model_answer = generate_response(model, tokenizer, messages)
                
                # 打印结果
                print(f"\n--- Sample {i} ---")
                print(f"System: {system_content}")
                print(f"User: {user_content}")
                # print(f"Assistant: {assistant_content}")
                print(f"Model Answer: {model_answer}")
                print("-" * 50)
        
        del model, tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 