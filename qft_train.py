#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import os
from datetime import datetime
import csv
import json
import numpy as np
import random

qfcoeff = 0.99
qfbeta = 1000
# 模型路径
model_path = "/home/ls/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct"
# model_path = "./output/model_noResidual/"
data_path = "./train_12000.jsonl"
validation_path = "./validate_6.jsonl"

# 设置使用两块A6000 GPU (GPU 1和GPU 3)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 添加seed设置函数

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed = 1  # 可自定义
set_seed(seed)


class CustomTrainer(Trainer):
    """
    自定义Trainer，能够将当前训练step传递给模型
    """
    
    def __init__(self, *args, qfcoeff=0.99, qfbeta=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_data = []
        self.validation_data = []
        self.eval_results = []  # 添加eval结果列表
        self.qfcoeff = qfcoeff  # 保存qfcoeff参数
        self.qfbeta = qfbeta  # 保存qfbeta参数
        # 添加loss移动平均相关参数
        self.loss_moving_avg = None
        self.moving_avg_decay = 0.9  # 移动平均衰减因子
        self.loss_history = []  # 存储历史loss用于计算移动平均
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写compute_loss方法，传递当前step给模型
        """
        # 获取当前训练step
        current_step = self.state.global_step
        
        # 将step添加到inputs中
        inputs['step'] = current_step
        
        # 计算alpha，使用qfcoeff参数
        alpha_raw = self.qfcoeff ** current_step
        alpha = 0.0 if alpha_raw < 0.01 else alpha_raw
  
        
        # 调用模型的forward方法
        outputs = model(**inputs)
        
        # 获取loss
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # 计算loss移动平均
        self._update_loss_moving_average(loss.item())
        
        # 记录loss数据
        self.loss_data.append({
            'step': current_step,
            'loss': float(f"{loss.item():.2f}"),
            'loss_moving_avg': float(f"{self.loss_moving_avg:.2f}"),
            'alpha': float(f"{alpha:.4f}")
        })
        
        return (loss, outputs) if return_outputs else loss
    
    def _update_loss_moving_average(self, current_loss):
        """
        更新loss移动平均
        """
        if self.loss_moving_avg is None:
            self.loss_moving_avg = current_loss
        else:
            self.loss_moving_avg = self.moving_avg_decay * self.loss_moving_avg + (1 - self.moving_avg_decay) * current_loss
        
        self.loss_history.append(current_loss)
    
    def log(self, logs):
        """
        重写log方法，添加step、alpha和loss移动平均到日志中
        """
        # 获取当前step
        current_step = self.state.global_step
        
        # 计算alpha，使用qfcoeff参数
        alpha_raw = self.qfcoeff ** current_step
        alpha = 0.0 if alpha_raw < 0.01 else alpha_raw
        
        # 添加自定义字段到日志（全部保留两位小数）
        logs['step'] = current_step
        logs['alpha'] = float(f"{alpha:.4f}")
        if self.loss_moving_avg is not None:
            logs['loss_moving_avg'] = float(f"{self.loss_moving_avg:.2f}")
        # learning_rate、loss等也格式化
        if 'loss' in logs:
            logs['loss'] = float(f"{logs['loss']:.2f}")
        if 'learning_rate' in logs:
            logs['learning_rate'] = float(f"{logs['learning_rate']:.6f}")
        if 'grad_norm' in logs:
            logs['grad_norm'] = float(f"{logs['grad_norm']:.2f}")
        
        # 调用父类的log方法
        super().log(logs)
    
    def save_eval_data(self, output_dir):
        """
        保存eval数据到CSV文件
        """
        if self.eval_results:
            eval_file = os.path.join(output_dir, 'eval.csv')
            with open(eval_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['step', 'eval_loss', 'eval_accuracy', 'alpha'])
                writer.writeheader()
                writer.writerows(self.eval_results)
            print(f"Eval数据已保存到: {eval_file}")
        else:
            print("没有eval数据需要保存")
    
    def save_loss_data(self, output_dir):
        """
        保存loss数据到CSV文件，包含移动平均
        """
        loss_file = os.path.join(output_dir, 'loss.csv')
        with open(loss_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'loss', 'loss_moving_avg', 'alpha'])
            writer.writeheader()
            writer.writerows(self.loss_data)
        print(f"Loss数据已保存到: {loss_file}")
        
        # 保存loss历史数据（用于绘图）
        history_file = os.path.join(output_dir, 'loss_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump({
                'loss_history': self.loss_history,
                'moving_avg_decay': self.moving_avg_decay
            }, f, indent=2)
        print(f"Loss历史数据已保存到: {history_file}")
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写evaluate方法，添加准确率计算并保存结果
        """
        # 调用父类的evaluate方法
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 计算准确率
        if eval_dataset is not None:
            accuracy = self._compute_accuracy(eval_dataset)
            metrics[f"{metric_key_prefix}_accuracy"] = accuracy
            print(f"Validation Accuracy: {accuracy:.4f}")
            
            # 生成并打印预测答案示例
            self._generate_predictions(eval_dataset, num_samples=3)
        
        # 保存eval结果
        current_step = self.state.global_step
        eval_result = {
            'step': current_step,
            'eval_loss': float(f"{metrics.get('eval_loss', 0):.4f}"),
            'eval_accuracy': float(f"{metrics.get('eval_accuracy', 0):.4f}"),
            'alpha': float(f"{self.qfcoeff ** current_step:.4f}")
        }
        self.eval_results.append(eval_result)
        
        return metrics
    
    def _compute_accuracy(self, eval_dataset):
        """
        计算验证集准确率
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(len(eval_dataset)):
                # 获取样本
                sample = eval_dataset[i]
                input_ids = torch.tensor([sample['input_ids']]).to(self.model.device)
                attention_mask = torch.tensor([sample['attention_mask']]).to(self.model.device)
                
                # 获取最后一个token的预测
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # 获取最后一个token的预测和真实标签
                last_token_logits = logits[0, -1, :]
                predicted_token = torch.argmax(last_token_logits).item()
                true_token = sample['input_ids'][-1]
                
                if predicted_token == true_token:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _generate_predictions(self, eval_dataset, num_samples=3):
        """
        生成预测答案并打印
        """
        self.model.eval()
        
        # 获取tokenizer
        tokenizer = self.tokenizer if hasattr(self, 'tokenizer') else None
        if tokenizer is None:
            print("警告: 无法获取tokenizer，跳过预测答案生成")
            return
        
        print(f"\n=== 预测答案示例 (Step {self.state.global_step}) ===")
        
        with torch.no_grad():
            for i in range(min(num_samples, len(eval_dataset))):
                # 获取样本
                sample = eval_dataset[i]
                input_ids = torch.tensor([sample['input_ids']]).to(self.model.device)
                attention_mask = torch.tensor([sample['attention_mask']]).to(self.model.device)
                
                # 生成预测
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    step=None  # 推理时使用step=None
                )
                
                # 解码输入和输出
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取预测部分（去掉输入部分）
                if input_text in generated_text:
                    prediction = generated_text[len(input_text):].strip()
                else:
                    prediction = generated_text.strip()
                
                print(f"\n样本 {i+1}:")
                print(f"输入: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
                print(f"预测: {prediction[:100]}{'...' if len(prediction) > 100 else ''}")
                print("-" * 50)
        
        print("=" * 50)

def train_with_dynamic_alpha():
    """
    使用动态alpha进行训练
    只在第23层使用自定义激活函数：silu(x) = x^(qfcoeff^step) * σ(x)
    """

    
    # 定义qfcoeff参数
    # qfcoeff = 0.99  # 可自定义的衰减系数
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
    
    # 生成时间戳文件夹名称
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    output_dir = f"./output/model{timestamp}"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    import json
    from datasets import Dataset
    
    def read_jsonl(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    
    def preprocess(example):
        messages = example["messages"]
        text = ""
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n"
        return {"text": text}
    
    # 构建训练Dataset
    raw_data = list(read_jsonl(data_path))
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(preprocess)
    
    # 构建验证Dataset
    validation_raw_data = list(read_jsonl(validation_path))
    validation_dataset = Dataset.from_list(validation_raw_data)
    validation_dataset = validation_dataset.map(preprocess)
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    # 将qfcoeff参数设置到模型配置中
    model.config.qfcoeff = qfcoeff
    model.config.qfbeta = qfbeta

    # 冻结除第22、23、24层外的所有层
    print("冻结除第22、23、24层外的所有层...")
    for name, param in model.named_parameters():
        # 检查是否是第22、23、24层的参数
        if any(f"layers.{layer}" in name for layer in ["22", "23", "24"]):
            print(f"保持可训练: {name}")
            param.requires_grad = True
        else:
            print(f"冻结参数: {name}")
            param.requires_grad = False
    
    # 统计可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # 分词
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        save_strategy="no",  # 不保存checkpoint
        logging_steps=10,
        fp16=True,
        report_to="none",
        # 每1000步评估一次
        eval_steps=1000,
        eval_strategy="steps",
        load_best_model_at_end=False,  # 不需要加载最佳模型
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,  # 加入seed参数
        # 单GPU训练参数
        dataloader_pin_memory=False,  # 避免内存问题
        remove_unused_columns=False,  # 保留所有列
        # 移除可能导致版本兼容性问题的参数
        dataloader_num_workers=0,  # 避免多进程问题
        # 固定学习率设置
        learning_rate=5e-5,  # 设置固定学习率
        lr_scheduler_type="constant",  # 使用常数学习率调度器
        warmup_steps=0,  # 不使用warmup
    )
    
    # 创建自定义Trainer，传递qfcoeff参数
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_validation_dataset,
        qfcoeff=qfcoeff,  # 传递qfcoeff参数
        qfbeta=qfbeta,  # 传递qfbeta参数
        data_collator=lambda x: {'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in x]),
                                'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in x]),
                                'labels': torch.stack([torch.tensor(item['input_ids']) for item in x])}
    )
    
    # 将tokenizer添加到trainer中，以便在evaluation时使用
    trainer.tokenizer = tokenizer
    
    # 开始训练
    print("开始训练，使用动态alpha...")
    print(f"输出目录: {output_dir}")
    print(f"QF系数: {qfcoeff}")
    print(f"QFbeta: {qfbeta}")
    print("训练配置:")
    print("- 只有第22、23、24层会被训练，其他层保持冻结")
    print("- 第22、24层：使用标准激活函数，参数可训练")
    print("- 第23层：使用自定义激活函数，参数可训练")
    print("- 只在第23层的MLP残差连接使用alpha衰减")
    print("- Self-Attention残差连接使用标准形式")
    print("- 其他层使用标准SiLU激活函数")
    print(f"- 自定义激活函数: silu(x) = x^({qfcoeff}^step) * σ(x)")
    print(f"- MLP残差连接: residual * ({qfcoeff}^step) + hidden_states (仅第23层)")
    print("- Alpha会随着step增加逐渐衰减到零")
    print("- 当alpha < 0.001时，设置为0")
    print("- 推理时默认alpha = 0")
    print("- Loss移动平均衰减因子: 0.9")
    print("- 每1000步进行一次评估")
    print("- 不保存checkpoint，只保存最终模型")
    print("- 学习率: 固定为 5e-5 (不使用学习率调度)")
    print(f"- 使用 {validation_path} 进行验证，计算准确率和loss")
    print(f"- 训练数据: {data_path}")
    print()
    
    trainer.train()
    
    # 最终验证
    print("\n=== 最终验证结果 ===")
    final_metrics = trainer.evaluate()
    print(f"Final Validation Loss: {final_metrics['eval_loss']:.6f}")
    print(f"Final Validation Accuracy: {final_metrics.get('eval_accuracy', 0):.4f}")
    
    # 保存模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存eval数据
    trainer.save_eval_data(output_dir)
    
    # 保存loss数据
    trainer.save_loss_data(output_dir)
    
    # 保存验证结果
    validation_results_file = os.path.join(output_dir, 'validation_results.json')
    with open(validation_results_file, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"验证结果已保存到: {validation_results_file}")
    
    print(f"训练完成！模型已保存到: {output_dir}")
    print("训练总结:")
    print("- 只有第22、23、24层被训练，其他层保持原始权重")
    print("- 第22、24层：使用标准激活函数，参数已更新")
    print("- 第23层：使用自定义激活函数，参数已更新")
    print("- 只在第23层的MLP残差连接应用alpha衰减")
    print("- Self-Attention残差连接保持标准形式")
    print("- Loss移动平均已计算并保存")
    print(f"- 验证集 ({validation_path}) 准确率和loss已计算并保存")
    print("- 动态alpha变化规律:")
    print(f"- alpha = {qfcoeff}^step")
    print("- 当alpha < 0.001时，设置为0")
    print(f"- Step 0: alpha = 1.000")
    print(f"- Step 100: alpha ≈ {qfcoeff**100:.6f}")
    print(f"- Step 1000: alpha = {qfcoeff**1000:.6f}")
    print("- 随着step增加，alpha逐渐衰减到零")
    print("- 推理时alpha = 0，相当于不使用MLP残差连接")
    print(f"- 训练数据: {data_path}")
    print(f"- 验证数据: {validation_path}")

if __name__ == "__main__":
    train_with_dynamic_alpha() 