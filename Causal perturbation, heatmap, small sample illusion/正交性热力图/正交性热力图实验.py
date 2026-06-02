#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正交性热力图 - 最终版（英文标签，高迭代）
跨模型：Qwen2-7B, Llama-3-8B, Mistral-7B
危害类别：Drugs, Hack, Theft, Murder, Bomb
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from datasets import load_dataset
from tqdm import tqdm
import joblib

# ========== 配置 ==========
DEVICE = "cuda"
MODEL_PATHS = {
    "Qwen2-7B": "/mnt/workspace/qwen2-7b-model/qwen/Qwen2-7B-Instruct",
    "Llama-3-8B": "/mnt/workspace/llama3_model/LLM-Research/Meta-Llama-3-8B-Instruct",
    "Mistral-7B": "/mnt/workspace/mistral_model/mistralai/Mistral-7B-Instruct-v0.3",
}
DATA_PATH = "/mnt/workspace/beavertails_data"
CACHE_DIR = "./heatmap_cache"   # 保存每个模型的隐状态和标签
os.makedirs(CACHE_DIR, exist_ok=True)

# 类别定义（英文关键词映射）
CATEGORY_KEYWORDS = {
    "Drugs": ["methamphetamine", "cocaine", "heroin", "drug", "meth", "amphetamine", "weed"],
    "Hack": ["hack", "bank account", "credit card", "phishing", "password", "breach"],
    "Theft": ["steal a car", "shoplift", "robbery", "theft", "steal", "burglary"],
    "Murder": ["murder", "kill", "assassinate", "homicide", "manslaughter"],
    "Bomb": ["bomb", "explosive", "dynamite", "c4", "detonate", "ied"],
}

def assign_category(prompt):
    prompt_lower = prompt.lower()
    for cat, kwlist in CATEGORY_KEYWORDS.items():
        if any(kw in prompt_lower for kw in kwlist):
            return cat
    return "Other"

# ========== 加载数据集并准备有害样本（按类别）==========
print("加载 BeaverTails 数据集...")
dataset_dict = load_dataset(DATA_PATH, trust_remote_code=False)
split_name = list(dataset_dict.keys())[0]
dataset = dataset_dict[split_name]

# 收集所有有害样本及其类别
harmful_samples = []  # (prompt, category)
for row in dataset:
    prompt = row.get("prompt", "")
    is_safe = row.get("is_safe", None)
    if is_safe == 0:
        cat = assign_category(prompt)
        if cat != "Other":
            harmful_samples.append((prompt, cat))
    if len(harmful_samples) >= 500:
        break
print(f"共收集有害样本: {len(harmful_samples)} 条，类别分布：")
for cat in CATEGORY_KEYWORDS.keys():
    count = sum(1 for _, c in harmful_samples if c == cat)
    print(f"  {cat}: {count}")

# 收集无害样本（固定200条，用于每个类别的平衡训练）
harmless_prompts = []
for row in dataset:
    prompt = row.get("prompt", "")
    is_safe = row.get("is_safe", None)
    if is_safe == 1:
        harmless_prompts.append(prompt)
    if len(harmless_prompts) >= 200:
        break
print(f"无害样本总数: {len(harmless_prompts)}")

# 对每个类别，取出最多 40 条有害样本
SAMPLE_PER_CAT = 40
category_samples = {}
for cat in CATEGORY_KEYWORDS.keys():
    cat_prompts = [p for p, c in harmful_samples if c == cat]
    if len(cat_prompts) > SAMPLE_PER_CAT:
        cat_prompts = cat_prompts[:SAMPLE_PER_CAT]
    category_samples[cat] = cat_prompts
    print(f"{cat}: 使用 {len(cat_prompts)} 条有害样本")

# ========== 辅助函数：应用聊天模板 ==========
def apply_chat_template(prompt, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ========== 提取隐状态（带缓存）==========
def get_hidden_states(model, tokenizer, texts, batch_size=8):
    model.eval()
    all_hidden = []
    for i in tqdm(range(0, len(texts), batch_size), desc="提取隐状态"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_hidden = last_hidden[torch.arange(len(batch)), seq_lens, :].float().cpu().numpy()
        all_hidden.append(batch_hidden)
    return np.concatenate(all_hidden, axis=0)

# 存储结果
results = {model_name: {} for model_name in MODEL_PATHS}

for model_name, model_path in MODEL_PATHS.items():
    print(f"\n===== 处理模型: {model_name} =====")
    
    # 尝试加载缓存
    cache_file = os.path.join(CACHE_DIR, f"{model_name}_embeddings.pkl")
    if os.path.exists(cache_file):
        print("加载缓存的隐状态...")
        cache_data = joblib.load(cache_file)
        X_dict = cache_data["X_dict"]
        y_dict = cache_data["y_dict"]
    else:
        # 加载 tokenizer 和模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        X_dict = {}
        y_dict = {}
        for cat, harmful_list in category_samples.items():
            if len(harmful_list) == 0:
                continue
            # 取等量无害样本
            np.random.seed(42)
            harm_inds = np.random.choice(len(harmless_prompts), size=len(harmful_list), replace=False)
            harmless_list = [harmless_prompts[i] for i in harm_inds]
            all_prompts = harmful_list + harmless_list
            labels = [1]*len(harmful_list) + [0]*len(harmless_list)
            
            # 转为聊天格式
            chat_texts = [apply_chat_template(p, tokenizer) for p in all_prompts]
            # 提取隐状态
            X = get_hidden_states(model, tokenizer, chat_texts, batch_size=8)
            X_dict[cat] = X
            y_dict[cat] = np.array(labels)
            print(f"  {cat}: X.shape = {X.shape}")
        
        # 缓存
        joblib.dump({"X_dict": X_dict, "y_dict": y_dict}, cache_file)
        del model
        torch.cuda.empty_cache()
    
    # 对每个类别计算 cos_sim
    for cat, X in X_dict.items():
        y = y_dict[cat]
        # 训练 SVM
        svm = LinearSVC(C=1.0, dual='auto', random_state=42, max_iter=20000)
        svm.fit(X, y)
        w = svm.coef_[0]
        w = w / np.linalg.norm(w)
        # PCA 方向
        pca = PCA(n_components=1)
        pca.fit(X)
        pca_dir = pca.components_[0].flatten()
        pca_dir = pca_dir / np.linalg.norm(pca_dir)
        cos_sim = np.dot(w, pca_dir)
        results[model_name][cat] = cos_sim
        print(f"  {cat}: cos_sim = {cos_sim:.4f}")

# ========== 绘制热力图 ==========
categories = list(CATEGORY_KEYWORDS.keys())
model_names = list(MODEL_PATHS.keys())
data = np.array([[results[m][c] for c in categories] for m in model_names])

plt.figure(figsize=(10, 6))
sns.heatmap(data, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
            xticklabels=categories, yticklabels=model_names)
plt.title("Cosine Similarity between PCA direction and SVM normal vector", fontsize=14)
plt.xlabel("Harmful Category", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.tight_layout()
plt.savefig("heatmap_final.png", dpi=300)
plt.show()
print("\n热力图已保存为 heatmap_final.png")