# MiniMind DPO 训练详解

## 📋 DPO 训练流程概览

这个仓库的 DPO 实现非常清晰，核心文件是 `trainer/train_dpo.py`。训练流程如下：

```
偏好数据 (chosen vs rejected)
    ↓
DPODataset 处理成 token 序列
    ↓
双模型前向传播
    ├─ ref_model (冻结，不更新)
    └─ policy_model (训练中)
    ↓
计算 log_probs
    ↓
DPO loss 计算
    ↓
更新 policy_model
```

---

## 🔑 核心实现细节

### 1. 数据格式 (`DPODataset`)

数据集 `dpo.jsonl` 中每条记录包含两个字段：

```json
{
  "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "好的回答"}],
  "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "差的回答"}]
}
```

`DPODataset` 将其转换为：
- `x_chosen` / `x_rejected`：输入序列
- `y_chosen` / `y_rejected`：目标序列（下一个 token）
- `mask_chosen` / `mask_rejected`：只对 assistant 的回答部分计算 loss

### 2. DPO Loss 实现 (train_dpo.py 第 33-49 行)

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 1. 对每个样本求和 log_probs
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    
    # 2. 分离 chosen 和 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]
    
    # 3. 计算对数几率
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # 4. DPO 核心公式
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    
    return loss.mean()
```

**关键点**：
- `pi_logratios`：当前策略模型中 chosen vs rejected 的对数几率差
- `ref_logratios`：参考模型中 chosen vs rejected 的对数几率差
- `logits = pi_logratios - ref_logratios`：衡量策略相对于参考模型对 chosen 的偏好提升
- `beta`：温度参数（默认 0.15），控制对齐强度

### 3. 训练循环 (train_dpo.py 第 52-128 行)

每个 batch 的处理：

```python
# 拼接 chosen 和 rejected 数据
x = torch.cat([x_chosen, x_rejected], dim=0)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)

# 参考模型前向（不计算梯度）
with torch.no_grad():
    ref_outputs = ref_model(x)
    ref_logits = ref_outputs.logits
ref_log_probs = logits_to_log_probs(ref_logits, y)

# 策略模型前向
outputs = model(x)
logits = outputs.logits
policy_log_probs = logits_to_log_probs(logits, y)

# 计算 DPO loss
dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
loss = dpo_loss_val + outputs.aux_loss  # 加上 MoE 的负载均衡 loss
```

### 4. 双模型初始化 (train_dpo.py 第 181-188 行)

```python
# 策略模型：需要训练
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

# 参考模型：冻结，不更新
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)
```

两个模型都加载相同的初始权重（通常是 SFT 后的权重），但只有 `model` 会被更新。

---

## 🎯 与标准 DPO 公式的对应

标准 DPO 公式：
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} - \log \frac{\pi_{ref}(y_w|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$$

代码中的实现：
```python
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs  # log(π(y_w)/π(y_l))
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs       # log(π_ref(y_w)/π_ref(y_l))
logits = pi_logratios - ref_logratios                               # 两项相减
loss = -F.logsigmoid(beta * logits)                                # -log σ(β·logits)
```

完全对应！

---

## 📊 训练参数

```bash
python trainer/train_dpo.py \
    --from_weight full_sft \      # 基于 SFT 权重
    --data_path ../dataset/dpo.jsonl \
    --batch_size 4 \
    --learning_rate 4e-8 \        # 学习率很小，避免遗忘
    --beta 0.15 \                 # DPO 温度参数
    --epochs 1 \
    --max_seq_len 1024
```

---

## ✨ 实现亮点

1. **纯 PyTorch 实现**：没有依赖 trl 等库，代码透明
2. **显存友好**：参考模型用 `torch.no_grad()`，只计算一次前向
3. **支持 MoE**：自动处理 `aux_loss`（负载均衡）
4. **分布式训练**：支持 DDP
5. **断点续训**：可自动恢复训练状态

---

## 📚 相关文件

- `trainer/train_dpo.py`：DPO 训练主程序
- `dataset/lm_dataset.py`：包含 `DPODataset` 类
- `summary/README_dpo_ppo_grpo.md`：DPO/PPO/GRPO 对比说明
