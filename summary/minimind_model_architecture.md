# MiniMind 模型架构解析

本文档对 `model_minimind.py` 文件的各个模块进行详细解析。

---

## 模块总览

| 模块名称 | 行号 | 功能描述 |
|---------|------|---------|
| MiniMindConfig | 10-44 | 模型超参数配置 |
| RMSNorm | 49-59 | RMS 归一化层 |
| 位置编码相关 | 61-88 | RoPE 旋转位置嵌入 |
| Attention | 90-133 | 多头注意力机制 |
| FeedForward | 135-145 | SwiGLU 前馈网络 |
| MOEFeedForward | 147-175 | 混合专家网络 |
| MiniMindBlock | 177-193 | Transformer 解码器块 |
| MiniMindModel | 195-227 | 基础模型主体 |
| MiniMindForCausalLM | 229-280 | 完整因果语言模型 |

---

## 1. MiniMindConfig (模型配置)

定义所有模型超参数：

- **基础参数**: `hidden_size`, `num_hidden_layers`, `vocab_size`
- **注意力配置**:
  - `num_attention_heads`: Q 头数量
  - `num_key_value_heads`: KV 头数量（GQA 优化）
  - `flash_attn`: 是否启用 Flash Attention
- **MoE 配置**: `use_moe`, `num_experts`, `topk_experts`
- **RoPE 配置**: `rope_theta`, `rope_scaling` (YaRN 插值，支持 32K 上下文)

---

## 2. RMSNorm (归一化)

```python
output = weight * (x / sqrt(mean(x²) + eps))
```

**特点**:
- 比 LayerNorm 更快、更省显存
- 仅使用均方根统计，无需均值
- 用于 Q/K 的归一化 (`q_norm`, `k_norm`)

---

## 3. 位置编码 (RoPE)

### 预计算频率
```python
precompute_freqs_cis(hidden_dim, max_seq_len, theta, scaling_factor)
```

### 应用 RoPE
```python
apply_rotary_pos_emb(q, k, cos, sin)
```

**特性**:
- 支持 **YaRN 插值**：通过 `rope_scaling` 扩展上下文长度
- 预计算 cos/sin 缓存，推理时直接复用

---

## 4. Attention (多头注意力)

### 核心特性
- **GQA (分组查询注意力)**: Q 头数 > KV 头数，减少 KV 缓存
- **Flash Attention**: 高效计算注意力（需 `flash_attn=True`）
- **Q/K RMSNorm**: 提升训练稳定性
- **KV 缓存支持**: 推理时缓存 past_key_value

### 核心公式
```python
# repeat_kv: 将 KV 头复制 N 次
k = repeat_kv(hidden_states, n_rep)
# 注意力计算
attn_output = F.scaled_dot_product_attention(q, k, v)
```

---

## 5. FeedForward (前馈网络)

### SwiGLU 架构
```python
def forward(x):
    return down(act_fn(gate(x)) * up(x))
```

**特点**:
- 使用 **SiLU/Swish 激活函数**
- 门控机制： gate_proj × up_proj → element-wise 乘积
- 比 GELU 表现更好

---

## 6. MOEFeedForward (混合专家网络)

### 架构
- **多个 FFN 专家**（默认 4 个）
- **Top-K 门控**：每次选择 1 个专家（可配置 topk）
- **负载均衡**：通过 `router_aux_loss_coef` 防止专家坍塌

```python
expert_outputs = [expert(x) for expert in experts]
gate_logits = gate(x)
weights = top_k(softmax(gate_logits), k=topk)
output = sum(weights[i] * expert_outputs[i])
```

---

## 7. MiniMindBlock (Transformer Block)

```
Input → RMSNorm → Attention → Add → RMSNorm → MLP → Add → Output
```

包含残差连接和后归一化设计。

---

## 8. MiniMindModel (基础模型)

结构组成：
1. **embed_tokens**: 词嵌入层
2. **layers**: N 层 MiniMindBlock
3. **norm**: 输出 RMSNorm
4. **freqs_cis**: RoPE 频率缓存

---

## 9. MiniMindForCausalLM (完整语言模型)

### 核心特性
- **权重共享**: `embed_tokens` 与 `lm_head` 共享 embedding 矩阵
- **语言模型头**: `lm_head` 将 hidden_states 投影到词表维度
- **自回归生成**: 继承 `GenerationMixin`，支持自回归采样

### Forward 流程
```python
def forward(self, input_ids, labels=None):
    hidden_states = self.model(input_ids)
    logits = self.lm_head(hidden_states)
    
    if labels is not None:
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        return loss
    return logits
```

### 生成参数
- `temperature`: 温度采样
- `top_p`: Nucleus 采样
- `top_k`: Top-K 采样
- `repetition_penalty`: 重复惩罚

---

## 架构图示

```
MiniMindForCausalLM
├── lm_head (输出层，vocab_size)
│
└── MiniMindModel
    ├── embed_tokens (词嵌入)
    │
    ├── layers: [MiniMindBlock × num_layers]
    │   ├── Attention (GQA + Flash + RoPE)
    │   │   ├── q_norm / k_norm (RMSNorm)
    │   │   └── repeat_kv
    │   │
    │   └── MLP
    │       ├── SwiGLU (标准模型)
    │       └── MOEFeedForward (MoE 模型)
    │
    └── norm (RMSNorm)
```

---

## 关键技术点总结

| 技术 | 作用 |
|------|------|
| GQA | 减少 KV 缓存，提升推理效率 |
| Flash Attention | 加速注意力计算，降低显存 |
| RoPE + YaRN | 支持长上下文（32K+）|
| SwiGLU | 更强的非线性表达能力 |
| MoE | 扩展模型容量，保持推理效率 |
| 权重共享 | 减少参数量 |

---

*文档生成时间: 2024*