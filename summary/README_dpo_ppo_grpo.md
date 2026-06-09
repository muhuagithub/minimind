# README: DPO、PPO、GRPO 三种对齐训练方法总结

## 1. 总览

在大语言模型对齐训练中，常见的三类方法是：

- **DPO**：Direct Preference Optimization
- **PPO**：Proximal Policy Optimization
- **GRPO**：Group Relative Policy Optimization

它们的目标都很像：

> 让模型更偏向“人类更喜欢”的回答，或者更偏向“高 reward”的回答。

但它们的训练方式很不一样：

- **DPO**：直接用偏好对训练，不做在线强化学习
- **PPO**：标准 actor-critic 强化学习
- **GRPO**：不训练 critic，而是用组内相对比较替代 value baseline

---

## 2. 一句话理解三者

### DPO
**直接让模型学会：chosen 比 rejected 更应该被偏好。**

### PPO
**先采样回答，再打 reward，再用 critic 估 value，最后用 PPO 更新策略。**

### GRPO
**先对同一个 prompt 采样多条回答，再在组内比较谁更好，直接构造 advantage 更新策略。**

---

## 3. 三者的核心流程

## 3.1 DPO 流程

```text
prompt
-> 偏好数据 (chosen, rejected)
-> ref_model 计算 chosen/rejected 的 logprob
-> 当前 policy 计算 chosen/rejected 的 logprob
-> DPO loss
-> 更新 policy
```

### 核心特点
- 不需要 rollout
- 不需要 reward model 在线打分
- 不需要 critic
- 不需要 GAE / return / value loss
- 更像监督学习

---

## 3.2 PPO 流程

```text
prompt
-> actor rollout 生成回答
-> reward model / 规则函数打分
-> critic 估计 value
-> GAE 算 advantage / return
-> policy loss + value loss
-> 更新 actor 和 critic
```

### 核心特点
- 有 actor
- 有 critic
- 通常有 reward model
- 通常有 ref model 做 KL 约束
- 是标准强化学习 actor-critic 结构

---

## 3.3 GRPO 流程

```text
prompt
-> actor rollout（同一个 prompt 多采样几条回答）
-> reward model / 规则函数打分
-> 同组回答 reward 做标准化，得到 advantage
-> policy loss
-> 更新 actor
```

### 核心特点
- 不需要 critic
- 需要同 prompt 多采样
- advantage 来自组内相对 reward
- 通常仍然有 ref model 做 KL 约束
- 比 PPO 更轻一些

---

## 4. 三者最本质的区别

三者最核心的区别，不在“目标”，而在于：

> **advantage / 优化信号是怎么来的**

### DPO
不显式构造 reward、value、advantage。  
直接用：

- chosen
- rejected
- 当前 policy 和 reference model 的相对 logprob

构造一个偏好损失。

### PPO
advantage 来自：

```text
reward + critic(value) + GAE
```

也就是：

- 先 rollout
- 再打 reward
- 再用 critic 估 value
- 最后得到 advantage

### GRPO
advantage 来自：

```text
同一个 prompt 下多条回答的 reward 组内标准化
```

也就是：

- 不用 critic
- 直接把组内 reward 相对比较，转成 advantage

---

## 5. 三者是否需要哪些组件

| 组件 | DPO | PPO | GRPO |
|---|---|---|---|
| policy / actor | 需要 | 需要 | 需要 |
| ref model | 通常需要 | 通常需要 | 通常需要 |
| reward model | 不一定在线需要 | 通常需要 | 通常需要 |
| critic / value model | 不需要 | 需要 | 不需要 |
| rollout | 不需要 | 需要 | 需要 |
| 多回答组内比较 | 不需要 | 不需要 | 需要 |

---

## 6. 三者如何看待“参考模型 ref model”

### DPO
ref model 很重要。  
它用来定义“当前 policy 相对 reference 更偏向 chosen 的程度”。

### PPO
ref model 常用来加 KL 约束，防止 actor 偏离原始模型太远。

### GRPO
ref model 的作用和 PPO 很像，也是为了做 KL 惩罚，控制策略不要漂太远。

---

## 7. 三者和 reward 的关系

## 7.1 DPO
DPO 一般不直接在线使用 reward。  
它依赖的是：

- 人类偏好数据
- 或离线偏好对数据

即：

```text
(prompt, chosen, rejected)
```

所以它更像“偏好监督学习”。

---

## 7.2 PPO
PPO 需要 reward。  
reward 可以来自：

- reward model
- 规则函数
- 程序可验证结果
- 环境反馈

reward 是 PPO 的核心输入之一。

---

## 7.3 GRPO
GRPO 也需要 reward。  
但 reward 的作用不是配合 critic 做 value baseline，而是：

> 在同一 prompt 的多条回答内部做相对比较。

---

## 8. 三者的 advantage 是怎么来的

### DPO
没有显式 advantage。

### PPO
有显式 advantage，通常来自：

A = R - V

或者更准确地说，来自 GAE。

### GRPO
有显式 advantage，但不是来自 critic，而是来自组内相对奖励：

A_i = (r_i - mu_group) / (sigma_group + epsilon)

---

## 9. 三者训练成本对比

## 9.1 DPO
最省事、最省资源的一类。

原因：
- 不需要 rollout
- 不需要 critic
- 不需要在线 reward
- 不需要 PPO 这类复杂策略优化

通常训练成本最接近 SFT。

---

## 9.2 PPO
最重的一类。

原因：
- 需要 actor
- 需要 critic
- 需要 reward model
- 通常还需要 ref model
- 需要 rollout
- 需要保存 old logprob / value / advantage / return 等中间量

所以 PPO 通常最吃显存和算力。

---

## 9.3 GRPO
比 PPO 轻，但通常比 DPO 重。

原因：
- 不需要 critic，省掉一套模型和 value loss
- 但需要同 prompt 多采样，所以 rollout 成本会升高

因此它通常介于 DPO 和 PPO 之间。

---

## 10. 三者的优点

## 10.1 DPO 的优点
- 简单
- 稳定
- 显存友好
- 不需要 rollout
- 不需要 critic
- 不需要在线 reward

## 10.2 PPO 的优点
- 最标准的 RL 路线
- 可以直接最大化 reward
- 有 critic/value baseline
- 更适合真正在线强化学习和环境反馈

## 10.3 GRPO 的优点
- 不需要 critic
- 比 PPO 简单
- 仍然保留“基于 reward 的策略优化”味道
- 特别适合“同 prompt 下多候选比较”的场景

---

## 11. 三者的缺点

## 11.1 DPO 的缺点
- 依赖高质量偏好对数据
- 没有在线探索
- 不能像 RL 那样直接利用环境 reward 闭环

## 11.2 PPO 的缺点
- 最复杂
- 最吃显存
- 训练最不稳定
- 需要多个组件同时配合

## 11.3 GRPO 的缺点
- 必须同一个 prompt 多采样
- rollout 开销不小
- advantage 是组内相对优势，不是 critic 提供的绝对价值基线
- 更偏向“回答级”相对比较

---

## 12. 三者适合什么场景

## 更适合 DPO 的场景
- 已经有偏好对数据
- 不想上复杂 RL 训练
- 资源有限
- 想快速做对齐

## 更适合 PPO 的场景
- 需要真正利用 reward 闭环
- 有程序可验证 reward / 环境反馈
- 愿意承担更高的显存和工程复杂度
- 希望使用 actor-critic 路线

## 更适合 GRPO 的场景
- 希望保留 reward 驱动的训练方式
- 不想再训练 critic
- 可以接受同 prompt 多采样
- 任务更适合做组内相对比较

---

## 13. 最直观的类比

假设同一个问题，模型要回答。

### DPO
你已经有两份答案：
- A 更好
- B 更差

训练目标是：
让模型以后更偏向 A，而不是 B。

### PPO
模型先自己回答一份，然后老师打分。  
同时还有一个 critic 预测：
这份答案大概值多少分。

如果最终分数比预估高，就鼓励这种回答；  
如果比预估低，就减少这种回答。

### GRPO
模型对同一个问题一下子写出多份答案。  
老师分别打分，然后看：
这几份答案里，哪份相对最好？

相对更好的答案得到正 advantage，  
相对更差的得到负 advantage。

---

## 14. 最后一张对比表

| 方法 | 核心思想 | 是否需要 critic | 是否需要 rollout | 是否需要偏好对 | 是否需要 reward | 训练成本 |
|---|---|---:|---:|---:|---:|---:|
| DPO | 直接优化 chosen > rejected | 否 | 否 | 是 | 否（在线） | 低 |
| PPO | reward + critic + GAE + PPO 更新 | 是 | 是 | 否 | 是 | 高 |
| GRPO | 组内相对 reward 构造 advantage | 否 | 是 | 否 | 是 | 中 |

---

## 15. 一句话总结

- **DPO**：最像“偏好监督学习”
- **PPO**：最标准、最重的强化学习路线
- **GRPO**：介于两者之间，用“组内相对比较”替代 critic/value

如果只记一句话：

> **DPO 直接学偏好，PPO 用 critic 学 value，GRPO 用组内比较替代 critic。**
