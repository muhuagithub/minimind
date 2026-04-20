# MiniMind PPO 强化学习代码总结

这份总结面向“已经看过代码，但想把整体流程彻底串起来”的场景。

---

## 1. 这套代码整体在做什么

这是一套基于 **PPO（Proximal Policy Optimization）** 的 LLM 强化学习训练代码。

它不是从零训练语言模型，而是建立在已经完成：

- 预训练（Pretrain）
- 监督微调（Full SFT）

之后，再继续做强化学习优化。

它的目标是：

> 让模型不仅会“按监督数据模仿回答”，还会根据 reward 信号，进一步学会产出更高分的回答。

---

## 2. PPO 训练里几个核心角色

### 2.1 Actor

Actor 就是当前真正生成回答的语言模型，也就是 **policy model**。

它负责：

- 接收 prompt
- 逐 token 生成 response
- 在 PPO 中被更新

代码里对应：

- `actor_model`
- `rollout_engine` 内部用到的 policy model

可以把它理解成：

> 真正答题的学生。

---

### 2.2 Reward Model

Reward model 负责给最终回答打分。

它不生成文本，而是在模型回答完之后，根据：

- prompt / messages
- answer

输出一个 reward score。

代码里对应：

- `reward_model = LMForRewardModel(...)`
- `calculate_rewards(...)` 里调用的 `reward_model.get_score(...)`

可以把它理解成：

> 最终给答卷打分的老师。

---

### 2.3 Critic

Critic 是价值模型（value model）。

它不负责生成回答，而是负责估计：

> 当前生成到这个位置时，未来大概还能拿到多少总奖励。

代码里对应：

- `CriticModel`
- `critic_model`

Critic 输出的是 **value**，不是 token logits。

可以把它理解成：

> 在答题过程中，预估“这题最后大概能拿几分”的人。

---

### 2.4 Reference Model

Reference model 是一个冻结的参考模型，用来约束 PPO 训练不要把 actor 更新得太猛。

代码里对应：

- `ref_model`

它的主要作用是计算一个 reference KL penalty：

> 当前 actor 不要偏离原来的 SFT 模型太远。

否则强化学习容易出现：

- 语言风格漂移
- 奖励投机
- 输出质量下降

---

### 2.5 Rollout Engine

Rollout engine 是采样引擎，用来让 actor 真正生成回答。

它支持两种后端：

- `torch`：本地 PyTorch generate
- `sglang`：远程推理服务

代码里对应：

- `create_rollout_engine(...)`
- `TorchRolloutEngine`
- `SGLangRolloutEngine`

它返回：

- `output_ids`
- `completion_ids`
- `per_token_logps`
- `completions`

---

## 3. 这套 PPO 代码的整体链路

可以先把完整过程看成：

```text
Prompt
-> Actor rollout 生成 response
-> Reward Model + 规则打分得到 reward
-> Critic 估计 value
-> 用 reward 和 value 算 advantage / return
-> 用 PPO 更新 actor
-> 用 value loss 更新 critic
```

更细一点：

```text
1. 从 RLAIFDataset 取 prompt
2. tokenizer 编码
3. rollout_engine 用 actor 生成 response
4. calculate_rewards 对 response 打分
5. critic 对 response token 输出 old values
6. actor 重新前向，取 old logprobs
7. ref_model 前向，取 reference logprobs
8. 用 GAE 计算 advantages
9. 用 advantages + old values 得到 returns
10. 多轮 PPO mini-batch 更新 actor 和 critic
11. 定期同步 rollout engine 的 policy
12. 定期保存 actor / critic / optimizer / scheduler
```

---

## 4. 主入口脚本在做什么

PPO 主入口代码做了这些事：

1. 解析训练参数
2. 初始化分布式环境和随机种子
3. 构建 `MiniMindConfig`
4. 初始化：
   - `actor_model`
   - `ref_model`
   - `critic_model`
   - `reward_model`
   - `rollout_engine`
5. 构建：
   - `RLAIFDataset`
   - `actor_optimizer`
   - `critic_optimizer`
   - `actor_scheduler`
   - `critic_scheduler`
6. 如果需要则恢复 checkpoint
7. 可选 `torch.compile` 和 DDP 包装
8. 按 epoch 调用 `ppo_train_epoch(...)`
9. 训练结束后清理分布式进程

### 4.1 Actor 初始化

Actor 从 `full_sft` 权重开始加载。

这表示 PPO 训练不是从零开始，而是：

> 先有一个已经会基本对话和回答的 SFT 模型，再继续做强化学习。

### 4.2 Reference Model 初始化

Reference model 也从同一个 `full_sft` 权重加载，但会被冻结。

它不训练，只用来提供 reference KL。

### 4.3 Critic 初始化

Critic 的 backbone 也复用同一个基础权重，但额外多了一个 `value_head`。

也就是说：

- Actor：输出 token logits
- Critic：输出 token value

### 4.4 Reward Model 初始化

Reward model 从单独的 reward model 路径加载，用来给回答打分。

### 4.5 Rollout Engine 初始化

Rollout engine 绑定当前 actor，用于后续 rollout。

---

## 5. Reward 是怎么来的

`calculate_rewards(...)` 里使用的是 **混合奖励**。

总 reward 不是只来自 reward model，而是几项叠加：

### 5.1 长度奖励

- 回答长度在合理范围内：加分
- 太短或太长：减分

### 5.2 thinking 格式奖励

如果回答里带 `</think>`，代码会检查：

- 思考部分长度是否合理
- `</think>` 是否只出现一次

如果格式好，会加分；否则减分。

### 5.3 重复惩罚

用 `rep_penalty(...)` 检查 n-gram 重复。

如果回答重复很多，会扣分。

### 5.4 Reward Model 分数

最后再加上 reward model 对 `(prompt, answer)` 的打分。

所以总 reward 可以看成：

```text
总reward = 规则奖励 + 格式奖励 - 重复惩罚 + reward_model_score
```

---

## 6. CriticModel 在做什么

`CriticModel` 继承自 `MiniMindForCausalLM`，但不再使用 `lm_head` 做词表预测，而是额外加了：

- `value_head = nn.Linear(hidden_size, 1)`

它的 forward 过程是：

```text
input_ids
-> backbone model
-> hidden_states
-> value_head
-> values [B, T]
```

也就是：

> 对每个 token 位置输出一个 value。

这个 value 表示：

> 站在当前位置看，未来大概能得到多少总奖励。

---

## 7. Rollout 阶段在做什么

在 `ppo_train_epoch(...)` 里，每个 batch 先取出：

- `prompts`

再做 tokenizer 编码，得到：

- `input_ids`
- `attention_mask`

然后：

```text
rollout_engine.rollout(...)
```

让 actor 真正生成回答。

生成后得到：

- `gen_out`：完整序列 `[prompt + response]`
- `responses_text`：解码后的文本回答

随后调用：

```text
calculate_rewards(prompts, responses_text, reward_model)
```

得到每条样本的最终 reward。

---

## 8. 为什么只训练 response，而不训练 prompt

在 PPO 里，prompt 是输入，不是 actor 自己生成的动作。

所以真正需要更新的是：

> response 这部分 token 的策略。

因此代码会构造很多 mask：

- `full_mask`
- `resp_mask`
- `resp_policy_mask`
- `resp_value_mask`

它们的目的都是：

> 只让 response token 参与 PPO policy loss 和 value loss，prompt 部分不参与。

---

## 9. Old logprob、old value、reference logprob 是什么

### 9.1 old_resp_logp

这是 rollout 时旧 actor 对每个 response token 的 logprob。

后面 PPO 会重新算当前 actor 的 logprob，然后做比值：

```text
ratio = exp(current_logprob - old_logprob)
```

### 9.2 old_resp_values

这是 rollout 时 critic 对每个 response token 给出的旧 value 估计。

后面用于：

- GAE
- return
- value clipping

### 9.3 ref_resp_logp

这是 reference model 对每个 response token 的 logprob。

后面用于构造 KL penalty，约束 actor 不要偏离 reference model 太远。

---

## 10. reward 为什么只加在最后一个 token 上

代码里会先构造：

- `token_rewards`

然后把整条 response 的最终 reward 放到最后一个有效 token 上。

这是因为：

> 在很多 LLM 强化学习设置里，reward 是整条回答结束后才知道的。

所以一种常见做法是：

- 中间 token 暂时 reward 为 0
- 最后一个 token 拿到整个 sequence reward
- 再通过 GAE / return 机制把这个信息往前传播

---

## 11. GAE 是怎么工作的

### 11.1 什么是 GAE

GAE（Generalized Advantage Estimation）是广义优势估计。

它的目的不是直接拿 `reward - value`，而是用一种更平滑、更低方差的方式估计 advantage。

核心递推里会用到：

- `gamma`：折扣因子
- `lam`：GAE 平滑参数

### 11.2 这段代码里的作用

代码会从后往前遍历 response token，递推计算：

- `delta`
- `lastgaelam`
- `advantages`

最终得到：

- `advantages [B, R]`
- `returns = advantages + old_resp_values`

### 11.3 Advantage 直观解释

Advantage 表示：

> 这次生成比 critic 原本预期的更好多少 / 更差多少。

如果 advantage 大于 0，说明这类行为应该更鼓励；
如果小于 0，说明应该被压低概率。

---

## 12. PPO 更新阶段在做什么

Rollout 完一批数据后，代码不会立刻只做一步更新，而是：

- 把这一批 rollout 数据复用多轮
- 每轮再切成 mini-batch
- 在这些 mini-batch 上做 PPO 更新

这就是：

- `ppo_update_iters`
- `mini_batch_size`

存在的原因。

因为 rollout 很贵，所以同一批样本通常会重复利用几轮。

---

## 13. PPO 的核心量：ratio

PPO 里最核心的是：

```text
ratio = pi_theta / pi_old
```

在代码里是：

```text
ratio = exp(current_logprob - old_logprob)
```

它表示：

> 当前策略和 rollout 旧策略相比，对同一个 token 的概率改了多少。

如果 ratio 太大或太小，说明更新过猛。

---

## 14. PPO policy loss 是怎么回事

PPO policy loss 的核心思想是：

- 如果 advantage > 0：提高这些 token 的概率
- 如果 advantage < 0：降低这些 token 的概率

但不能无限制地改，所以要做 clipping：

```text
clip(ratio, 1-eps, 1+eps)
```

这样可以防止：

- 一次更新走太猛
- 策略瞬间偏离 old policy 太远

最终 policy loss 还会额外加上：

- reference KL penalty

也就是：

> 不仅别离 old policy 太远，也别离 ref_model 太远。

---

## 15. Critic 的 value loss 是怎么回事

Critic 的目标不是输出 reward，而是拟合 return。

所以 value loss 本质上是在做：

```text
让 critic 的 value 接近 returns
```

代码里还用了 value clipping。

它的目的和 PPO policy clipping 类似：

> 防止 critic 一次更新跳太大。

---

## 16. Approx KL 和 KL_ref 的区别

### 16.1 Approx KL

这是当前 actor 和 rollout 时旧 actor 的差距。

它主要用于：

- PPO 训练中的 early stop
- 监控更新是否过猛

### 16.2 KL_ref

这是当前 actor 和参考模型 ref_model 的差距。

它主要用于：

- 保持 actor 不偏离基础 SFT 模型太远
- 防止 reward hacking 或语言能力退化

---

## 17. 为什么会有 early stop KL

代码中如果 `approx_kl` 超过阈值：

- 就会触发 `stop_ppo`

原因是：

> 如果当前策略和旧策略差太多，继续更新可能会让 PPO 不稳定。

值得注意的是，代码不是直接 `break`，而是让 loss 乘 0。

这样做是为了：

- 不打断 DDP 的 forward-backward 通信闭环
- 避免某些卡提前退出，另一些卡继续 backward，导致死锁

这是比较工程化、很重要的细节。

---

## 18. Actor 和 Critic 是如何一起更新的

每个 mini-batch 中，代码都会算：

- `policy_loss`
- `value_loss`
- `aux_loss`（如果 MoE）

然后总 loss 是：

```text
loss = policy_loss + vf_coef * value_loss + aux_loss
```

接着反向传播。

之后会分别：

- clip actor 梯度
- clip critic 梯度
- `actor_optimizer.step()`
- `critic_optimizer.step()`
- `actor_scheduler.step()`
- `critic_scheduler.step()`

也就是说：

- actor 和 critic 同时在一个训练循环里更新
- 但它们有各自独立的 optimizer 和 scheduler

---

## 19. 为什么 rollout_engine 要定期 update_policy

训练过程中 actor 在不断更新。

如果 rollout engine 还拿旧权重做采样，后续生成和训练就会不一致。

所以代码里会定期：

```text
rollout_engine.update_policy(actor_model)
```

如果是 torch rollout，这只是替换本地模型引用；
如果是 sglang rollout，则会把新权重同步给远程推理服务。

---

## 20. checkpoint 里保存了什么

PPO 训练的 checkpoint 比普通训练复杂，因为要保存两套模型和优化状态：

- actor_model
- critic_model
- actor_optimizer
- critic_optimizer
- actor_scheduler
- critic_scheduler
- 当前 epoch / step

主流程恢复训练时，也会把这些全部恢复回来。

---

## 21. 这套 PPO 训练和普通 SFT 的本质区别

### 普通 SFT

流程是：

```text
(input_ids, labels)
-> 前向
-> cross entropy loss
-> backward
-> optimizer.step
```

训练目标是：

> 模仿监督数据里的标准答案。

### PPO 强化学习

流程是：

```text
prompt
-> actor rollout 生成回答
-> reward model + 规则打分
-> critic 估 value
-> GAE 算 advantage
-> PPO 更新 actor
-> value loss 更新 critic
```

训练目标是：

> 让模型学会产出 reward 更高的回答。

所以 PPO 比 SFT 多出来的关键部分有：

- rollout
- reward
- critic
- advantage / return
- PPO clipping
- reference KL penalty

---

## 22. 一张完整流程图

```text
主流程初始化：
  1. 解析参数
  2. 初始化分布式 / 随机种子
  3. 初始化 actor / ref_model / critic / reward_model / rollout_engine
  4. 初始化数据集、优化器、scheduler
  5. 如有 checkpoint 则恢复
  6. 可选 compile / DDP

训练阶段（每个 batch）：
  1. 从 RLAIFDataset 取 prompt
  2. tokenizer 编码 prompt
  3. rollout_engine 用 actor 生成 response
  4. calculate_rewards 得到 reward
  5. critic 计算 old values
  6. actor 计算 old logprobs
  7. ref_model 计算 ref logprobs
  8. 构造 token_rewards
  9. 用 GAE 计算 advantages
 10. 得到 returns
 11. 对同一批 rollout 做多轮 PPO mini-batch 更新：
      - 当前 actor 重新算 logprob
      - 当前 critic 重新算 value
      - 算 ratio / approx_kl / kl_ref / clipfrac
      - 算 policy_loss
      - 算 value_loss
      - backward
      - actor_optimizer.step()
      - critic_optimizer.step()
 12. 定期更新 rollout_engine 的 actor 权重
 13. 记录日志
 14. 保存 checkpoint
```

---

## 23. 最后一句总结

这套代码实现的是一个完整的 **LLM PPO 强化学习训练框架**：它先从 SFT 模型初始化 actor、ref_model 和 critic，再通过 rollout 让 actor 对 prompt 生成回答，用 reward model 和手工规则计算 reward，用 critic 输出 value，并通过 GAE 得到 advantage / return；随后在同一批 rollout 数据上执行多轮 PPO 更新，通过 clipped policy loss 优化 actor、通过 value loss 优化 critic，同时用 reference KL 限制 actor 不要偏离基础模型太远，最后周期性同步 rollout 引擎并保存完整训练状态。
