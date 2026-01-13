这是一个基于我们之前所有深度探讨整理出的、可用于工程落地的完整架构设计方案。

我将这个架构命名为 **Phys-SetTransformer (PST)**。它结合了集合深度学习（处理无序观测）、Transformer（处理时序动力学）和物理几何约束（处理隐式关联）。

---

# 架构设计方案：Phys-SetTransformer (PST) 端到端状态估计器

## 1. 系统定义与问题建模

### 1.1 状态空间定义

我们需要估计的目标包含两部分：随时间变化的**动力学状态 (Dynamic State)** 和 不随时间变化的**静态参数 (Static Parameters)**。

* **动力学状态 $x_t \in \mathbb{R}^8$:**
  * 位置 $p_t = [p_x, p_y, p_z]^T$
  * 线速度 $v_t = [v_x, v_y, v_z]^T$
  * 偏航角 (Yaw) $\psi_t$ (以 $[\cos\psi, \sin\psi]$ 形式表示)
  * 角速度 $\omega_t$ (标量，仅 $\omega_z$)
* **静态参数 $\theta \in \mathbb{R}^3$:**
  * 长宽半径 $r_1, r_2$
  * 高度偏差 $z_{off}$ (假设为待估计量，若已知可直接固定)

### 1.2 观测模型

* **输入:** 时间序列 $O_{1:T}$。
* **单帧观测 $O_t$:** 一个大小可变的无序点集 $\{o_{t,1}, ..., o_{t,k}\}$，其中 $k \in \{0, 1, 2\}$。
* **观测特征:** 每个点包含 $[x_{obs}, y_{obs}, z_{obs}, \alpha, \beta]$ (位置+角度)。

---

## 2. 整体架构概览

模型由四个核心模块组成：

1. **Set-Embedding Module:** 处理无序、变长的点云输入。
2. **Temporal Transformer Backbone:** 捕捉系统动力学与时序依赖。
3. **Decoupled Estimation Heads:** 分别回归动力学状态与静态形状。
4. **Physics-Informed Projection (Loss):** 利用几何约束实现隐式数据关联。

---

## 3. 详细模块设计

### 3.1 模块一：集合嵌入层 (Set-Embedding Module)

**目标：** 将每一帧 $t$ 的观测集合 $O_t$ 压缩为固定维度的特征向量 $E_t$。

* **输入张量:** `(Batch, Time, Max_Points=2, Feat_Dim=5)`。不足的点用 0 填充，同时生成 `Mask`。
* **Point-MLP (共享权重):**
  * 对每个观测点独立进行特征提取。
  * 结构: `Linear(5->64) -> BN -> ReLU -> Linear(64->128)`。
* **置换不变聚合 (Permutation Invariant Aggregation):**
  * 使用 **Max Pooling** 沿 `Max_Points` 维度聚合。这保证了无论输入的两个点谁在前谁在后，输出特征一致。
  * *特殊处理:* 如果某帧没有观测点 (k=0)，Pooling 结果设为全 0 向量或可学习的 `Empty_Token`。
* **输出:** 观测特征序列 $Z_{seq}$，形状为 `(Batch, Time, 128)`。

### 3.2 模块二：时序骨干 (Temporal Backbone)

**目标：** 模拟滤波过程，融合历史信息，平滑噪声。

* **位置编码 (Positional Encoding):**
  * 由于物理运动对时间间隔敏感，必须叠加正弦位置编码 `PE`。
  * $Z'_{seq} = Z_{seq} + \text{PE}(Time)$
* **Transformer Encoder:**
  * **层数:** 3-4 层 (不需要太深，避免过拟合)。
  * **多头注意力:** 4 Heads, Hidden Dim = 128。
  * **Causal Mask (因果掩码):** **关键！** 在计算 $t$ 时刻特征时，Mask 掉 $t+1$ 及以后的数据。这确保模型可以实时推理（Online Inference），不仅仅是离线平滑。
* **输出:** 隐状态序列 $H_{seq}$，形状 `(Batch, Time, 128)`。

### 3.3 模块三：解耦估计头 (Decoupled Estimation Heads)

**目标：** 从隐状态解码物理量。我们将输出分为“瞬时流”和“全局流”。

#### A. 动力学头 (Dynamic Head) - 瞬时流

输入为当前时刻的隐状态 $h_t$。

* **结构:** MLP `(128 -> 64 -> Output_Dim)`。
* **输出分支:**
  1. **平移分支:** 输出 $\hat{p}_t, \hat{v}_t$ (6维)。
  2. **旋转分支:** 输出 $\hat{c}_t, \hat{s}_t$ (2维，代表 $\cos, \sin$) 和 $\hat{\omega}_t$ (1维)。
     * *后处理:* 对 $[\hat{c}_t, \hat{s}_t]$ 进行 L2 归一化，确保在单位圆上。

#### B. 静态形状头 (Static Shape Head) - 全局流

输入为**整个时间窗口**隐状态的平均值 $\bar{h} = \text{GlobalAvgPool}(H_{seq})$。
*理由：形状参数不随时间变化，看的时间越久估计越准。*

* **结构:** MLP `(128 -> 64 -> 3)`。
* **输出:** $\hat{r}_1, \hat{r}_2, \hat{z}_{off}$。
* **激活函数:** 使用 `Softplus` 确保半径和偏移量为正数。

---

## 4. 核心：物理几何损失函数 (Physics-Informed Loss)

这是本架构能工作的灵魂。总 Loss 由两部分组成：

$$
\mathcal{L}_{total} = \mathcal{L}_{supervised} + \lambda \cdot \mathcal{L}_{geometry}
$$

### 4.1 监督损失 (Supervised Loss)

如果有 Ground Truth (GT)，直接监督。

* **位置/速度:** MSE Loss。
* **旋转:** $1 - (c_{pred}c_{gt} + s_{pred}s_{gt})$ (余弦相似度损失)。
* **形状:** MSE Loss。

### 4.2 几何一致性损失 (Chamfer Projection Loss)

如果没有 GT 或者为了增强物理约束，使用此 Loss 替代 JPDA。

**步骤 1: 构建预测的体坐标系关键点**
基于网络预测的 $\hat{r}_1, \hat{r}_2, \hat{z}_{off}$，构建 4 个特征点：

$$
\mathcal{P}_{body} = \left\{
\underbrace{\begin{bmatrix} \hat{r}_1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} -\hat{r}_1 \\ 0 \\ 0 \end{bmatrix}}_{\text{Face 1 & 3 (Low)}},
\underbrace{\begin{bmatrix} 0 \\ \hat{r}_2 \\ \hat{z}_{off} \end{bmatrix}, \begin{bmatrix} 0 \\ -\hat{r}_2 \\ \hat{z}_{off} \end{bmatrix}}_{\text{Face 2 & 4 (High)}}
\right\}
$$

**步骤 2: 投影到世界坐标系**
利用预测的状态 $\hat{p}_t$ 和偏航角 $\psi_t$ (由 cos/sin 构建旋转矩阵 $R_z$)：

$$
\mathcal{P}_{pred}^t = \{ \hat{p}_t + R_z(\hat{\psi}_t) \cdot \mathbf{v} \mid \mathbf{v} \in \mathcal{P}_{body} \}
$$

**步骤 3: 计算倒角距离 (Chamfer Distance)**
计算当前帧观测点集 $O_t$ 与预测点集 $\mathcal{P}_{pred}^t$ 的距离。

$$
\mathcal{L}_{chamfer}^t = \sum_{x \in O_t} \min_{y \in \mathcal{P}_{pred}^t} ||x - y||_2^2
$$

**机制解析:**

* 网络为了降低这个 Loss，必须让预测出的长方体表面贴合观测点。
* 由于 $\hat{z}_{off}$ 的存在，如果观测点很高，只有匹配到 Face 2/4 才能最小化距离。**网络会自动学会利用高度差来区分观测对应哪个面，无需人工关联。**

---

## 5. 训练与实施细节 (Recipe for Success)

### 5.1 数据标准化 (非常重要)

* **输入归一化:** 不要直接把世界坐标 (如 x=1000m) 喂给网络。
  * **相对坐标策略:** 将输入观测减去序列第一帧的观测中心：$O'_t = O_t - \text{Mean}(O_0)$。让网络预测相对于起点的累积运动。
  * 或者直接预测**帧间增量** (Delta prediction)，然后累加。

### 5.2 训练策略 (Curriculum Learning)

1. **阶段一 (Warm-up):**
   * 假设 $z_{off}$ 和 $r$ 已知（固定住 Static Head），只训练动力学状态。
   * 使用强监督 Loss。
2. **阶段二 (Joint Training):**
   * 放开 $z_{off}$ 和 $r$ 的梯度。
   * 加入 Chamfer Loss。
3. **阶段三 (Hard Cases):**
   * 在训练数据中随机 Mask 掉更多点（模拟严重遮挡），强迫 Transformer 利用历史惯性进行预测。

### 5.3 推理 (Inference)

* 由于使用了 Causal Mask，推理时你只需要维护一个固定长度的滑动窗口 (Context Window, e.g., T=50)。
* 每来一帧新数据，将其 Append 进窗口，丢弃最旧的一帧，跑一次前向传播即可得到当前最优估计。

---

## 6. 伪代码实现 (PyTorch Style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysSetTransformer(nn.Module):
    def __init__(self, dim_model=128):
        super().__init__()
      
        # 1. Set Embedding
        self.point_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, dim_model)
        )
      
        # 2. Temporal Backbone
        self.pos_encoder = PositionalEncoding(dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
      
        # 3. Heads
        # Dynamic: p(3) + v(3) + rot(2) + omega(1) = 9
        self.dynamic_head = nn.Sequential(
            nn.Linear(dim_model, 64), nn.ReLU(), nn.Linear(64, 9)
        )
        # Static: r1, r2, z_off = 3
        self.static_head = nn.Sequential(
            nn.Linear(dim_model, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softplus()
        )

    def forward(self, x, mask=None):
        # x: [Batch, Time, 2, 5]
        # mask: [Batch, Time, 2] (True if valid point)
      
        B, T, N, D = x.shape
      
        # --- 1. Set Embedding ---
        # Flatten points: [B*T*N, 5] -> [B*T*N, 128]
        point_feats = self.point_mlp(x.view(-1, D))
        point_feats = point_feats.view(B, T, N, -1)
      
        # Masking invalid points (set feat to -inf for max pooling)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1) # [B, T, N, 1]
            point_feats = point_feats.masked_fill(~mask_expanded, -1e9)
          
        # Max Pooling over set: [B, T, 128]
        seq_feats, _ = torch.max(point_feats, dim=2)
        # Handle cases where no points observed (all masked) -> return 0
        seq_feats = torch.nan_to_num(seq_feats, nan=0.0, neginf=0.0)

        # --- 2. Transformer ---
        seq_feats = self.pos_encoder(seq_feats)
        # Generate causal mask for time
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        output_seq = self.transformer(seq_feats, mask=causal_mask, is_causal=True)
      
        # --- 3. Dynamic Output (Last Token) ---
        last_state = output_seq[:, -1, :] # [B, 128]
        dyn_out = self.dynamic_head(last_state)
      
        pos = dyn_out[:, 0:3]
        vel = dyn_out[:, 3:6]
        rot_vec = F.normalize(dyn_out[:, 6:8], p=2, dim=1) # cos, sin
        omega = dyn_out[:, 8:9]
      
        # --- 4. Static Output (Global Average) ---
        avg_state = torch.mean(output_seq, dim=1) # [B, 128]
        static_params = self.static_head(avg_state) # [B, 3] (r1, r2, z_off)
      
        return {
            "pos": pos, "vel": vel, "rot": rot_vec, "omega": omega,
            "static": static_params
        }
```

这份设计不仅理论自洽，而且充分考虑了你描述的“垂直旋转”和“Z轴偏差”特性，是一个可以立即着手实验的高水平架构。
