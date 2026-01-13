import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SetAttentionEncoder(nn.Module):
    """
    专门处理无序点集的编码器。
    使用 Learnable Query 进行 Attention Aggregation。
    """
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 1. Point-wise 特征提取 (类似于 PointNet 的前两层)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # 2. Aggregation Query (可学习的聚合向量)
        # 类似于 "我在寻找什么样的特征组合"
        self.agg_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. Multi-head Attention 用于聚合
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        
        # 4. 聚合后的 FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x: [Batch, Max_Points, Feat_Dim]
        mask: [Batch, Max_Points] (True表示是有效点, False表示填充点)
        """
        B, N, D = x.shape
        
        # --- Step 1: 提取每个点的独立特征 ---
        # point_feats: [B, N, d_model]
        point_feats = self.point_mlp(x)
        
        # --- Step 2: Attention Aggregation ---
        # Query: [B, 1, d_model] (广播)
        query = self.agg_query.repeat(B, 1, 1)
        
        # Key/Value: [B, N, d_model]
        key = value = point_feats
        
        # 处理 Mask: PyTorch MHA 需要 key_padding_mask (True表示要忽略/Padding)
        # 我们的 mask 定义通常是 True 表示有效，所以取反
        key_padding_mask = ~mask if mask is not None else None
        
        # Attention
        # attn_out: [B, 1, d_model]
        attn_out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        
        # Residual + Norm + FFN
        out = self.norm(attn_out + query) # 残差连接
        out = out + self.ffn(out)
        
        # Squeeze: [B, 1, d_model] -> [B, d_model]
        return out.squeeze(1)


class PhysSetTransformer(nn.Module):
    def __init__(self, 
                 input_dim=5, 
                 d_model=128, 
                 num_layers=3, 
                 nhead=4):
        super().__init__()
        
        # --- 1. Set Embedding (Observation Encoder) ---
        # 处理每一帧的无序点集
        self.set_encoder = SetAttentionEncoder(input_dim, d_model)
        
        # --- 2. Temporal Backbone (Dynamics Learner) ---
        self.d_model = d_model
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. Decoupled Heads ---
        
        # A. 动力学状态头 (Dynamic Head)
        # 输出: [px, py, pz, vx, vy, vz, cos, sin, omega] -> 9 dims
        self.dynamic_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 9) 
        )
        
        # B. 静态形状头 (Static Head)
        # 输出: [r1, r2, z_off] -> 3 dims
        self.static_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, obs, mask=None):
        """
        obs: [Batch, Seq_Len, Max_Points, 5]
        mask: [Batch, Seq_Len, Max_Points] (True if valid)
        """
        B, T, N, D = obs.shape
        
        # --- 1. Frame-wise Set Embedding ---
        # 将 Batch 和 Time 维度合并，当作一堆独立的帧来处理
        # Input: [B*T, N, 5]
        obs_flat = obs.view(B * T, N, D)
        mask_flat = mask.view(B * T, N) if mask is not None else None
        
        # Encoding
        # Output: [B*T, d_model]
        frame_feats = self.set_encoder(obs_flat, mask_flat)
        
        # 处理全丢帧 (如果在某一帧里 mask全是False, attention输出可能是NaN)
        # 简单的做法是把 NaN 替换为 0 (表示这一帧没有信息)
        frame_feats = torch.nan_to_num(frame_feats, nan=0.0)
        
        # 恢复时序维度: [B, T, d_model]
        seq_feats = frame_feats.view(B, T, self.d_model)
        
        # --- 2. Temporal Processing ---
        # 添加位置编码
        seq_feats = self.pos_encoder(seq_feats)
        
        # 生成因果掩码 (Causal Mask) - 确保 t 时刻看不到 t+1
        # mask shape: [T, T], 上三角为 -inf
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(obs.device)
        
        # Transformer Forward
        # Output: [B, T, d_model]
        hidden_states = self.transformer(seq_feats, mask=causal_mask, is_causal=True)
        
        # --- 3. Output Decoding ---
        
        # 分支一：动力学 (Dynamic) - 每个时刻都有输出
        # Input: [B, T, d_model] -> Output: [B, T, 9]
        dynamic_out = self.dynamic_head(hidden_states)
        
        # 拆解输出并施加约束
        pos = dynamic_out[..., 0:3]
        vel = dynamic_out[..., 3:6]
        # 旋转: 归一化 cos, sin
        raw_rot = dynamic_out[..., 6:8]
        rot_vec = F.normalize(raw_rot, p=2, dim=-1) # [B, T, 2]
        omega = dynamic_out[..., 8:9]
        
        # 分支二：静态参数 (Static) - 全局池化后输出
        # 我们假设静态参数可以看整个历史来估计，所以用 Global Average Pooling
        # Input: [B, d_model] (Avg over T) -> Output: [B, 3]
        global_feat = torch.mean(hidden_states, dim=1) 
        static_out = self.static_head(global_feat)
        
        # 施加物理约束: 半径和偏移必须 > 0
        # 使用 Softplus + eps 保证数值稳定性
        static_params = F.softplus(static_out) + 1e-4 # [B, 3] (r1, r2, z_off)
        
        return {
            "pos": pos,          # [B, T, 3]
            "vel": vel,          # [B, T, 3]
            "rot": rot_vec,      # [B, T, 2] (cos, sin)
            "omega": omega,      # [B, T, 1]
            "static": static_params # [B, 3] (r1, r2, z_off)
        }

class PositionalEncoding(nn.Module):
    """标准的正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        return x + self.pe[:, :x.size(1), :]

# --- 简单的测试代码 ---
if __name__ == "__main__":
    # 模拟数据
    B, T, N, D = 32, 50, 2, 5
    model = PhysSetTransformer(input_dim=D)
    
    # 模拟输入
    dummy_obs = torch.randn(B, T, N, D)
    # 模拟 Mask: 假设第二个点有一半概率是无效的 (False)
    dummy_mask = torch.ones(B, T, N, dtype=torch.bool)
    dummy_mask[:, :, 1] = torch.rand(B, T) > 0.5 
    
    # 前向传播
    out = model(dummy_obs, dummy_mask)
    
    print("--- Model Output Shapes ---")
    print(f"Pos: {out['pos'].shape}")       # [32, 50, 3]
    print(f"Rot: {out['rot'].shape}")       # [32, 50, 2]
    print(f"Static: {out['static'].shape}") # [32, 3]
    
    # 检查归一化约束
    rot_norm = torch.norm(out['rot'], dim=-1)
    print(f"Rotation Norm (should be 1.0): {rot_norm.mean().item():.4f}")
    
    # 检查正数约束
    print(f"Min Static Param (should be > 0): {out['static'].min().item():.4f}")