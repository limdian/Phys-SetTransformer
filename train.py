import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 导入之前的模块
from SimDataset import CuboidSystemSimulator
from module import PhysSetTransformer

# --- 配置 ---
CONFIG = {
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 300,         # 快速验证，30轮应该能看到收敛
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seq_len': 50,
    'lambda_chamfer': 1.0, # 物理Loss的权重
    'lambda_static': 10.0   # 静态参数通常较难收敛，给大一点权重
}

class PhysicsLoss(nn.Module):
    """
    包含监督Loss和物理Chamfer Loss的综合损失函数
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def construct_body_points(self, r1, r2, z_off):
        """
        根据预测的静态参数构建体坐标系下的4个关键点
        r1, r2, z_off: [B, 1] (或者广播后的形状)
        Return: [B, 4, 3]
        """
        # P0, P2 (Face 1 & 3): z=0
        # P1, P3 (Face 2 & 4): z=z_off
        
        zeros = torch.zeros_like(r1)
        
        # Point 0: [r1, 0, 0]
        p0 = torch.stack([r1, zeros, zeros], dim=-1)
        # Point 1: [0, r2, z_off]
        p1 = torch.stack([zeros, r2, z_off], dim=-1)
        # Point 2: [-r1, 0, 0]
        p2 = torch.stack([-r1, zeros, zeros], dim=-1)
        # Point 3: [0, -r2, z_off]
        p3 = torch.stack([zeros, -r2, z_off], dim=-1)
        
        return torch.stack([p0, p1, p2, p3], dim=1) # [B, 4, 3]

    def forward(self, preds, batch, mean_offset):
        """
        preds: 模型输出字典
        batch: 数据集字典
        mean_offset: [B, 3] 用于还原世界坐标的偏移量
        """
        # --- 1. 数据准备 ---
        # 提取 Ground Truth
        gt_pos = batch['state_gt'][..., 0:3].to(preds['pos'].device)
        gt_vel = batch['state_gt'][..., 3:6].to(preds['pos'].device)
        gt_rot = torch.cat([batch['state_gt'][..., 6:7], batch['state_gt'][..., 7:8]], dim=-1).to(preds['pos'].device) # cos, sin
        gt_omega = batch['state_gt'][..., 8:9].to(preds['pos'].device)
        gt_static = batch['static_gt'].to(preds['pos'].device) # [B, 3]

        # 提取预测值
        pred_pos = preds['pos']
        pred_vel = preds['vel']
        pred_rot = preds['rot']
        pred_omega = preds['omega']
        pred_static = preds['static']

        # --- 2. 监督损失 (Supervised Loss) ---
        target_pos_normalized = gt_pos - mean_offset.unsqueeze(1)
        
        l_pos = self.mse(pred_pos, target_pos_normalized).mean()
        l_vel = self.mse(pred_vel, gt_vel).mean()
        l_omega = self.mse(pred_omega, gt_omega).mean()
        
        cos_sim = torch.sum(pred_rot * gt_rot, dim=-1)
        l_rot = (1.0 - cos_sim).mean()
        
        l_static = self.mse(pred_static, gt_static).mean()

        # --- 3. 物理几何损失 (Chamfer Loss) ---
        
        # A. 还原预测的世界坐标点云
        pred_pos_world = pred_pos + mean_offset.unsqueeze(1) # [B, T, 3]
        
        # 构建体坐标系点 [B, 4, 3]
        body_pts = self.construct_body_points(
            pred_static[:, 0], pred_static[:, 1], pred_static[:, 2]
        )
        
        # 扩展时间维度: [B, 1, 4, 3] -> [B, T, 4, 3]
        body_pts = body_pts.unsqueeze(1).expand(-1, preds['pos'].shape[1], -1, -1)
        
        # 构建旋转矩阵 Rz [B, T, 3, 3]
        c, s = pred_rot[..., 0], pred_rot[..., 1]
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        Rz = torch.stack([
            torch.stack([c, -s, zeros], dim=-1),
            torch.stack([s, c, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2) # [B, T, 3, 3]
        
        # 目标: 将 [B, T, 3, 3] 的矩阵应用到 [B, T, 4, 3] 的点上
        # 1. 将 Rz 扩展为 [B, T, 1, 3, 3] 以便广播
        Rz_expanded = Rz.unsqueeze(2) 
        
        # 2. 将点变为列向量 [B, T, 4, 3, 1]
        body_pts_expanded = body_pts.unsqueeze(-1)
        
        # 3. 矩阵乘法
        # [B, T, 1, 3, 3] @ [B, T, 4, 3, 1] -> [B, T, 4, 3, 1]
        rotated_body = torch.matmul(Rz_expanded, body_pts_expanded).squeeze(-1) # [B, T, 4, 3]

        # 预测表面点 = 中心位置 + 旋转后的体坐标点
        # [B, T, 1, 3] + [B, T, 4, 3] -> [B, T, 4, 3]
        pred_surface_pts = pred_pos_world.unsqueeze(2) + rotated_body
        
        # B. 计算与观测点的距离 (不变)
        obs = batch['obs'].to(preds['pos'].device)
        obs_xyz = obs[..., :3]
        mask = batch['mask'].to(preds['pos'].device)
        
        dist_mat = torch.norm(
            obs_xyz.unsqueeze(3) - pred_surface_pts.unsqueeze(2), 
            dim=-1
        )
        
        min_dist, _ = torch.min(dist_mat, dim=-1)
        l_chamfer = (min_dist * mask.float()).sum() / (mask.sum() + 1e-6)

        return {
            'loss': l_pos + l_vel + l_rot + l_omega + CONFIG['lambda_static']*l_static + CONFIG['lambda_chamfer']*l_chamfer,
            'l_pos': l_pos.item(),
            'l_rot': l_rot.item(),
            'l_static': l_static.item(),
            'l_chamfer': l_chamfer.item()
        }

def train():
    print(f"Running on {CONFIG['device']}")
    
    # 1. Dataset & Model
    dataset = CuboidSystemSimulator(num_samples=2000, seq_len=CONFIG['seq_len'])
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    
    model = PhysSetTransformer(input_dim=5).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    loss_fn = PhysicsLoss()
    
    # 2. Training Loop
    model.train()
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        metrics = {'l_pos': 0, 'l_rot': 0, 'l_static': 0, 'l_chamfer': 0}
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch in pbar:
            optimizer.zero_grad()
            
            # --- 输入归一化 (Input Normalization) ---
            obs = batch['obs'].to(CONFIG['device']) # [B, T, 2, 5]
            mask = batch['mask'].to(CONFIG['device'])
            
            # 计算 Batch 中每个样本第0帧的可见点中心作为 Offset
            # 简单起见，取第0帧第一个点的坐标，如果有mask则取0 (batch内近似)
            # 更严谨的做法是求 masked mean
            valid_points_0 = obs[:, 0, :, :3] * mask[:, 0, :].unsqueeze(-1)
            sum_points = valid_points_0.sum(dim=1) 
            count_points = mask[:, 0, :].sum(dim=1).unsqueeze(-1).clamp(min=1.0)
            initial_mean = sum_points / count_points # [B, 3] (Offset)
            
            # 观测去中心化 (仅位置 x,y,z)
            obs_normalized = obs.clone()
            obs_normalized[..., :3] = obs[..., :3] - initial_mean.view(-1, 1, 1, 3)
            
            # --- Forward ---
            preds = model(obs_normalized, mask)
            
            # --- Loss Calculation ---
            # 注意：Loss内部需要 initial_mean 来还原世界坐标做 Chamfer 计算
            loss_dict = loss_fn(preds, batch, initial_mean)
            
            loss = loss_dict['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止爆炸
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            for k in metrics:
                metrics[k] += loss_dict[k]
                
            pbar.set_postfix({
                'L_Pos': f"{loss_dict['l_pos']:.3f}", 
                'L_Stat': f"{loss_dict['l_static']:.3f}",
                'L_Cham': f"{loss_dict['l_chamfer']:.3f}"
            })
            
        # End of Epoch
        avg_loss = epoch_loss / len(loader)
        avg_metrics = {k: v/len(loader) for k, v in metrics.items()}
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        print(f"  >> Static Err: {avg_metrics['l_static']:.4f} | Chamfer: {avg_metrics['l_chamfer']:.4f}")

    # 3. Simple Evaluation (Check one sample)
    model.eval()
    with torch.no_grad():
        print("\n--- Final Validation (Single Sample) ---")
        # Reuse last batch
        gt_r1 = batch['static_gt'][0, 0].item()
        gt_r2 = batch['static_gt'][0, 1].item()
        gt_zoff = batch['static_gt'][0, 2].item()
        
        pred_r1 = preds['static'][0, 0].item()
        pred_r2 = preds['static'][0, 1].item()
        pred_zoff = preds['static'][0, 2].item()
        
        print(f"GT  Params: r1={gt_r1:.3f}, r2={gt_r2:.3f}, z_off={gt_zoff:.3f}")
        print(f"Pred Params: r1={pred_r1:.3f}, r2={pred_r2:.3f}, z_off={pred_zoff:.3f}")
        print(f"Error: r1={abs(gt_r1-pred_r1):.3f}, z_off={abs(gt_zoff-pred_zoff):.3f}")

if __name__ == "__main__":
    train()