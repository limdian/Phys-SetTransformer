import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CuboidSystemSimulator(Dataset):
    def __init__(self, 
                 num_samples=1000, 
                 seq_len=50, 
                 dt=0.01,
                 noise_level=0.01):
        """
        Args:
            num_samples: 虚拟数据集的大小 (用于设定一个epoch的长度)
            seq_len: 时间序列长度 T
            dt: 采样时间间隔
            noise_level: 观测噪声的标准差 (米)
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dt = dt
        self.noise_level = noise_level
        
        # 物理参数范围限制
        self.r_range = (0.1, 0.5)      # 半径范围
        self.z_off_range = (0.0, 0.2)  # 高度偏差范围
        
        # 运动范围 (相机前方)
        self.pos_x_range = (2.0, 10.0) # 距离相机 2-10米
        self.pos_y_range = (-5.0, 5.0) # 左右 5米
        self.pos_z_range = (-1.0, 1.0) # 上下 1米

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        生成一条完整的轨迹数据
        """
        # --- 1. 随机采样静态物理参数 ---
        r1 = np.random.uniform(*self.r_range)
        r2 = np.random.uniform(*self.r_range)
        z_off = np.random.uniform(*self.z_off_range)
        
        # 定义体坐标系下的4个关键点 (Body Frame)
        # P0: Front, P1: Left, P2: Back, P3: Right
        # 偶数点(0,2)高度为0，奇数点(1,3)高度为z_off
        body_points = np.array([
            [r1, 0, 0],          # P0
            [0, r2, z_off],      # P1 (Offset)
            [-r1, 0, 0],         # P2
            [0, -r2, z_off]      # P3 (Offset)
        ]) # Shape: (4, 3)

        # 定义每个面的法向量 (用于判断可见性)
        # 简化：点的位置向量即为法向量方向 (除了z轴)
        normals = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]
        ])

        # --- 2. 生成平滑轨迹 (Ground Truth) ---
        # 使用随机加速度生成平滑的位置和偏航角
        
        # 初始状态
        pos = np.zeros((self.seq_len, 3))
        vel = np.zeros((self.seq_len, 3))
        yaw = np.zeros(self.seq_len)
        omega = np.zeros(self.seq_len)
        
        # 随机初始化位置和速度
        curr_pos = np.array([
            np.random.uniform(*self.pos_x_range),
            np.random.uniform(*self.pos_y_range),
            np.random.uniform(*self.pos_z_range)
        ])
        curr_vel = np.random.randn(3) * 0.5
        curr_yaw = np.random.uniform(-np.pi, np.pi)
        curr_omega = np.random.randn() * 0.5 # 初始角速度

        for t in range(self.seq_len):
            # 记录状态
            pos[t] = curr_pos
            vel[t] = curr_vel
            yaw[t] = curr_yaw
            omega[t] = curr_omega
            
            # 动力学积分 (含随机扰动模拟受力)
            acc_pos = np.random.randn(3) * 0.1  # 随机线加速度
            acc_omega = np.random.randn() * 0.1 # 随机角加速度
            
            # 简单的欧拉积分
            curr_pos += curr_vel * self.dt
            curr_vel += acc_pos * self.dt
            curr_yaw += curr_omega * self.dt
            curr_omega += acc_omega * self.dt
            
            # 简单的边界反弹 (防止飞出相机视野)
            if not (self.pos_x_range[0] < curr_pos[0] < self.pos_x_range[1]): curr_vel[0] *= -1
            if not (self.pos_y_range[0] < curr_pos[1] < self.pos_y_range[1]): curr_vel[1] *= -1
            if not (self.pos_z_range[0] < curr_pos[2] < self.pos_z_range[1]): curr_vel[2] *= -1

        # --- 3. 生成观测数据 (Observation) ---
        # 格式: [Batch, Time, Max_Points=2, Feats=5]
        # Feats: [x, y, z, azimuth, elevation]
        
        observations = np.zeros((self.seq_len, 2, 5), dtype=np.float32)
        valid_mask = np.zeros((self.seq_len, 2), dtype=bool) # 标记是否真的有观测点
        
        for t in range(self.seq_len):
            # 构建旋转矩阵 R_z (只绕Z轴)
            cy, sy = np.cos(yaw[t]), np.sin(yaw[t])
            R = np.array([
                [cy, -sy, 0],
                [sy,  cy, 0],
                [0,   0,  1]
            ])
            
            # 将4个点变换到世界坐标系 (相机系)
            # P_world = P_center + R * P_body
            p_world_all = pos[t] + (R @ body_points.T).T # Shape (4, 3)
            
            # 计算法向量在世界系的方向
            normals_world = (R @ normals.T).T
            
            # 判断可见性: 视线向量 v_view = P_center - Camera(0,0,0)
            # 简化判断：面法向量指向相机为可见 => normal dot (-p_world) > 0
            # 这里取物体中心向量取反
            v_view = -pos[t] 
            v_view /= np.linalg.norm(v_view)
            
            visibility_scores = np.sum(normals_world * v_view, axis=1) # Dot product
            visible_indices = np.where(visibility_scores > 0)[0] # 找到可见面的索引
            
            # 模拟观测 (选取最多2个可见点)
            # 打乱顺序，模拟“无序性”，这是为了训练Set-Transformer
            np.random.shuffle(visible_indices)
            
            count = 0
            for idx in visible_indices[:2]: # 最多取2个
                # 获取真实坐标
                p_true = p_world_all[idx]
                
                # 添加高斯噪声
                noise = np.random.randn(3) * self.noise_level
                p_obs = p_true + noise
                
                # 计算角度 (Azimuth, Elevation)
                # Azimuth: atan2(y, x), Elevation: atan2(z, sqrt(x^2+y^2))
                # 注意: 这是基于X轴朝前的定义
                x, y, z = p_obs
                azi = np.arctan2(y, x)
                ele = np.arctan2(z, np.sqrt(x**2 + y**2))
                
                # 填入观测向量 [x, y, z, azi, ele]
                observations[t, count] = np.array([x, y, z, azi, ele])
                valid_mask[t, count] = 1
                count += 1

        # --- 4. 整理输出数据 ---
        # 状态 Ground Truth: [p(3), v(3), cos, sin, omega, r1, r2, z_off]
        # 注意: 这里把静态参数也拼接到每个时刻，方便某些监督训练，或者只在static head监督
        state_gt = np.concatenate([
            pos,                            # [T, 3]
            vel,                            # [T, 3]
            np.cos(yaw)[:, None],           # [T, 1]
            np.sin(yaw)[:, None],           # [T, 1]
            omega[:, None]                  # [T, 1]
        ], axis=1) # Total dynamic dim: 9
        
        static_gt = np.array([r1, r2, z_off]) # [3]

        # 转为 Tensor
        return {
            'obs': torch.from_numpy(observations).float(),    # [T, 2, 5]
            'mask': torch.from_numpy(valid_mask).bool(),      # [T, 2]
            'state_gt': torch.from_numpy(state_gt).float(),   # [T, 9]
            'static_gt': torch.from_numpy(static_gt).float()  # [3]
        }

if __name__ == "__main__":
    # 实例化数据集
    sim_dataset = CuboidSystemSimulator(num_samples=1000, seq_len=50)
    
    # 实例化 DataLoader
    train_loader = DataLoader(sim_dataset, batch_size=32, shuffle=True)
    
    # 获取一个 Batch
    batch = next(iter(train_loader))
    
    print("--- 维度检查 ---")
    print(f"Observation Shape: {batch['obs'].shape}")       # [32, 50, 2, 5]
    print(f"Mask Shape:        {batch['mask'].shape}")      # [32, 50, 2]
    print(f"Dynamic GT Shape:  {batch['state_gt'].shape}")  # [32, 50, 9]
    print(f"Static GT Shape:   {batch['static_gt'].shape}") # [32, 3]
    
    print("\n--- 数据一致性检查 ---")
    # 检查 z_off 是否真的在观测中体现
    # 如果观测点高度 z 很大，应该对应 mask=True
    sample_idx = 0
    t_idx = 0
    obs = batch['obs'][sample_idx, t_idx]
    mask = batch['mask'][sample_idx, t_idx]
    z_off_gt = batch['static_gt'][sample_idx, 2]
    print(f"GT z_off: {z_off_gt:.4f}")
    print(f"Obs points (z-coord): {obs[:, 2]}")
    print(f"Valid Mask: {mask}")