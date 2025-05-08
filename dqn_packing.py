import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

##########################################
# 环境部分：PackingEnv
##########################################
class PackingEnv:
    def __init__(self, current_layer_nodes, query_workload, current_layer_level):

        if not current_layer_nodes:
            raise ValueError("current_layer_nodes cannot be empty")
        self.current_layer = current_layer_nodes  # 当前层待打包节点
        self.query_workload = query_workload
        self.m = len(query_workload)
        self.N = len(current_layer_nodes)  # 下层节点数量，决定预创建上层节点的数量
        self.current_layer_level = current_layer_level #当前层号

        # 预先创建 N 个空的上层节点（固定数量），非叶节点采用 bitmap 存储关键词信息
        self.upper_layer = []
        for i in range(self.N):
            # 每个上层节点赋予临时 id（层号后续在主流程中更新）
            node = {'id': i, 'layer': self.current_layer_level+1, 'labels': [], 'children': [], 'MBR': None}
            self.upper_layer.append(node)

        self.current_step = 0
        if self.current_layer:
            self.current_node = self.current_layer[self.current_step]
        else:
            self.current_node = None
        # 添加累积奖励跟踪
        self.total_reward = 0.0

    def _mbr_intersect(self, mbr, query):
        """
        判断 mbr（字典形式，包含 'min_lat', 'max_lat', 'min_lon', 'max_lon'）是否与 query 的区域相交。
        这里 query 包含一个 'area' 键，其值为包含边界信息的字典。
        """
        area = query['area']
        return not (
                mbr['max_lat'] < area['min_lat'] or
                mbr['min_lat'] > area['max_lat'] or
                mbr['max_lon'] < area['min_lon'] or
                mbr['min_lon'] > area['max_lon']
        )

    def _get_state(self):
        """
        生成状态向量，总维度为 ((m+1)*N + m)：
          - 对于每个预先创建的上层节点（固定 N 个），前 m 个维度表示该节点对各查询的匹配情况，
            匹配条件为：若该节点的标签中包含查询的任一关键词且其 MBR 与查询区域相交，则置为1，否则为0。
          - 紧跟其后的一维表示该节点已连接的底层节点数。
          - 最后追加 m 个维度用于表示下一底层节点的信息（因为当前底层节点排列在列表中）。
        """
        state_dim = (self.m + 1) * self.N + self.m
        state = np.zeros(state_dim)

        # 对每个上层节点，填充关键词匹配（考虑 mbr 是否相交且至少有一个关键词在query中）和孩子节点数量
        for i, node in enumerate(self.upper_layer):
            # 跳过空节点但保留位置（论文要求固定N个节点）
            if node['MBR'] is not None:
                for j, query in enumerate(self.query_workload):
                    # 如果该上层节点标签中包含 query 中任一关键词，并且其 MBR存在且与 query 区域相交，则匹配置1
                    spatial_intersect = self._mbr_intersect(node['MBR'], query)
                    keyword_intersect = any(kw in node['labels'] for kw in query['keywords'])
                    state[i * (self.m + 1) + j] = 1 if (spatial_intersect and keyword_intersect) else 0
                # 孩子节点数量
                state[i * (self.m + 1) + self.m] = len(node['children'])#最底层是cluster得来的，children有很多，不能设为0

        # 保存下一底层节点的信息
        if self.current_step + 1 < len(self.current_layer):
            next_node = self.current_layer[self.current_step + 1]
            for j, query in enumerate(self.query_workload):
                next_node_spatial = self._mbr_intersect(next_node['MBR'], query) if next_node['MBR'] else False
                next_node_keyword = any(kw in next_node['labels'] for kw in query['keywords'])
                state[self.N * (self.m + 1) + j] = 1 if (next_node_spatial and next_node_keyword) else 0

        else:
            # 如果没有下一底层节点，则后面的 m 维置为0
            for j in range(self.m):
                state[self.N * (self.m + 1) + j] = 0
        # # 添加当前层级作为一个整数特征
        # state = np.append(state, self.current_layer_level)
        return state

    def get_action_mask(self):
        mask = [False] * self.N
        # 找到所有空节点的索引（即 children 为空的节点）
        empty_indices = [i for i, node in enumerate(self.upper_layer) if node['MBR'] is None]
        if empty_indices:
            # 仅允许第一个空节点
            mask[empty_indices[0]] = True
            # 对于非空节点，允许选择
            for i, node in enumerate(self.upper_layer):
                if node['MBR'] is not None:
                    mask[i] = True
        else:
            mask = [True] * self.N
        return mask

    def get_valid_actions(self):
        mask = self.get_action_mask()
        """直接返回布尔掩码列表（而非索引列表）"""
        return mask

    def step(self, action):
        done = False
        prev_access = self._calculate_access_cost()
        # 之前会出现节点增加的原因；下层的空节点也会参与打包
        # 跳过下层中 MBR 为空的节点（空节点不打包）
        while self.current_node is not None and self.current_node['MBR'] is None:
            #print(f"Skipping empty lower node at index {self.current_step}")
            self.current_step += 1
            if self.current_step < len(self.current_layer):
                self.current_node = self.current_layer[self.current_step]
            else:
                done = True
                return None, 0, done

        # 将当前底层节点合并到选定的上层节点
        target = self.upper_layer[action]
        current_labels = list(self.current_node['labels'])
        target['labels'] = list(set(target['labels'] + current_labels))
        # 只存储下层节点的 index，而非整个节点内容
        #target['children'].append(self.current_step)
        # 只存储节点ID
        target['children'].append(self.current_node['id'])
        # 如果当前底层节点非空，更新目标节点的 MBR
        if self.current_node['MBR'] is not None:
            if target['MBR'] is None:
                target['MBR'] = self.current_node['MBR']
            else:
                target['MBR'] = self._merge_mbr(target['MBR'], self.current_node['MBR'])

        self.current_step += 1
        if self.current_step >= len(self.current_layer):
            done = True
            next_state = None
        else:
            self.current_node = self.current_layer[self.current_step]
            next_state = self._get_state()

        new_access = self._calculate_access_cost()
        # 采用平均每个查询节点访问数减少作为 reward

        raw_reward = prev_access - new_access

        # # 归一化相对改进
        # if prev_access > 0:
        #     relative_improvement = (prev_access - new_access) / prev_access
        #     base_reward = relative_improvement * 10.0  # 缩放因子
        # else:
        #     base_reward = prev_access - new_access
        #以下版本
        # # 对奖励进行归一化
        # normalized_reward = raw_reward / self.N
        # # 正则化以防止过度打包
        # children_count = len(target['children'])
        # optimal_children = max(2, self.N // 5)  # 每个节点的目标子节点
        # regularization = -0.2 * ((children_count - optimal_children) / optimal_children) ** 2

        # reward = raw_reward + regularization
        reward = raw_reward

        # 更新累积奖励，用于终止条件判断
        self.total_reward += reward
        # # 应用论文终止条件：如果累积奖励不大于-N，提前终止
        # if self.total_reward <= -self.N:
        #     done = True
        # if  not done and (new_access > self.N):
        #     done = True
        #     print("终止条件触发：当前访问节点数大于初始访问节点数")
        return next_state, reward, done

    def _calculate_access_cost(self):
        """
        计算所有查询的平均节点访问数。
        对于每个查询，我们计算：
        1. 需要访问的非空上层节点数量
        2. 当上层节点与查询匹配时，还需要访问其子节点
        3. 未打包的下层节点直接计入访问成本
        """
        total_access = 0

        for query in self.query_workload:
            query_access = 0

            # 上层节点部分
            for node in self.upper_layer:
                # 跳过空节点
                if node['MBR'] is None:
                    continue

                # 每个非空上层节点都需要访问一次
                query_access += 1  # 基本访问成本

                # 如果节点与查询匹配，还需要访问其子节点
                if self._mbr_intersect(node['MBR'], query) and any(kw in node['labels'] for kw in query['keywords']):
                    # 每个子节点的访问成本
                    query_access += len(node['children'])

            # 下层部分：未打包的节点
            for i in range(self.current_step, len(self.current_layer)):
                if self.current_layer[i]['MBR'] is not None:
                    query_access += 1  # 未打包节点的访问成本


            total_access += query_access

        # 返回平均每个查询的访问成本
        return total_access / len(self.query_workload)


    def _merge_mbr(self, mbr1, mbr2):
        return {
            'min_lat': min(mbr1['min_lat'], mbr2['min_lat']),
            'max_lat': max(mbr1['max_lat'], mbr2['max_lat']),
            'min_lon': min(mbr1['min_lon'], mbr2['min_lon']),
            'max_lon': max(mbr1['max_lon'], mbr2['max_lon'])
        }

    def reset(self):
        self.current_step = 0
        if self.current_layer:
            self.current_node = self.current_layer[self.current_step]
        else:
            self.current_node = None
        # 重置累积奖励
        self.total_reward = 0.0
        return self._get_state()


##########################################
# DQN部分：标准结构，输入状态，输出所有动作的 Q 值
##########################################
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 第二隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 第三隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        q_values = self.net(state)
        # 剪裁Q值以防止极端值
        return torch.clamp(q_values, min=-10.0, max=10.0)


##########################################
# DQNAgent
##########################################
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.995, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = deque(maxlen=10000) #与论文设定不同256
        self.prioritized_memory = True  # 添加优先经验回放
        self.batch_size = 256
        self.gamma = gamma

        # 初始探索率相关参数
        self.initial_epsilon = 1.0  # 初始探索率
        self.epsilon = self.initial_epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # 添加epsilon衰减计数器
        self.epsilon_update_freq = 10  # 每10个epoch更新一次epsilon

        # 添加步数计数和目标网络更新频率
        self.steps_done = 0
        self.target_update_freq = 25  # 每25步更新一次

    def reset_epsilon(self):
        """重置探索率，用于每一层训练开始时"""
        self.epsilon = self.initial_epsilon
        self.epsilon_update_counter = 0

    def select_action(self, state, valid_actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        if random.random() < self.epsilon:
            # 在合法动作中随机选择
            valid_indices = [i for i, valid in enumerate(valid_actions) if valid]
            return random.choice(valid_indices)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)[0]  # 一次前向传播得到所有动作的 Q 值
            # 将非法动作的 Q 值设为负无穷，确保不会被选中
            # 应用动作掩码
            mask = torch.tensor(valid_actions, dtype=torch.bool, device=device)
            valid_q = torch.where(mask, q_values, torch.tensor(-float('inf'), device=device))
            return valid_q.argmax().item()

    # 添加新的优先经验采样方法
    def sample_batch(self):
        """使用优先经验回放采样批次"""
        if not self.prioritized_memory or len(self.memory) < self.batch_size:
            # 随机采样
            samples = random.sample(self.memory, min(len(self.memory), self.batch_size))
            # 去掉优先级部分返回
            return [(s[0], s[1], s[2], s[3]) for s in samples]

        # 提取所有优先级
        priorities = np.array([exp[4] for exp in self.memory])
        probabilities = priorities / np.sum(priorities)

        # 选择样本
        batch_size = min(self.batch_size, len(self.memory))
        replace = len(self.memory) < self.batch_size
        indices = np.random.choice(
            len(self.memory),
            batch_size,
            p=probabilities,
            replace=replace
        )

        # 获取选中的样本并去掉优先级部分返回
        batch = [self.memory[idx] for idx in indices]
        return [(s[0], s[1], s[2], s[3]) for s in batch]  # 不返回优先级

    def compute_td_error(self, state, action, reward, next_state):
        """计算TD误差用于优先经验回放"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # 计算当前Q值
            current_q = self.policy_net(state_tensor).gather(1, torch.tensor([[action]], device=device)).item()

            # 计算下一状态最大Q值
            next_q = 0
            if next_state is not None:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                next_q = self.target_net(next_state_tensor).max(1)[0].item()

            # 计算TD误差
            td_error = reward + self.gamma * next_q - current_q

        return abs(td_error) + 0.1  # 加一个小常数确保所有经验都有机会被选中

    def store_transition(self, state, action, reward, next_state):
        # self.memory.append((state, action, reward, next_state))

        # 对奖励较大的转换赋予更高优先级
        # priority = abs(reward) + 0.1  # 小常数确保所有经验都有一定概率被选中
        # TD
        priority = self.compute_td_error(state, action, reward, next_state)
        # 存储经验和优先级
        self.memory.append((state, action, reward, next_state, priority))


    def update_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0, 0.0 # 返回loss, max_q, min_q
        # 计数更新步数
        self.steps_done += 1
#for
        training_iterations = 5

        avg_loss = 0.0
        avg_max_q = 0.0
        avg_min_q = 0.0

        for _ in range(training_iterations):
            # 批量采样（考虑使用新的优先级方法）
            # batch = random.sample(self.memory, self.batch_size)
            batch = self.sample_batch()
            states, actions, rewards, next_states = zip(*batch)

            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)

            # 处理非终止状态
            non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool, device=device)
            non_final_next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None])).to(device)

            # 计算当前 Q 值
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # 计算目标Q值
            next_q = torch.zeros(self.batch_size, device=device)
            if non_final_mask.sum() > 0:
                # 使用策略网络选择动作
                next_action_values = self.policy_net(non_final_next_states)
                next_actions = next_action_values.max(1)[1].detach()
                # 使用目标网络评估这些动作（Double DQN）
                next_q[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
            target_q = rewards + self.gamma * next_q

            # 使用Huber损失代替MSE以提高稳定性
            loss = nn.SmoothL1Loss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) #减小梯度裁剪范数
            self.optimizer.step()

            # 更新统计量
            with torch.no_grad():
                avg_loss += loss.item()
                avg_max_q += current_q.max().item()
                avg_min_q += current_q.min().item()

        # 每C步更新一次目标网络
        if self.steps_done % self.target_update_freq == 0:
            self.soft_update_target()

        # 返回平均统计量
        return avg_loss / training_iterations, avg_max_q / training_iterations, avg_min_q / training_iterations

    def soft_update_target(self, tau=0.01):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def update_epsilon(self):
        """在每个epoch结束后调用，控制epsilon的衰减频率"""
        self.epsilon_update_counter += 1
        if self.epsilon_update_counter >= self.epsilon_update_freq:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.epsilon_update_counter = 0


##########################################
# 分层训练阶段：每层训练只保留当前层与上层的信息
# 将训练阶段与重跑构造阶段分离，训练阶段保存每一层训练得到的 RL 模型
##########################################
def hierarchical_packing_training(bottom_nodes, query_workload, max_level=20):
    # 首先确保底层节点有正确的 id 和 layer 信息
    for i, node in enumerate(bottom_nodes):
        if 'id' not in node:
            node['id'] = i  # 简单整数id，表示节点序号
        if 'layer' not in node:
            node['layer'] = 0

    current_layer = bottom_nodes
    current_layer_level = 0
    training_upper_layers = []  # 保存每层的中间上层节点（训练阶段结果）
    all_rewards = []          # 保存每层训练时的奖励曲线

    # 添加列表跟踪每层的访问成本
    layer_access_costs = []

    # 初始化全局单一agent（所有层共用）
    # 计算状态空间和动作空间维度
    m = len(query_workload)
    N = len(bottom_nodes)
    state_dim = (m + 1) * N + m
    action_dim = N
    global_agent = DQNAgent(state_dim, action_dim)

    for level in range(max_level):
        # 避免多层经验混淆
        global_agent.memory.clear()
         # 初始化前一层的非空节点数
        prev_non_empty = sum(1 for node in current_layer if  node['MBR'] is not None)
        print(f"Level {level + 1}: Non-empty nodes = {prev_non_empty}")

        # 终止条件：只剩一个非空节点
        if prev_non_empty == 1:
            print(f"Reached one non-empty node. Stopping training.")
            break

        print(f"\n[Training Phase] Level {level + 1} Training...")

        # ===== 终止条件1：计算当前层的初始平均访问成本 =====
        init_env = PackingEnv(current_layer, query_workload, current_layer_level)
        initial_layer_cost = init_env._calculate_access_cost()
        print(f"Level {level + 1} 初始平均访问成本: {initial_layer_cost:.4f}")

        # 重置探索率，每一层都从高探索率开始
        global_agent.reset_epsilon()
        print(f"已重置探索率 epsilon 为 {global_agent.epsilon}")

        # 训练当前层：当前层节点与预先初始化的 N 个空上层节点组成固定状态:仅在第一层应用排序
        _, total_rewards = train_single_level(current_layer, query_workload, current_layer_level, global_agent, is_first_layer=(level == 0))
        all_rewards.append(total_rewards)

        # 用当前训练好的代理对当前层数据进行一次模拟打包（去除探索：：仅在第一层使用排序
        if level == 0:
            ordered_nodes = get_ordered_nodes(current_layer, order_strategy='hilbert')
        else:
            ordered_nodes = current_layer  # 保持原始顺序

        env = PackingEnv(ordered_nodes, query_workload, current_layer_level)
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            # 使用纯贪婪策略选择动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = global_agent.policy_net(state_tensor)[0]
                mask = torch.tensor(valid_actions, dtype=torch.bool, device=device)
                valid_q = torch.where(mask, q_values, torch.tensor(-float('inf'), device=device))
                action = valid_q.argmax().item()

            next_state, reward, done = env.step(action)

            if next_state is not None:
                state = next_state
            else:
                break

        # ===== 终止条件2：计算打包后的平均访问成本并比较 =====
        final_layer_cost = env._calculate_access_cost()
        layer_access_costs.append(final_layer_cost)
        print(f"Level {level + 1} 最终平均访问成本: {final_layer_cost:.4f}")

        # 终止条件2.1：如果打包后的成本高于该层的初始成本，终止
        if final_layer_cost > initial_layer_cost:
            print(f"警告: Level {level + 1} 打包后的访问成本({final_layer_cost:.4f})高于初始成本({initial_layer_cost:.4f})")
            print(f"打包效果不佳，终止层级构建")
            break

        # 终止条件2.2：如果打包后的成本高于初始节点数N，终止
        if final_layer_cost > N:
            print(f"警告: Level {level + 1} 打包后的访问成本({final_layer_cost:.4f})高于初始节点数({N})")
            print(f"打包效果不佳，终止层级构建")
            break

        # 打印本层打包后上层节点的详细信息
        print(f"\nAfter packing Level {level + 1}, upper_layer nodes:")
        print_layer_nodes(env.upper_layer, current_layer_level + 1)

        # 检查生成的upper_layer是否为空
        if not env.upper_layer:
            print("Error: Upper layer is empty. Training terminated.")
            break

        current_layer = env.upper_layer  # 下一层的输入为本层的所有上层节点
        current_non_empty = sum(1 for node in current_layer if  node['MBR'] is not None)
        current_layer_level += 1


        # 保存当前层的上层节点（全部 N 个，状态维度固定）
        training_upper_layers.append(env.upper_layer.copy())

        # 终止条件3 ：上层空节点只剩一个
        if current_non_empty == 1:
            print(f"Reached one non-empty node. Stopping training.")
            break

        prev_non_empty = current_non_empty

    # 绘制各层训练奖励曲线
    n_levels = len(all_rewards)
    fig, axs = plt.subplots(n_levels, 1, figsize=(10, 4 * n_levels))
    if n_levels == 1:
        axs = [axs]
    for i, rewards in enumerate(all_rewards):
        rewards_array = np.array([float(r) for r in rewards])
        axs[i].plot(range(1, len(rewards_array) + 1), rewards_array, marker='o', label=f"Level {i + 1}")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Total Reward")
        axs[i].set_title(f"Convergence Curve for Level {i + 1}")
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()
    plt.show()

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 绘制层级访问成本变化图
    if layer_access_costs:
        plt.figure(figsize=(10, 6))
        levels = range(1, len(layer_access_costs) + 1)
        plt.plot(levels, layer_access_costs, marker='o', linestyle='-', label='层级访问成本')
        plt.axhline(y=N, color='r', linestyle='--', label=f'初始节点数 N={N}')
        plt.xlabel('Layer')
        plt.ylabel('Avg access cost')
        plt.title('Each level costs change after packing ')
        plt.xticks(levels)
        plt.grid(True)
        plt.legend()
        plt.show()

    print(f"\n[Training Phase] Total Levels Trained: {current_layer_level}")
    return global_agent, training_upper_layers, current_layer_level

##########################################
# 单层训练函数：训练 RL 代理以优化当前层的 packing 策略
##########################################
def train_single_level(current_layer, query_workload, current_layer_level, agent, epochs=800, is_first_layer=False):
    # 添加数据验证
    if len(query_workload) == 0:
        raise ValueError("Query workload is empty")
    if len(current_layer) == 0:
        raise ValueError("Current layer nodes are empty")

    total_rewards = []  # 记录每轮训练的总奖励
    loss_history = []  # 新增loss记录
    q_values = []  # 新增Q值记录

    for epoch in range(epochs):
        # 保持原始顺序不变
        # env = PackingEnv(current_layer, query_workload, current_layer_level)

        # Order nodes according to strategy
        # 仅在第一层应用排序策略
        if is_first_layer:
            ordered_nodes = get_ordered_nodes(current_layer, order_strategy='hilbert')
        else:
            ordered_nodes = current_layer  # 使用原始顺序

        env = PackingEnv(ordered_nodes, query_workload, current_layer_level)

        # # 随机打乱节点顺序增加多样性
        # if random.random() < 0:  # 30%概率打乱顺序
        #     shuffled_indices = list(range(len(current_layer)))
        #     random.shuffle(shuffled_indices)
        #     shuffled_layer = [current_layer[i] for i in shuffled_indices]
        #     env = PackingEnv(shuffled_layer, query_workload, current_layer_level)
        # else:
        #     env = PackingEnv(current_layer, query_workload, current_layer_level)



        # 更积极的打乱方式
        #random.seed(epoch)
        # if epoch % 3 == 0:  # 每3个epoch
        #     shuffled_indices = list(range(len(current_layer)))
        #     random.shuffle(shuffled_indices)
        #     shuffled_layer = [current_layer[i] for i in shuffled_indices]
        #     env = PackingEnv(shuffled_layer, query_workload, current_layer_level)
        # elif epoch % 3 == 1:  # 每3+1个epoch
        #     # 尝试反转顺序
        #     reversed_layer = current_layer.copy()
        #     reversed_layer.reverse()
        #     env = PackingEnv(reversed_layer, query_workload, current_layer_level)
        # else:
        #     # 正常顺序
        #     env = PackingEnv(current_layer, query_workload, current_layer_level)

        # # 使用课程学习方法
        # if epoch < epochs // 3:  # 训练的前三分之一
        #     # 更多探索，一些打乱
        #     if epoch % 5 == 0:  # 降低打乱频率
        #         shuffled_indices = list(range(len(current_layer)))
        #         random.shuffle(shuffled_indices)
        #         shuffled_layer = [current_layer[i] for i in shuffled_indices]
        #         env = PackingEnv(shuffled_layer, query_workload, current_layer_level)
        #     else:
        #         env = PackingEnv(current_layer, query_workload, current_layer_level)
        # else:
        #     # 训练后期，使用一致的顺序以保持稳定
        #     env = PackingEnv(current_layer, query_workload, current_layer_level)

        state = env.reset()
        total_reward = 0
        done = False
        epoch_loss = []
        epoch_max_q = []
        epoch_min_q = []
        updates = 0  # 跟踪当前epoch的网络更新次数

        while not done:
            valid_actions = env.get_valid_actions() # 获取布尔掩码
            action = agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)

            total_reward += reward

            # 存储经验
            if next_state is not None:
                agent.store_transition(state, action, reward, next_state)
                # 每个状态转换后进行网络更新
                if len(agent.memory) >= agent.batch_size:
                    loss, max_q, min_q = agent.update_model()
                    if loss != 0.0:
                        epoch_loss.append(loss)
                        epoch_max_q.append(max_q)
                        epoch_min_q.append(min_q)
                    updates += 1
                state = next_state
            else:
                agent.store_transition(state, action, reward, next_state)
                break

        agent.update_epsilon()

        # 记录统计量
        avg_loss = np.mean(epoch_loss) if epoch_loss else 0.0
        avg_max_q = np.mean(epoch_max_q) if epoch_max_q else 0.0
        avg_min_q = np.mean(epoch_min_q) if epoch_min_q else 0.0
        loss_history.append(avg_loss)
        q_values.append((avg_max_q, avg_min_q))
        total_rewards.append(total_reward)
        # 修改打印语句
        print(f"Epoch {epoch + 1} | ε={agent.epsilon:.3f} | Loss: {avg_loss:.2f} | "
                  f"Q(max/min): {avg_max_q:.2f}/{avg_min_q:.2f} | Reward: {total_reward:.2f}")

        # 添加监控曲线绘制
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'b-')
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot([q[0] for q in q_values], 'r-', label='Max Q')
    plt.plot([q[1] for q in q_values], 'g-', label='Min Q')
    plt.title("Q-Value Range")
    plt.legend()
    plt.show()

    return agent, total_rewards

##########################################
# 重跑构造阶段：利用训练阶段学到的 RL 模型，重新跑一遍所有数据以生成最终树结构
##########################################
def final_tree_construction(bottom_nodes, query_workload, global_agent):
    """
    利用训练阶段学到的RL模型，重新构造最终树结构
    添加终止条件：
    1. 当某层打包后的平均访问成本高于该层的初始成本时终止
    2. 当某层打包后的平均访问成本高于初始节点数N时终止
    3. 当只剩一个非空节点时终止（已达到最优压缩）
    4. 当连续多层非空节点数量不变时终止（检测停滞）
    """
    # 确保底层节点有正确的 id 和 layer 信息
    for i, node in enumerate(bottom_nodes):
        if 'id' not in node:
            node['id'] = i  # 简单整数id
        if 'layer' not in node:
            node['layer'] = 0

    current_layer = bottom_nodes
    filter_current_layer =  [node for node in current_layer if  node['MBR'] is not None]
    final_tree_structure = [filter_current_layer.copy()]  # 保存完整层级结构 (第0层 # 每层的最终上层节点（固定 N 个）
    level = 0
    current_layer_level = 0

    # 设置为贪婪策略（不再探索）
    global_agent.epsilon = 0.0

    # 初始节点数
    N = len(bottom_nodes)
    # 记录层级访问成本变化
    layer_access_costs = []
    # 跟踪非空节点数，用于检测停滞
    non_empty_counts = []
    stagnation_limit = 5  # 连续相同非空节点数的层数限制

    while True:
        print(f"\n[Final Construction Phase] Processing Level {level + 1} ...")
        # 计算当前层非空节点数
        current_non_empty = sum(1 for node in current_layer if node['MBR'] is not None)
        non_empty_counts.append(current_non_empty)
        print(f"Level {level + 1}: Non-empty nodes = {current_non_empty}")

        # 终止条件3：只剩一个非空节点
        if current_non_empty == 1:
            print(f"Reached one non-empty node. Stopping final construction.")
            break

        # 终止条件4：检测非空节点数停滞
        if len(non_empty_counts) >= stagnation_limit:
            last_counts = non_empty_counts[-stagnation_limit:]
            if all(count == last_counts[0] for count in last_counts):
                print(f"Detected stagnation: Non-empty node count remained at {last_counts[0]} for {stagnation_limit} consecutive levels.")
                print(f"Stopping final construction to avoid excessive layers.")
                break

        # 计算当前层的初始访问成本
        init_env = PackingEnv(current_layer, query_workload, current_layer_level)
        initial_layer_cost = init_env._calculate_access_cost()
        print(f"Level {level + 1} 初始平均访问成本: {initial_layer_cost:.4f}")


        # 在重跑阶段采用贪婪策略：设置 ε = 0
        # 仅在第一层应用排序
        if level == 0:
            ordered_nodes = get_ordered_nodes(current_layer, order_strategy='hilbert')
        else:
            ordered_nodes = current_layer  # 使用原始顺序
        env = PackingEnv(ordered_nodes, query_workload, current_layer_level)

        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = global_agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            total_reward += reward

            if next_state is not None:
                state = next_state
            else:
                break

        # 计算打包后的平均访问成本
        final_layer_cost = env._calculate_access_cost()
        layer_access_costs.append(final_layer_cost)
        print(f"Level {level + 1} 最终平均访问成本: {final_layer_cost:.4f}")

        # 终止条件1：打包后的成本高于该层初始成本
        if final_layer_cost > initial_layer_cost:
            print(f"警告: Level {level + 1} 打包后的访问成本({final_layer_cost:.4f})高于该层初始成本({initial_layer_cost:.4f})")
            print(f"打包效果不佳，终止层级构建")
            break

        # 终止条件2：打包后的成本高于初始节点数N
        if final_layer_cost > N:
            print(f"警告: Level {level + 1} 打包后的访问成本({final_layer_cost:.4f})高于初始节点数({N})")
            print(f"打包效果不佳，终止层级构建")
            break

        # 打印当前层级打包后的上层节点信息
        print(f"\nAfter final construction of Level {current_layer_level + 1}, upper_layer nodes:")
        print_layer_nodes(env.upper_layer, current_layer_level + 1)

        filter_upper_layer = [node for node in env.upper_layer if node['MBR'] is not None]
        final_tree_structure.append(filter_upper_layer.copy())

        current_layer = env.upper_layer  # 下一层输入为本层所有上层节点
        current_layer_level += 1
        level += 1
    # 绘制层级访问成本变化图
    if layer_access_costs:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.figure(figsize=(10, 6))
        levels = range(1, len(layer_access_costs) + 1)
        plt.plot(levels, layer_access_costs, marker='o', linestyle='-', label='层级访问成本')
        plt.axhline(y=N, color='r', linestyle='--', label=f'初始节点数 N={N}')
        plt.xlabel('层级')
        plt.ylabel('平均访问成本')
        plt.title('重跑阶段各层级打包后的平均访问成本变化')
        plt.xticks(levels)
        plt.grid(True)
        plt.legend()
        plt.show()
    print(f"\n[Final Construction Phase] Total Levels in Final Tree: {level}")
    print_tree_structure(final_tree_structure)
    return final_tree_structure

##########################################
# build_nested_tree 通过层级映射将重跑构造的索引结构转换为实际节点对象
##########################################
def build_nested_tree(tree_structure):
    """
    从底层向上构建嵌套的树结构
    Args: tree_structure: 由层次结构组成的列表，每个元素是该层的节点列表
    Returns:顶层的根节点列表（可能包含多个根节点）
    """
    if not tree_structure or len(tree_structure) == 0:
        return None

    # 从底层向上构建
    for layer_idx in range(1, len(tree_structure)):
        current_layer = tree_structure[layer_idx]  # 当前处理的层
        lower_layer = tree_structure[layer_idx - 1]  # 下一层（子节点所在层）

        # 为每一层创建ID映射
        lower_layer_map = {node['id']: node for node in lower_layer}

        for node in current_layer:
            if node['MBR'] is None:  # 跳过空节点
                continue

            # 收集实际的子节点对象
            real_children = []
            # for child_idx in node['children']:
            #     # 确保索引有效
            #     if isinstance(child_idx, int) and child_idx < len(lower_layer):
            #         child_node = lower_layer[child_idx]
            #         if child_node['MBR'] is not None:  # 只添加非空节点
            #             real_children.append(child_node)
            for child_id in node['children']:
                # 从下层映射中查找对应节点
                if child_id in lower_layer_map:
                    child_node = lower_layer_map[child_id]
                    if child_node['MBR'] is not None:
                        real_children.append(child_node)
                else:
                    print(f"警告：在层{layer_idx - 1}中未找到ID为{child_id}的节点")
            # 更新节点的children字段为实际节点引用
            node['children'] = real_children

    # 获取顶层的所有非空节点作为根节点
    root_nodes = [node for node in tree_structure[-1] if node['MBR'] is not None]

    # 打印构建的嵌套树结构
    print_nested_tree(root_nodes)

    # 根据根节点数量返回适当的结果
    if len(root_nodes) == 0:
        print("警告：未找到有效的根节点。")
        return None
    elif len(root_nodes) == 1:
        print(f"树结构有1个根节点。")
        return root_nodes[0]  # 只有一个根节点时返回该节点
    else:
        print(f"树结构有{len(root_nodes)}个根节点。")
        return root_nodes  # 多个根节点时返回节点列表

def print_layer_nodes(nodes, layer_level):
    print(f"----- Layer {layer_level} Nodes Detail -----")
    for idx, node in enumerate(nodes):
        children_count = len(node['children'])
        labels = node['labels']
        mbr = node.get('MBR')
        print(f"\nNode {idx}: children_count = {children_count}, labels = {labels}, MBR = {mbr}")


def print_tree_structure(tree_structure):
    """
    打印树结构的统计信息和每层节点的简要信息

    Args:
        tree_structure: final_tree_construction返回的列表，包含每层节点列表
    """
    print("\n======= Tree Structure Summary =======")
    print(f"Total Levels: {len(tree_structure)}")

    for level, nodes in enumerate(tree_structure):
        # 只计算非空节点
        non_empty_nodes = [node for node in nodes if node['MBR'] is not None]
        print(f"\nLevel {level}: {len(non_empty_nodes)} non-empty nodes")

        # 打印每个节点的简要信息
        for i, node in enumerate(non_empty_nodes):
            children_count = len(node['children'])
            label_count = len(node['labels']) if 'labels' in node else 0
            mbr_info = "Present" if node['MBR'] is not None else "None"
            print(f"  Node {i}: {children_count} children, {label_count} labels, MBR: {mbr_info}")


def print_nested_tree(root_nodes):
    """
    打印构建好的嵌套树结构，只显示每层节点数
    """
    print("\n======= Nested Tree Structure =======")

    # 处理单个节点或节点列表
    if isinstance(root_nodes, list):
        print(f"总共有 {len(root_nodes)} 个根节点")
        for i, node in enumerate(root_nodes):
            print(f"\n根节点 {i + 1}:")
            # 用字典记录每层的节点数
            layer_counts = {}
            _count_nodes_by_layer(node, 0, layer_counts)

            # 打印每层的节点数
            for layer, count in sorted(layer_counts.items()):
                print(f"Layer {layer}: {count} nodes")
    else:
        print("单一根节点树结构:")
        layer_counts = {}
        _count_nodes_by_layer(root_nodes, 0, layer_counts)

        for layer, count in sorted(layer_counts.items()):
            print(f"Layer {layer}: {count} nodes")


def _count_nodes_by_layer(node, level, layer_counts):
    """
    递归辅助函数，统计每层的节点数
    """
    # 记录当前层的节点
    layer = node.get('layer', level)
    if layer not in layer_counts:
        layer_counts[layer] = 0
    layer_counts[layer] += 1

    # 递归处理子节点
    if 'children' in node and len(node['children']) > 0:
        if isinstance(node['children'][0], dict):  # 子节点是对象
            for child in node['children']:
                _count_nodes_by_layer(child, level + 1, layer_counts)


def _print_node_recursive(node, level=0):
    """
    递归辅助函数，用于打印嵌套树中的节点
    """
    indent = "  " * level
    children_count = len(node['children']) if 'children' in node else 0
    label_count = len(node['labels']) if 'labels' in node else 0

    print(f"{indent}Level {node.get('layer', '?')} Node: {children_count} children, {label_count} labels")

    # 递归打印子节点
    if 'children' in node and isinstance(node['children'][0], dict):  # 确保children是节点对象而非索引
        for i, child in enumerate(node['children']):
            if i < 5:  # 只打印前5个子节点，避免输出过长
                _print_node_recursive(child, level + 1)
            elif i == 5:
                print(f"{indent}  ... and {len(node['children']) - 5} more children")
                break


import numpy as np


def get_ordered_nodes(current_layer, order_strategy='hilbert'):
    """
    Order nodes based on different strategies, with special handling for empty nodes.

    Args:
        current_layer: List of nodes to be ordered
        order_strategy: Strategy to order nodes
            - 'original': Keep original order (as obtained from clustering)
            - 'lat': Order by latitude center
            - 'lon': Order by longitude center
            - 'hilbert': Order by Hilbert space-filling curve
            - 'z_order': Order by Z-order space-filling curve (Morton order)
            - 'random': Random ordering
            - 'non_empty_first': Non-empty nodes first, then empty nodes

    Returns:
        Ordered list of nodes
    """
    # Separate empty and non-empty nodes
    empty_nodes = [node for node in current_layer if node.get('MBR') is None]
    non_empty_nodes = [node for node in current_layer if node.get('MBR') is not None]

    if order_strategy == 'original':
        # Keep original order (mixed empty and non-empty)
        return current_layer

    elif order_strategy == 'non_empty_first':
        # Place all non-empty nodes first, then empty nodes
        # Use original order within each group
        return non_empty_nodes + empty_nodes

    elif order_strategy == 'lat':
        # Order non-empty nodes by latitude center
        ordered_non_empty = sorted(non_empty_nodes,key=lambda n: (n['MBR']['min_lat'] + n['MBR']['max_lat']) / 2)

        # Put empty nodes at the end (or beginning based on preference)
        return ordered_non_empty + empty_nodes

    elif order_strategy == 'lon':
        # Order non-empty nodes by longitude center
        ordered_non_empty = sorted(non_empty_nodes,key=lambda n: (n['MBR']['min_lon'] + n['MBR']['max_lon']) / 2)

        # Put empty nodes at the end
        return ordered_non_empty + empty_nodes

    elif order_strategy == 'hilbert':
        # Order non-empty nodes by Hilbert space-filling curve
        hilbert_ordered = []

        # Get coordinates for valid nodes
        coords = []
        for node in non_empty_nodes:
            # Use center point of MBR
            mbr = node['MBR']
            x = (mbr['min_lon'] + mbr['max_lon']) / 2  # Center longitude
            y = (mbr['min_lat'] + mbr['max_lat']) / 2  # Center latitude
            coords.append((x, y, node))

        # Normalize coordinates to [0,1] for Hilbert mapping
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Avoid division by zero
            x_range = max_x - min_x or 1
            y_range = max_y - min_y or 1

            # Assign Hilbert distance
            hilbert_coords = []
            for x, y, node in coords:
                # Normalize to [0,1]
                nx = (x - min_x) / x_range
                ny = (y - min_y) / y_range

                # Calculate Hilbert distance (2D Hilbert curve)
                h_dist = hilbert_distance(nx, ny, 5)  # 5 iterations for precision
                hilbert_coords.append((h_dist, node))

            # Sort by Hilbert distance
            hilbert_coords.sort(key=lambda x: x[0])
            hilbert_ordered = [hc[1] for hc in hilbert_coords]

        # Put empty nodes at the end
        return hilbert_ordered + empty_nodes

    elif order_strategy == 'z_order':
        # Order non-empty nodes by Z-order curve (Morton order)
        z_ordered = []

        # Get coordinates for valid nodes
        coords = []
        for node in non_empty_nodes:
            # Use center point of MBR
            mbr = node['MBR']
            x = (mbr['min_lon'] + mbr['max_lon']) / 2  # Center longitude
            y = (mbr['min_lat'] + mbr['max_lat']) / 2  # Center latitude
            coords.append((x, y, node))

        # Normalize coordinates to [0,1] for Z-order mapping
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Avoid division by zero
            x_range = max_x - min_x or 1
            y_range = max_y - min_y or 1

            # Assign Z-order distance
            z_coords = []
            for x, y, node in coords:
                # Normalize to [0,1]
                nx = (x - min_x) / x_range
                ny = (y - min_y) / y_range

                # Calculate Z-order (interleave bits)
                z_dist = interleave(nx, ny)
                z_coords.append((z_dist, node))

            # Sort by Z-order distance
            z_coords.sort(key=lambda x: x[0])
            z_ordered = [zc[1] for zc in z_coords]

        # Put empty nodes at the end
        return z_ordered + empty_nodes

    elif order_strategy == 'random':
        # Random ordering of non-empty nodes
        shuffled_non_empty = non_empty_nodes.copy()
        random.shuffle(shuffled_non_empty)

        # Put empty nodes at the end (or randomize their position too if preferred)
        return shuffled_non_empty + empty_nodes

    else:
        print(f"Unknown ordering strategy: {order_strategy}. Using original order.")
        return current_layer


def hilbert_distance(x, y, iterations=5):
    """
    Calculate the distance along a Hilbert curve for a point (x,y) in [0,1]×[0,1]
    Uses iterative algorithm for 2D Hilbert curve.
    """
    # Convert to integer coordinates
    x = int(x * (2 ** iterations - 1))
    y = int(y * (2 ** iterations - 1))

    d = 0
    s = 2 ** iterations // 2

    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0

        # Add contribution for this iteration
        d += s * s * ((3 * rx) ^ ry)

        # Rotate quadrant if needed to get next iteration's coordinates
        if ry == 0:
            if rx == 1:
                x = (2 ** iterations - 1) - x
                y = (2 ** iterations - 1) - y

            # Swap x and y
            x, y = y, x

        s //= 2

    return d


def interleave(x, y):
    """
    Interleave bits of x and y to create Z-order value.
    Assumes x, y are normalized to [0,1]
    """
    # Convert to integer coordinates with 16-bit precision
    x_int = int(x * 65535)
    y_int = int(y * 65535)

    # Interleave bits
    z = 0
    for i in range(16):
        z |= ((x_int & (1 << i)) << i) | ((y_int & (1 << i)) << (i + 1))

    return z