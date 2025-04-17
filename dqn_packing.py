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
        self.current_layer = current_layer_nodes
        self.query_workload = query_workload
        self.m = len(query_workload)
        self.N = len(current_layer_nodes)
        self.current_layer_level = current_layer_level

        # 预计算查询区域边界
        self.query_areas = []
        for query in query_workload:
            self.query_areas.append({
                'min_lat': query['area']['min_lat'],
                'max_lat': query['area']['max_lat'],
                'min_lon': query['area']['min_lon'],
                'max_lon': query['area']['max_lon'],
                'keywords': set(query['keywords'])
            })

        # 预先创建上层节点
        self.upper_layer = []
        for i in range(self.N):
            node = {'id': i, 'layer': self.current_layer_level + 1, 'labels': [], 'children': [], 'MBR': None}
            self.upper_layer.append(node)

        self.current_step = 0
        if self.current_layer:
            self.current_node = self.current_layer[self.current_step]
        else:
            self.current_node = None
        self.total_reward = 0.0

        # 缓存以加速计算
        self._intersect_cache = {}
        self._keyword_match_cache = {}

    def _mbr_intersect(self, mbr, query_area):
        """优化的MBR交集计算，使用缓存"""
        # 创建缓存键
        cache_key = (
            mbr.get('min_lat', float('inf')),
            mbr.get('max_lat', float('-inf')),
            mbr.get('min_lon', float('inf')),
            mbr.get('max_lon', float('-inf')),
            query_area['min_lat'],
            query_area['max_lat'],
            query_area['min_lon'],
            query_area['max_lon']
        )

        # 检查缓存
        if cache_key in self._intersect_cache:
            return self._intersect_cache[cache_key]

        # 计算交集
        result = not (
                mbr['max_lat'] < query_area['min_lat'] or
                mbr['min_lat'] > query_area['max_lat'] or
                mbr['max_lon'] < query_area['min_lon'] or
                mbr['min_lon'] > query_area['max_lon']
        )

        # 存入缓存
        self._intersect_cache[cache_key] = result
        return result

    def _get_state(self):
        """优化的状态向量生成，使用向量化操作"""
        state_dim = (self.m + 1) * self.N + self.m
        state = np.zeros(state_dim, dtype=np.float32)

        # 预计算当前所有上层节点的关键词集合
        node_keywords = []
        for node in self.upper_layer:
            node_keywords.append(set(node['labels']))

        # 并行计算所有节点对所有查询的匹配情况
        for i, node in enumerate(self.upper_layer):
            if node['MBR'] is None:
                continue  # 跳过空节点

            # 优化：批量计算所有查询的空间和关键词匹配
            for j, query_area in enumerate(self.query_areas):
                # 缓存节点-查询的关键词匹配结果
                kw_cache_key = (i, j)
                if kw_cache_key not in self._keyword_match_cache:
                    self._keyword_match_cache[kw_cache_key] = bool(node_keywords[i] & query_area['keywords'])

                # 空间和关键词都匹配才设置为1
                if (node['MBR'] is not None and
                        self._mbr_intersect(node['MBR'], query_area) and
                        self._keyword_match_cache[kw_cache_key]):
                    state[i * (self.m + 1) + j] = 1

            # 设置孩子节点数量
            state[i * (self.m + 1) + self.m] = len(node['children'])

        # 下一底层节点信息
        if self.current_step + 1 < len(self.current_layer):
            next_node = self.current_layer[self.current_step + 1]
            next_node_keywords = set(next_node['labels']) if next_node.get('labels') else set()

            for j, query_area in enumerate(self.query_areas):
                # 空间和关键词匹配
                next_node_spatial = self._mbr_intersect(next_node['MBR'], query_area) if next_node.get('MBR') else False
                next_node_keyword = bool(next_node_keywords & query_area['keywords'])

                if next_node_spatial and next_node_keyword:
                    state[self.N * (self.m + 1) + j] = 1

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
        """环境步进函数，优化MBR计算和缓存失效处理"""
        done = False
        prev_access = self._calculate_access_cost()

        # 跳过空节点
        while self.current_node is not None and self.current_node['MBR'] is None:
            self.current_step += 1
            if self.current_step < len(self.current_layer):
                self.current_node = self.current_layer[self.current_step]
            else:
                done = True
                return None, 0, done

        # 将当前底层节点合并到选定的上层节点
        target = self.upper_layer[action]

        # 当目标节点变化时，使相关缓存失效
        for key in list(self._keyword_match_cache.keys()):
            if key[0] == action:
                del self._keyword_match_cache[key]

        # 更新目标节点
        current_labels = list(self.current_node['labels'])
        target['labels'] = list(set(target['labels'] + current_labels))
        target['children'].append(self.current_step)

        # 更新MBR
        if self.current_node['MBR'] is not None:
            if target['MBR'] is None:
                target['MBR'] = self.current_node['MBR'].copy()  # 创建副本避免引用问题
            else:
                # MBR发生变化，清理相关的交集缓存
                old_mbr = (target['MBR']['min_lat'], target['MBR']['max_lat'],
                           target['MBR']['min_lon'], target['MBR']['max_lon'])

                # 更新MBR
                target['MBR'] = self._merge_mbr(target['MBR'], self.current_node['MBR'])

                # 清理受影响的缓存项
                for key in list(self._intersect_cache.keys()):
                    if key[0:4] == old_mbr:
                        del self._intersect_cache[key]

        # 前进到下一步
        self.current_step += 1
        if self.current_step >= len(self.current_layer):
            done = True
            next_state = None
        else:
            self.current_node = self.current_layer[self.current_step]
            next_state = self._get_state()

        new_access = self._calculate_access_cost()
        reward = prev_access - new_access
        self.total_reward += reward

        if done and (new_access > self.N):
            done = True
            print("终止条件触发：当前访问节点数大于初始访问节点数")

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

        # 预计算每个查询的总访问成本
        for query_idx, query in enumerate(self.query_workload):
            query_area = self.query_areas[query_idx]
            query_access = 0

            # 上层节点部分
            for node_idx, node in enumerate(self.upper_layer):
                if node['MBR'] is None:
                    continue

                # 基本访问成本
                query_access += 1

                # 检查匹配并计算子节点访问
                if self._mbr_intersect(node['MBR'], query_area) and any(
                        kw in node['labels'] for kw in query['keywords']):
                    query_access += len(node['children'])

            # 未打包节点部分
            for i in range(self.current_step, len(self.current_layer)):
                if self.current_layer[i]['MBR'] is not None:
                    query_access += 1

            total_access += query_access

        return total_access / len(self.query_workload)

    def _merge_mbr(self, mbr1, mbr2):
        return {
            'min_lat': min(mbr1['min_lat'], mbr2['min_lat']),
            'max_lat': max(mbr1['max_lat'], mbr2['max_lat']),
            'min_lon': min(mbr1['min_lon'], mbr2['min_lon']),
            'max_lon': max(mbr1['max_lon'], mbr2['max_lon'])
        }

    def reset(self):
        """重置环境状态"""
        self.current_step = 0
        if self.current_layer:
            self.current_node = self.current_layer[self.current_step]
        else:
            self.current_node = None
        self.total_reward = 0.0
        # 清空缓存
        self._intersect_cache = {}
        self._keyword_match_cache = {}
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
        return self.net(state)


##########################################
# DQNAgent
##########################################
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 创建策略网络和目标网络并移至GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 优化器设置
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # 增大内存缓冲区和批大小
        self.memory = deque(maxlen=512)
        self.batch_size = 128
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.epsilon_update_counter = 0
        self.epsilon_update_freq = 10

        self.steps_done = 0
        self.target_update_freq = 10

        # 预分配张量，避免频繁创建
        self.device = device
        self.actions_tensor = torch.zeros(self.batch_size, dtype=torch.long, device=device)

    def select_action(self, state, valid_actions):
        """选择动作，使用GPU加速"""
        state_tensor = torch.FloatTensor(state).to(self.device)

        if random.random() < self.epsilon:
            # 在合法动作中随机选择
            valid_indices = [i for i, valid in enumerate(valid_actions) if valid]
            return random.choice(valid_indices)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                mask = torch.tensor(valid_actions, dtype=torch.bool, device=self.device)
                valid_q = torch.where(mask, q_values, torch.tensor(-float('inf'), device=self.device))
                return valid_q.argmax().item()

    def store_transition(self, state, action, reward, next_state):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state))

    def update_model(self):
        """批量更新模型参数，使用GPU加速"""
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0, 0.0

        self.steps_done += 1

        # 多次训练，保持训练次数不变
        avg_loss = 0.0
        avg_max_q = 0.0
        avg_min_q = 0.0
        training_iterations = 50

        for _ in range(training_iterations):
            # 采样批量经验
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states = zip(*batch)

            # 将数组转换为张量并移至GPU
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = self.actions_tensor
            for i, a in enumerate(actions):
                actions_tensor[i] = a
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)

            # 处理非终止状态
            non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool, device=self.device)
            non_final_next_states = torch.FloatTensor(np.array([s for s in next_states if s is not None])).to(
                self.device)

            # 计算当前Q值和下一状态的最大Q值
            all_q_values = self.policy_net(states_tensor)
            current_q = all_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            # 使用目标网络计算下一状态的最大Q值
            next_q = torch.zeros(self.batch_size, device=self.device)
            if non_final_mask.sum() > 0:
                with torch.no_grad():
                    next_state_values = self.target_net(non_final_next_states).max(1)[0]
                next_q[non_final_mask] = next_state_values

            # 计算目标Q值
            target_q = rewards_tensor + self.gamma * next_q

            # 使用MSE损失函数
            loss = nn.MSELoss()(current_q, target_q)

            # 更新模型
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 更新统计量
            with torch.no_grad():
                avg_loss += loss.item()
                avg_max_q += current_q.max().item()
                avg_min_q += current_q.min().item()

        # 使用软更新目标网络
        if self.steps_done % self.target_update_freq == 0:
            self.soft_update_target()

        # 返回平均统计量
        return avg_loss / training_iterations, avg_max_q / training_iterations, avg_min_q / training_iterations

    def soft_update_target(self, tau=0.001):
        """软更新目标网络参数"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def update_epsilon(self):
        """控制探索率的衰减"""
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

    # 初始化全局单一agent（所有层共用）
    # 计算状态空间和动作空间维度
    m = len(query_workload)
    N = len(bottom_nodes)
    state_dim = (m + 1) * N + m
    action_dim = N
    global_agent = DQNAgent(state_dim, action_dim)

    for level in range(max_level):
         # 初始化前一层的非空节点数
        prev_non_empty = sum(1 for node in current_layer if  node['MBR'] is not None)
        print(f"Level {level + 1}: Non-empty nodes = {prev_non_empty}")

        if prev_non_empty == 1:
            print(f"Reached one non-empty node. Stopping training.")
            break

        print(f"\n[Training Phase] Level {level + 1} Training...")
        # 训练当前层：当前层节点与预先初始化的 N 个空上层节点组成固定状态
        _, total_rewards = train_single_level(current_layer, query_workload, current_layer_level, global_agent)
        all_rewards.append(total_rewards)

        # 用当前训练好的代理对当前层数据进行一次模拟打包（依然使用 ε-greedy 策略）
        env = PackingEnv(current_layer, query_workload, current_layer_level)
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = global_agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            total_reward += reward
            # # 应用论文终止条件
            # # total——reward计算的不对，应该是前后两次的差值
            # if total_reward <= -env.N or done:
            #     break
            if next_state is not None:
                state = next_state
            else:
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

    print(f"\n[Training Phase] Total Levels Trained: {current_layer_level}")
    return global_agent, training_upper_layers, current_layer_level

##########################################
# 单层训练函数：训练 RL 代理以优化当前层的 packing 策略
##########################################
def train_single_level(current_layer, query_workload, current_layer_level, agent, epochs=500):
    """优化的单层训练函数"""
    if len(query_workload) == 0:
        raise ValueError("Query workload is empty")
    if len(current_layer) == 0:
        raise ValueError("Current layer nodes are empty")

    total_rewards = []
    loss_history = []
    q_values = []

    # 可选：使用tqdm显示进度条
    try:
        from tqdm import tqdm
        epoch_iterator = tqdm(range(epochs), desc=f"Training Level {current_layer_level + 1}")
    except ImportError:
        epoch_iterator = range(epochs)

    for epoch in epoch_iterator:
        env = PackingEnv(current_layer, query_workload, current_layer_level)
        state = env.reset()
        total_reward = 0
        done = False
        epoch_loss = []
        epoch_max_q = []
        epoch_min_q = []

        # 环境交互和训练循环
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)

            total_reward += reward

            # 存储经验并训练
            if next_state is not None:
                agent.store_transition(state, action, reward, next_state)
                loss, max_q, min_q = agent.update_model()
                if loss != 0.0:
                    epoch_loss.append(loss)
                    epoch_max_q.append(max_q)
                    epoch_min_q.append(min_q)
                state = next_state
            else:
                break

        # 在每个epoch结束后更新探索率
        agent.update_epsilon()

        # 记录统计量
        avg_loss = np.mean(epoch_loss) if epoch_loss else 0.0
        avg_max_q = np.mean(epoch_max_q) if epoch_max_q else 0.0
        avg_min_q = np.mean(epoch_min_q) if epoch_min_q else 0.0
        loss_history.append(avg_loss)
        q_values.append((avg_max_q, avg_min_q))
        total_rewards.append(total_reward)

        # 每10个epoch打印一次信息
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1} | ε={agent.epsilon:.3f} | Loss: {avg_loss:.2f} | "
                  f"Q(max/min): {avg_max_q:.2f}/{avg_min_q:.2f} | Reward: {total_reward:.2f}")


        # print(f"Epoch {epoch + 1} | ε={agent.epsilon:.3f} | Total Reward: {total_reward:.2f}")
        # 添加监控曲线绘制
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(loss_history, 'b-')
    # plt.title("Training Loss")
    # plt.subplot(1, 2, 2)
    # plt.plot([q[0] for q in q_values], 'r-', label='Max Q')
    # plt.plot([q[1] for q in q_values], 'g-', label='Min Q')
    # plt.title("Q-Value Range")
    # plt.legend()
    # plt.show()

    return agent, total_rewards

##########################################
# 重跑构造阶段：利用训练阶段学到的 RL 模型，重新跑一遍所有数据以生成最终树结构
##########################################
def final_tree_construction(bottom_nodes, query_workload, global_agent):
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

    while True:
        print(f"\n[Final Construction Phase] Processing Level {level + 1} ...")
        # 在重跑阶段采用贪婪策略：设置 ε = 0

        env = PackingEnv(current_layer, query_workload, current_layer_level)
        state = env.reset()
        total_reward = 0
        done = False

        while True:
            valid_actions = env.get_valid_actions()
            action = global_agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            total_reward += reward

            if next_state is not None:
                state = next_state
            else:
                break

        # 打印当前层级打包后的上层节点信息
        print(f"\nAfter final construction of Level {current_layer_level + 1}, upper_layer nodes:")
        print_layer_nodes(env.upper_layer, current_layer_level + 1)

        filter_upper_layer = [node for node in env.upper_layer if node['MBR'] is not None]
        final_tree_structure.append(filter_upper_layer.copy())
        current_layer = env.upper_layer  # 下一层输入为本层所有上层节点
        current_layer_level += 1
        level += 1
        # 终止条件还需要更改

        if len(filter_upper_layer) == 1:
            print("Reached one non-empty node. Stopping final construction.")
            break

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

        for node in current_layer:
            if node['MBR'] is None:  # 跳过空节点
                continue

            # 收集实际的子节点对象
            real_children = []
            for child_idx in node['children']:
                # 确保索引有效
                if isinstance(child_idx, int) and child_idx < len(lower_layer):
                    child_node = lower_layer[child_idx]
                    if child_node['MBR'] is not None:  # 只添加非空节点
                        real_children.append(child_node)

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

