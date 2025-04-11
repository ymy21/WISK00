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
    def __init__(self, current_layer_nodes, query_workload, current_layer_level, w1=0.1, w2=0.5):
        # 成本参数：w1 表示扫描一个节点的固定成本，w2 表示验证该节点中对象的成本
        self.w1 = w1
        self.w2 = w2
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
            node = {'layer': self.current_layer_level+1, 'labels': [], 'children': [], 'MBR': None}
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
        target['children'].append(self.current_step)

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
        # 计算并应用奖励缩放，提高训练稳定性
        raw_reward = prev_access - new_access
        reward = raw_reward * 1.0  # 缩放因子，增加梯度信号强度

        # 更新累积奖励，用于终止条件判断
        self.total_reward += reward

        # 应用论文终止条件：如果累积奖励不大于-N，提前终止
        if self.total_reward <= -self.N:
            done = True

        return next_state, reward, done

    # 不匹配的也得访问检查
    # def _calculate_access_cost(self):
    #     total = 0
    #     for query in self.query_workload:
    #         # 上层部分：对于每个非空上层节点，都要访问，加上匹配时访问 children 的额外开销
    #         upper_cost = 0
    #         for node in self.upper_layer:
    #             if node['MBR'] is None:
    #                 continue
    #             # 每个非空节点的基本访问成本
    #             upper_cost += 1
    #             # 如果节点与查询匹配，则额外加上 children 数
    #             if self._mbr_intersect(node['MBR'], query) and any(kw in node['labels'] for kw in query['keywords']):
    #                 upper_cost += len(node['children'])
    #
    #         # 下层部分：每个未打包的非空节点都要访问
    #         lower_cost = 0
    #         for i in range(self.current_step, len(self.current_layer)):
    #             if self.current_layer[i]['MBR'] is not None:
    #                 lower_cost += 1
    #
    #         total += (upper_cost + lower_cost)
    #     return total / len(self.query_workload)
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
                query_access += self.w1  # 基本访问成本

                # 如果节点与查询匹配，还需要访问其子节点
                if self._mbr_intersect(node['MBR'], query) and any(kw in node['labels'] for kw in query['keywords']):
                    # 每个子节点的访问成本
                    query_access += len(node['children']) * self.w2

            # 下层部分：未打包的节点
            for i in range(self.current_step, len(self.current_layer)):
                if self.current_layer[i]['MBR'] is not None:
                    query_access += self.w1  # 未打包节点的访问成本

                    # 如果节点与查询匹配，计算验证成本
                    if self._mbr_intersect(self.current_layer[i]['MBR'], query) and any(
                            kw in self.current_layer[i]['labels'] for kw in query['keywords']):
                        query_access += self.w2  # 验证成本

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
            nn.Linear(hidden_dim, hidden_dim),  # 第三隐藏层（论文要求）
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

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # 添加步数计数和目标网络更新频率
        self.steps_done = 0
        self.target_update_freq = 10  # 每10步更新一次

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

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0, 0.0 # 返回loss, max_q, min_q
        # 计数更新步数
        self.steps_done += 1

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)

        # 处理非终止状态
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool, device=device)
        non_final_next_states = torch.FloatTensor( np.array([s for s in next_states if s is not None])).to(device)

        # 计算当前 Q 值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        next_q = torch.zeros(self.batch_size, device=device)
        if non_final_mask.sum() > 0:
            next_q[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 每C步更新一次目标网络，按论文要求
        if self.steps_done % self.target_update_freq == 0:
            self.soft_update_target()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # 添加Q值监控
        with torch.no_grad():
            max_q = current_q.max().item()
            min_q = current_q.min().item()
        return loss.item(), max_q, min_q

    def soft_update_target(self, tau=0.001):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)


##########################################
# 分层训练阶段：每层训练只保留当前层与上层的信息
# 将训练阶段与重跑构造阶段分离，训练阶段保存每一层训练得到的 RL 模型
##########################################
def hierarchical_packing_training(bottom_nodes, query_workload, max_level=10):
    current_layer = bottom_nodes
    current_layer_level = 0
    level_agents = []         # 保存每层训练得到的 RL 代理
    training_upper_layers = []  # 保存每层的中间上层节点（训练阶段结果）
    all_rewards = []          # 保存每层训练时的奖励曲线

    for level in range(max_level):
         # 初始化前一层的非空节点数
        prev_non_empty = sum(1 for node in current_layer if  node['MBR'] is not None)
        print(f"Level {level + 1}: Non-empty nodes = {prev_non_empty}")

        if prev_non_empty == 1:
            print(f"Reached one non-empty node. Stopping training.")
            break

        print(f"\n[Training Phase] Level {level + 1} Training...")
        # 训练当前层：当前层节点与预先初始化的 N 个空上层节点组成固定状态
        agent, total_rewards = train_single_level(current_layer, query_workload, current_layer_level)
        all_rewards.append(total_rewards)
        level_agents.append(agent)

        # 用当前训练好的代理对当前层数据进行一次模拟打包（依然使用 ε-greedy 策略）
        env = PackingEnv(current_layer, query_workload, current_layer_level)
        state = env.reset()
        total_reward = 0
        done = False

        while True:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            total_reward += reward
            # 应用论文终止条件
            # total——reward计算的不对，应该是前后两次的差值
            if total_reward <= -env.N or done:
                break
            state = next_state

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

        # # 终止条件3 ：上层空节点只剩一个
        # if current_non_empty == 1:
        #     print(f"Reached one non-empty node. Stopping training.")
        #     break

        prev_non_empty = current_non_empty

    # 绘制各层训练奖励曲线
    n_levels = len(all_rewards)
    fig, axs = plt.subplots(n_levels, 1, figsize=(10, 4 * n_levels))
    if n_levels == 1:
        axs = [axs]
    for i, rewards in enumerate(all_rewards):
        axs[i].plot(range(1, len(rewards) + 1), rewards, marker='o', label=f"Level {i + 1}")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Total Reward")
        axs[i].set_title(f"Convergence Curve for Level {i + 1}")
        axs[i].legend()
        axs[i].grid()
    plt.tight_layout()
    plt.show()

    print(f"\n[Training Phase] Total Levels Trained: {current_layer_level}")
    return level_agents, training_upper_layers, current_layer_level

##########################################
# 单层训练函数：训练 RL 代理以优化当前层的 packing 策略
##########################################
def train_single_level(current_layer, query_workload, current_layer_level, epochs=180):
    # 添加数据验证
    if len(query_workload) == 0:
        raise ValueError("Query workload is empty")
    if len(current_layer) == 0:
        raise ValueError("Current layer nodes are empty")
    m = len(query_workload)
    N = len(current_layer)
    state_dim = (m + 1) * N + m
    action_dim = N
    agent = DQNAgent(state_dim, action_dim)

    total_rewards = []  # 记录每轮训练的总奖励
    loss_history = []  # 新增loss记录
    q_values = []  # 新增Q值记录

    for epoch in range(epochs):
        env = PackingEnv(current_layer, query_workload, current_layer_level)
        state = env.reset()
        total_reward = 0
        done = False
        epoch_loss = []
        epoch_max_q = []
        epoch_min_q = []

        while not done:
            valid_actions = env.get_valid_actions() # 获取布尔掩码
            action = agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)

            total_reward += reward
            # if total_reward <= -env.N:  # 论文条件
            #     done = True

            agent.store_transition(state, action, reward, next_state)
            loss, max_q, min_q = agent.update_model()
            if loss != 0.0:
                epoch_loss.append(loss)
                epoch_max_q.append(max_q)
                epoch_min_q.append(min_q)
            if next_state is not None:
                state = next_state
            else:
                break
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
def final_tree_construction(bottom_nodes, query_workload, level_agents):
    current_layer = bottom_nodes
    filter_current_layer =  [node for node in current_layer if  node['MBR'] is not None]
    final_tree_structure = [filter_current_layer.copy()]  # 保存完整层级结构 (第0层 # 每层的最终上层节点（固定 N 个）
    level = 0
    current_layer_level = 0
    for agent in level_agents:
        print(f"\n[Final Construction Phase] Processing Level {level + 1} ...")
        # 在重跑阶段采用贪婪策略：设置 ε = 0
        agent.epsilon = 0.0
        env = PackingEnv(current_layer, query_workload, current_layer_level)
        state = env.reset()
        total_reward = 0
        done = False

        while True:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 应用论文终止条件
            if total_reward <= -env.N or done:
                break
            state = next_state

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
    return final_tree_structure
##########################################
# build_nested_tree 通过层级映射将重跑构造的索引结构转换为实际节点对象
##########################################
def build_nested_tree(tree_structure):
    """
    假设 tree_structure 是 hierarchical_packing() 返回的分层列表，
    且最后一层只有一个节点（树的根节点）。
    则该函数将返回这个根节点，从而形成嵌套的树结构。
    """
    if not tree_structure:
        return None
        # 从底层向上逐层建立索引映射
    for layer_idx in range(len(tree_structure) - 1, 0, -1):
        current_layer = tree_structure[layer_idx]
        lower_layer = tree_structure[layer_idx - 1]

        for node in current_layer:
            # 将children中的索引替换为下层实际节点
            node['children'] = [lower_layer[idx] for idx in node['children']]

    # 返回顶层根节点
    root_level = tree_structure[-1]
    if len(root_level) != 1:
        print("Warning: the tree_structure has more than one root.")
        return root_level
    return root_level[0]

def print_layer_nodes(nodes, layer_level):
    print(f"----- Layer {layer_level} Nodes Detail -----")
    for idx, node in enumerate(nodes):
        children_count = len(node['children'])
        labels = node['labels']
        mbr = node.get('MBR')
        print(f"\nNode {idx}: children_count = {children_count}, labels = {labels}, MBR = {mbr}")


