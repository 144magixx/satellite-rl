from satellite_rl.paths import project_path
import torch
import numpy as np
from datetime import datetime
import os

from satellite_rl.agents.ppo_direct import PPO
from satellite_rl.training import ppo_direct_utils as rl_utils
from satellite_rl.envs.satellite import SatelliteFreqState, BeamInfo, Env, CurrentBeamState


def main():
    beam_info = BeamInfo()
    satellite_freq_state = SatelliteFreqState()
    current_beam_state = CurrentBeamState(beam_info.beam_info)
    actor_lr = 1e-4  # 策略网络学习率
    critic_lr = 1e-3  # 价值网络学习率
    num_episodes = 4000  # 训练的总回合数
    gamma = 0.99  # 折扣因子
    lmbda = 0.95  # GAE 参数
    epochs = 4  # 每次更新的轮数
    eps = 0.2  # PPO 截断范围
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    # 创建环境

    env = Env(satellite_freq_state, beam_info, current_beam_state)
    env.reset(beam_info)  # 设置随机种子
    state_dim = 204
    agent = PPO(state_dim=state_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                lmbda=lmbda,
                epochs=epochs,
                eps=eps,
                gamma=gamma,
                device=device)  # 初始化 PPO 算法
    # 开始训练
    return_list, satisfaction_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)  # 使用工具函数训练
    rl_utils.plot_reward(return_list)
    rl_utils.plot_satisfaction(satisfaction_list)
    print("Training completed!")
    model_dir = project_path("saved_group_no_freq_models")
    os.makedirs(model_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    actor_path = os.path.join(model_dir, f"ppo_group_actor_{timestamp}.pth")
    critic_path = os.path.join(model_dir, f"ppo_group_critic_{timestamp}.pth")

    # 保存完整模型参数（包含网络结构和参数）
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),

    }, os.path.join(model_dir, f"full_group_model_{timestamp}.pth"))

    # 同时单独保存参数方便后续加载
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)

    print(f"Models saved to {model_dir} with timestamp {timestamp}")
    os.makedirs(project_path('PPO_data'), exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d%H%M')
    np.savez(project_path(f'PPO_data/metrics_{stamp}.npz'), returns=np.array(return_list), satisfactions=np.array(satisfaction_list))
    print(f"Saved metrics to PPO_data/metrics_{stamp}.npz")

if __name__ == '__main__':
    main()