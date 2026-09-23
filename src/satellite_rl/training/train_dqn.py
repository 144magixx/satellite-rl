from __future__ import annotations
from satellite_rl.paths import project_path
"""Run the original DQN training loop, logging, plotting, and model export."""

import os
import math
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

from satellite_rl.agents.dqn import DQNAgent, EPISODES, EPS_START, EPS_END, EPS_DECAY
from satellite_rl.envs.satellite import SatelliteFreqState, BeamInfo, CurrentBeamState, Env

# ---------------- Logger ---------------- #
class TrainLogger:
    def __init__(self):
        os.makedirs(project_path("dqn_group_log"), exist_ok=True)
        self.log_file = os.path.join(project_path("dqn_group_log"), f'dqn_log_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
        self.log_fields = [
            'episode','beam_id','satisfaction','satisfaction_increment','total_satisfaction',
            'success_slots','spectral_efficiency_avg','power','is_interfere','out','occupy',
            'required_rate','location','group','freq','origin_slots','sgm_rate','interfere_in_range'
        ]
        with open(self.log_file,'w',newline='') as f:
            csv.DictWriter(f,fieldnames=self.log_fields).writeheader()
    def log_step(self,ep:int,info:dict):
        data={
            'episode':ep,
            'beam_id':info['beam_id'],
            'satisfaction':float(f"{info['current_satisfaction']:.2f}"),
            'satisfaction_increment':float(f"{info['satisfaction_increment']:.2f}"),
            'total_satisfaction':float(f"{info['total_satisfaction']:.2f}"),
            'success_slots':info['success_slots'],
            'spectral_efficiency_avg':float(f"{info['spectral_efficiency_avg']:.2f}"),
            'power':float(f"{info['power']:.2f}"),
            'is_interfere':info['is_interfere'],
            'out':info['out'],
            'occupy':info['occupy'],
            'required_rate':info['required_rate'],
            'location':info['location'],
            'group':info['group'],
            'freq':info['freq'],
            'origin_slots':info['origin_slots'],
            'sgm_rate': float(np.asarray(info['sgm_rate']).squeeze()),
            'interfere_in_range':info['interfere_in_range']
        }
        with open(self.log_file,'a',newline='') as f:
            csv.DictWriter(f,fieldnames=self.log_fields).writerow(data)

# ---------------- Plot utils ---------------- #

def _smooth(arr,win):
    return [np.mean(arr[max(0,i-win):i+1]) for i in range(len(arr))]

def plot_curve(values,label,window,filename):
    plt.figure(figsize=(14,6))
    plt.plot(values,alpha=0.3,label=f'Raw {label}')
    plt.plot(_smooth(values,window),linewidth=2,label=f'Smooth {label}(w={window})')
    plt.title(f'DQN Episode {label}')
    plt.xlabel('Episode');plt.ylabel(label)
    plt.grid(True,ls='--',alpha=0.6);plt.legend();plt.tight_layout()
    os.makedirs(project_path('dqn_group_fig'),exist_ok=True)
    plt.savefig(os.path.join(project_path('dqn_group_fig'),filename),dpi=300)
    plt.show()

# ---------------- Env builder ---------------- #

def build_env():
    beam_info=BeamInfo();sat=SatelliteFreqState();curr=CurrentBeamState(beam_info.beam_info)
    return Env(sat,beam_info,curr),beam_info

# ---------------- Training ---------------- #

def main():
    logger=TrainLogger();agent=DQNAgent();eps=EPS_START
    reward_hist=[];sat_hist=[]
    for ep in range(EPISODES):
        env,beam_info=build_env();state=env.reset(beam_info);done=False;ep_ret=0;ep_sat=0
        while not done:
            f,slt,pwr=agent.choose_action(state,env,eps)
            env_action=[env.beam_idx%8,
                         f,
                         slt+1,
                         10*math.log10((pwr+1)*5)]
            next_state,rew,done,info=env.step(env_action)
            agent.remember(state,f,slt,pwr,rew,next_state,done)
            agent.update();agent.steps+=1;state=next_state;ep_ret+=rew;ep_sat=info['total_satisfaction']
            logger.log_step(ep,info)
        eps=max(EPS_END,eps-EPS_DECAY)
        reward_hist.append(ep_ret);sat_hist.append(ep_sat)
        print(f"Ep{ep:4d} R:{ep_ret:7.2f} Sat:{ep_sat:6.2f} ε:{eps:.3f} Buf:{len(agent.buffer):5d} Loss:{agent.last_loss if agent.last_loss is not None else 0:.4f}")
    # plots
    timestamp=datetime.now().strftime('%Y%m%d%H%M')
    plot_curve(reward_hist,'Reward',20,f'reward_{timestamp}.png')
    plot_curve(sat_hist,'Satisfaction',20,f'sat_{timestamp}.png')
    # save models
    os.makedirs(project_path('saved_dqn_models'),exist_ok=True)
    torch.save(agent.online.state_dict(),project_path('saved_dqn_models/dqn_online.pth'))
    torch.save(agent.target.state_dict(),project_path('saved_dqn_models/dqn_target.pth'))
    os.makedirs(project_path('DQN_full_data'), exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d%H%M')
    np.savez(project_path(f'DQN_full_data/metrics_{stamp}.npz'), returns=np.array(reward_hist), satisfactions=np.array(sat_hist))
    print(f"Saved metrics to DQN_full_data/metrics_{stamp}.npz")

if __name__=='__main__':
    main()
