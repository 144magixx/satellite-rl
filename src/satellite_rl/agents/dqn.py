from __future__ import annotations
"""Factorised DQN networks, replay buffer, and parameter updates."""

import random
from collections import deque
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from satellite_rl.envs.satellite import Env

# ---------------- Hyper‑params ---------------- #
STATE_DIM = 204
FREQ_DIM  = 100
SLOT_DIM  = 10
PWR_DIM   = 10
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 256
BUFFER_CAPACITY = 20000
TARGET_UPDATE_FREQ = 500  # steps
EPISODES = 4000
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY = (EPS_START - EPS_END) / (EPISODES/2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- ReplayBuffer ---------------- #
class ReplayBuffer:
    def __init__(self,cap:int):
        self.buf:deque=deque(maxlen=cap)
    def push(self,trans):
        self.buf.append(trans)
    def sample(self,batch):
        sample=random.sample(self.buf,batch)
        s,f,slt,pwr,r,s_,d=map(np.array,zip(*sample))
        to=lambda x:torch.tensor(x,device=DEVICE)
        return to(s).float(),to(f).long(),to(slt).long(),to(pwr).long(),to(r).float(),to(s_).float(),to(d).float()
    def __len__(self):
        return len(self.buf)

# ---------------- Q‑Network ---------------- #
class FactorisedDQN(nn.Module):
    def __init__(self,state_dim:int):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(state_dim,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU())
        self.freq_head=nn.Linear(256,FREQ_DIM)
        self.slot_head=nn.Linear(256,SLOT_DIM)
        self.pwr_head =nn.Linear(256,PWR_DIM)
    def forward(self,x:torch.Tensor):
        h=self.shared(x)
        return self.freq_head(h),self.slot_head(h),self.pwr_head(h)  # tuple of logits (batch,10)

# ---------------- DQN Agent ---------------- #
class DQNAgent:
    def __init__(self):
        self.online=FactorisedDQN(STATE_DIM).to(DEVICE)
        self.target=FactorisedDQN(STATE_DIM).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.opt=torch.optim.Adam(self.online.parameters(),lr=LEARNING_RATE)
        self.buffer=ReplayBuffer(BUFFER_CAPACITY)
        self.steps=0
        self.last_loss: float | None = None  # store latest critic loss

    def _gather_q(self,qs,f,slt,pwr):
        q_f=qs[0].gather(1,f.view(-1,1)).squeeze(1)
        q_s=qs[1].gather(1,slt.view(-1,1)).squeeze(1)
        q_p=qs[2].gather(1,pwr.view(-1,1)).squeeze(1)
        return q_f+q_s+q_p

    def choose_action(self,state:np.ndarray,env:Env,eps:float)->Tuple[int,int,int]:
        #mask=env.get_action_mask()
        state_t=torch.tensor(state,dtype=torch.float32,device=DEVICE).unsqueeze(0)
        if random.random()<eps:
            #valid_freqs=[i for i,m in enumerate(mask) if m]
            f=random.randrange(FREQ_DIM)
            s=random.randrange(SLOT_DIM)
            p=random.randrange(PWR_DIM)
            return f,s,p
        with torch.no_grad():
            qf,qs,qp=self.online(state_t)
            qf,qs,qp=qf.squeeze(0),qs.squeeze(0),qp.squeeze(0)
            # invalid=[i for i,m in enumerate(mask) if not m]
            # if invalid:
            #     qf[torch.tensor(invalid,device=DEVICE)]=-1e9
            return int(torch.argmax(qf)),int(torch.argmax(qs)),int(torch.argmax(qp))

    def update(self):
        if len(self.buffer)<BATCH_SIZE:
            return
        s,f,slt,pwr,r,s_,d=self.buffer.sample(BATCH_SIZE)
        q_curr=self._gather_q(self.online(s),f,slt,pwr)
        with torch.no_grad():
            f_next=torch.argmax(self.online(s_)[0],dim=1)
            slt_next=torch.argmax(self.online(s_)[1],dim=1)
            pwr_next=torch.argmax(self.online(s_)[2],dim=1)
            q_next=self._gather_q(self.target(s_),f_next,slt_next,pwr_next)
            target=r+GAMMA*(1-d)*q_next
        loss=F.mse_loss(q_curr,target)
        self.opt.zero_grad();loss.backward();self.opt.step()
        self.last_loss=float(loss.item())
        if self.steps%TARGET_UPDATE_FREQ==0:
            self.target.load_state_dict(self.online.state_dict())

    def remember(self,*tr):
        self.buffer.push(tr)


