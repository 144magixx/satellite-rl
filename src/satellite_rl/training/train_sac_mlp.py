from satellite_rl.agents.sac_mlp import ReplayBuffer, SAC
from satellite_rl.training.sac_runner import run_training


if __name__ == "__main__":
    run_training(SAC, ReplayBuffer)
