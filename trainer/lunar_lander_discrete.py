import numpy as np

from policies.lunarlander_feed_forward import FeedForwardAgent
from ppo.rollout_buffer import RolloutBuffer
from ppo.mult_proc_env import MultProcEnvGym
from ppo.ppo_trainer import PPOTrainer
from helpers.gym_render_episode import render_episode

import torch

def CartPoleTrainer(
    train_iterations=1000000,
    episodes_in_parallel=16,
    iterations_per_optimization=2048,
    train_steps_per_optimization=10,
    batch_size=256,
    gamma=0.99,
    gae_lam=0.95,
    max_grad_norm=0.5,
):
    import wandb
    wandb.login()

    if torch.cuda.is_available():
        device = "cuda"
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"

    agent_policy = FeedForwardAgent(input_size=8, ortho_init=True).to(device)
    optimizer = torch.optim.Adam(agent_policy.parameters(), 0.0003, eps=1e-05)

    ppo_trainer=PPOTrainer(
        policy=agent_policy,
        optimizer=optimizer,
        buffer_size=iterations_per_optimization * episodes_in_parallel,
        batch_size=batch_size,
        device=device,
        epochs=train_steps_per_optimization,
        ent_coef=0.0,
        vf_coef=0.5,
        clip_range=0.2,
        max_grad_norm=max_grad_norm,
        target_kl=None,
        clip_range_vf=None,
    )

    mp_env = MultProcEnvGym(
        "LunarLander-v2",
        num_envs=episodes_in_parallel,
    )
    mp_env.seed(42)
    initial_obs = mp_env.reset()
    buffer = RolloutBuffer(
        episodes_in_parallel=episodes_in_parallel,
        buffer_size=iterations_per_optimization,
        obs_size=8,
    )
    buffer.reset(initial_obs)

    with wandb.init(project="PPO_CartPole_FeedForward_Agent") as run:
        run.config.optimizer = optimizer.__class__.__name__
        run.watch(agent_policy)
        acc_reward = np.zeros(episodes_in_parallel)
        for iteration in range(train_iterations):
            all_rewards = []
            for gen_step in range(iterations_per_optimization):
                game_maps = buffer.game_maps[gen_step]
                actions, log_probs, values = agent_policy.evaluate_observation(torch.from_numpy(game_maps).to(device))
                new_obs, env_reward, dones, info = mp_env.step(list(actions))
                buffer_done = buffer.add(
                    gen_step,
                    env_reward,
                    actions,
                    values.to("cpu").detach().numpy(),
                    log_probs.to("cpu").detach().numpy(),
                    dones,
                    new_obs,
                )
                for ep in range(episodes_in_parallel):
                    if not dones[ep]:
                        acc_reward[ep] += env_reward[ep]
                    else:
                        all_rewards.append(acc_reward[ep])
                        acc_reward[ep] = 0
            assert buffer_done
            _, _, last_values = agent_policy.evaluate_observation(torch.from_numpy(buffer.game_maps[-2]).to(device))

            advantages, returns = buffer.compute_gae(last_values.to("cpu").detach().numpy(), dones, gamma, gae_lam)
            loss, policy_loss, value_loss = ppo_trainer.optimize(
                observations = buffer.game_maps,
                old_log_probs = buffer.log_probs,
                old_values = buffer.values,
                actions = buffer.actions,
                advantages = advantages,
                returns = returns,
            )
            avg_reward = sum(all_rewards)/len(all_rewards)
            run.log({
                "avg_reward": avg_reward,
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
            })
            if avg_reward > 150:
                render_episode("LunarLander-v2", agent_policy)
            buffer.restart_from_last()

if __name__ == "__main__":
    CartPoleTrainer()
