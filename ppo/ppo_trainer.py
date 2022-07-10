import torch
from torch.nn import functional as F
import numpy as np

class PPOTrainer:
    def __init__(
        self,
        policy,
        optimizer,
        buffer_size,
        batch_size,
        device,
        epochs,
        ent_coef,
        vf_coef,
        clip_range,
        max_grad_norm,
        target_kl=None,
        clip_range_vf=None,
    ):
        assert buffer_size % batch_size == 0, f"buffersize {buffer_size} should be a multiple of batchsize {batch_size}"

        self.policy = policy
        self.optimizer = optimizer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.clip_range_vf = clip_range_vf

    def optimize(
        self,
        observations,
        old_log_probs,
        actions,
        old_values,
        advantages,
        returns,
    ):
        """
        also see stable baselines 3 train method in ppo.py
        """
        # todo: observations need to be shaped differently for e.g. CNN policy
        observations = observations.reshape(-1, observations.shape[-1])
        old_log_probs = torch.tensor(old_log_probs.flatten()).to(self.device)
        actions = torch.tensor(actions.flatten(), dtype=torch.int64).to(self.device)
        old_values = torch.tensor(old_values.flatten()).to(self.device)
        advantages = torch.tensor(advantages.flatten()).to(self.device)
        returns = torch.tensor(returns.flatten()).to(self.device)

        continue_training = True
        for epoch in range(self.epochs):
            batch_ind = np.random.permutation(self.buffer_size)
            start_idx = 0
            while start_idx < self.buffer_size:
                ind = batch_ind[start_idx: start_idx + self.batch_size]
                cur_log_prob, cur_values, entropy = self.policy.evaluate_action(
                    torch.from_numpy(observations[ind]).to(self.device),
                    actions[ind],
                )
                cur_values = cur_values.flatten()
                cur_advantages = advantages[ind]

                # normalize advantages (introduces bias but stabilizes training)
                norm_advantages = (cur_advantages - cur_advantages.mean()) / (cur_advantages.std() + 1e-8)

                # ration between old and new policy
                ratio = torch.exp(cur_log_prob.flatten() - old_log_probs[ind])

                # clipped surrogate loss
                policy_loss_1 = norm_advantages * ratio
                policy_loss_2 = norm_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf:
                    cur_values = old_values[ind] + torch.clamp(
                        cur_values - old_values[ind], -self.clip_range_vf, self.clip_range_vf
                    )

                value_loss = F.mse_loss(returns[ind], cur_values)

                # maximize entropy to favor more uniform (explorative) policies
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Computing an approximate of the KL to stop training early
                # Necessary as even though we are clipping steps might be too large
                with torch.no_grad():
                    log_ratio = cur_log_prob - old_log_probs[ind]
                    # http://joschu.net/blog/kl-approx.html explains how we improve on basic sample estimation
                    # of kl_div by adding expectation 0 and negatively correlated variable
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    print(f"Early stopping at step {epoch} du to reaching max kl: {approx_kl_div:.2f}")

                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                start_idx += self.batch_size
            if not continue_training:
                break
        return loss, policy_loss, value_loss
