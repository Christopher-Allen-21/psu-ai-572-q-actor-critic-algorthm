import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt


def main():
    env = gym.make("Acrobot-v1")

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden_size = 128

    # Actor: pi(a | s)
    actor_model = torch.nn.Sequential(
        torch.nn.Linear(obs_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, n_actions),
        torch.nn.Softmax(dim=-1)
    )

    # Q Critic: Q(s, a) for every action
    critic_model = torch.nn.Sequential(
        torch.nn.Linear(obs_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, n_actions)
    )

    print("Observation size:", obs_size)
    print("Number of actions:", n_actions)

    print("\nActor Model:")
    print(actor_model)

    print("\nQ-Critic Model:")
    print(critic_model)

    # Set a faster learning rate for Critic as the Actor is updated using the Critic's Q-values
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.0005

    actor_optimizer = torch.optim.Adam(
        actor_model.parameters(),
        lr=actor_learning_rate
    )

    critic_optimizer = torch.optim.Adam(
        critic_model.parameters(),
        lr=critic_learning_rate
    )

    HORIZON = 500
    MAX_TRAJECTORIES = 1000
    gamma = 0.99

    # Scaling the reward because Acrobot returns -1 on almost every step. This keeps Q-values and policy gradients from becoming unnecessarily large
    reward_scale = 0.01

    # Added an entropy coefficient as without it the model kept collapsing
    # Entropy coefficient encourages continued exploration without replacing the Q actor update
    entropy_coefficient = 0.001

    scores = []
    rewards = []
    actor_losses = []
    critic_losses = []
    successes = []

    for trajectory in range(MAX_TRAJECTORIES):
        # Step 1: Initialize s and sample a ~ pi(a | s)
        curr_state, info = env.reset()
        curr_state_tensor = torch.from_numpy(curr_state).float()

        with torch.no_grad():
            action_probabilities = actor_model(curr_state_tensor)

        action = np.random.choice(
            np.arange(n_actions),
            p=action_probabilities.numpy()
        )

        terminated = False
        truncated = False

        total_reward = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        step_count = 0

        for t in range(HORIZON):
            # Step 2.1: Execute a and observe r and s'
            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            done = terminated or truncated
            scaled_reward = reward * reward_scale
            next_state_tensor = torch.from_numpy(next_state).float()

            # Step 2.2: Sample a' ~ pi(a' | s')
            next_action = None

            if not terminated:
                with torch.no_grad():
                    next_action_probabilities = actor_model(
                        next_state_tensor
                    )

                next_action = np.random.choice(
                    np.arange(n_actions),
                    p=next_action_probabilities.numpy()
                )

            # Compute Q(s, a) before either optimizer changes a model
            current_q_values = critic_model(curr_state_tensor)
            current_q_value = current_q_values[action]

            # Step 2.3: Update the actor
            #
            # Lecture notes used:
            #   theta <- theta + alpha * Q(s,a) * grad log pi(a|s)
            #
            # A state-value baseline is subtracted here:
            #   A(s,a) = Q(s,a) - sum_b pi(b|s)Q(s,b)
            #
            # This keeps the same on-policy Q Actor-Critic logic while
            # reducing variance. It is especially important in Acrobot
            # because all undiscounted Q-values are negative.
            # -----------------------------------------------------
            action_probabilities = actor_model(curr_state_tensor)

            with torch.no_grad():
                state_value = torch.sum(
                    action_probabilities * current_q_values.detach()
                )

                advantage = (
                    current_q_value.detach()
                    - state_value
                )

            selected_action_probability = action_probabilities[action]

            log_probability = torch.log(
                selected_action_probability + 1e-8
            )

            entropy = -torch.sum(
                action_probabilities
                * torch.log(action_probabilities + 1e-8)
            )

            actor_loss = (
                -advantage * log_probability
                - entropy_coefficient * entropy
            )

            actor_optimizer.zero_grad()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                actor_model.parameters(),
                max_norm=0.5
            )

            actor_optimizer.step()

            # Step 2.4: Compute the TD error and update Q
            # delta = r + gamma Q(s',a') - Q(s,a)
            with torch.no_grad():
                if terminated:
                    target_q_value = torch.tensor(
                        scaled_reward,
                        dtype=torch.float32
                    )
                else:
                    next_q_values = critic_model(next_state_tensor)
                    next_q_value = next_q_values[next_action]

                    target_q_value = (
                        scaled_reward
                        + gamma * next_q_value
                    )

            td_error = target_q_value - current_q_value
            critic_loss = 0.5 * td_error.pow(2)

            critic_optimizer.zero_grad()
            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                critic_model.parameters(),
                max_norm=1.0
            )

            critic_optimizer.step()

            total_reward += reward
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            step_count += 1

            if done:
                break

            # Step 2.5: Set a <- a' and s <- s'
            curr_state = next_state
            curr_state_tensor = next_state_tensor
            action = next_action

        scores.append(step_count)
        rewards.append(total_reward)

        actor_losses.append(
            total_actor_loss / max(step_count, 1)
        )

        critic_losses.append(
            total_critic_loss / max(step_count, 1)
        )

        success = terminated and not truncated
        successes.append(success)

        completed_trajectories = trajectory + 1

        if completed_trajectories % 50 == 0:
            average_score = np.mean(scores[-50:])
            average_reward = np.mean(rewards[-50:])
            success_rate = np.mean(successes[-50:]) * 100
            average_actor_loss = np.mean(actor_losses[-50:])
            average_critic_loss = np.mean(critic_losses[-50:])

            print(
                f"Trajectory {completed_trajectories}\t"
                f"Average Score: {average_score:.2f}\t"
                f"Average Reward: {average_reward:.2f}\t"
                f"Success Rate: {success_rate:.2f}%\t"
                f"Actor Loss: {average_actor_loss:.4f}\t"
                f"Critic Loss: {average_critic_loss:.4f}\t"
            )

    env.close()
    
    score = np.array(scores)
    generate_scatter_plot(score)


def generate_scatter_plot(score):
    avg_score = running_mean(score)

    plt.figure(figsize=(15, 7))
    plt.ylabel("Trajectory Duration", fontsize=12)
    plt.xlabel("Training Epochs", fontsize=12)

    plt.plot(
        np.arange(len(score)),
        score,
        color="gray",
        linewidth=1,
        label="Trajectory duration"
    )

    plt.scatter(
        np.arange(len(score)),
        score,
        color="green",
        linewidth=0.3,
        label="Individual trajectory"
    )

    if len(avg_score) > 0:
        avg_x = np.arange(
            49,
            49 + len(avg_score)
        )

        plt.plot(
            avg_x,
            avg_score,
            color="blue",
            linewidth=3,
            label="50-trajectory running mean"
        )

    plt.title("Q Actor-Critic Performance on Acrobot-v1")
    plt.legend()
    plt.tight_layout()

    print("\nRun complete. Close the plot window to exit.")
    plt.show()


def running_mean(x, window_size=50):
    if len(x) < window_size:
        return np.array([])

    kernel = np.ones(window_size) / window_size

    return np.convolve(
        x,
        kernel,
        mode="valid"
    )


if __name__ == "__main__":
    main()