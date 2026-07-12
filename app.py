import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt


def main():
    env = gym.make("CartPole-v1")

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden_size = 256

    model = torch.nn.Sequential(
        torch.nn.Linear(obs_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, n_actions),
        torch.nn.Softmax(dim=-1)
    )

    print("Observation size:", obs_size)
    print("Number of actions:", n_actions)
    print("\nModel Architecture:")
    print(model)

    learning_rate = 0.003
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    learning_rate = 0.003
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    Horizon = 500
    MAX_TRAJECTORIES = 500
    gamma = 0.99
    score = [] 

    for trajectory in range(MAX_TRAJECTORIES):
        curr_state, info = env.reset()
        done = False
        transitions = [] 
        
        for t in range(Horizon):
            act_prob = model(torch.from_numpy(curr_state).float())
            action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
            prev_state = curr_state
            curr_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transitions.append((prev_state, action, t+1)) 
            if done: 
                break
        score.append(len(transitions))
        reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,)) 

        batch_Gvals =[]
        for i in range(len(transitions)):
            new_Gval=0
            power=0
            for j in range(i,len(transitions)):
                new_Gval=new_Gval+((gamma**power)*reward_batch[j]).numpy()
                power+=1
            batch_Gvals.append(new_Gval)
        expected_returns_batch=torch.FloatTensor(batch_Gvals)
        
        
        expected_returns_batch /= expected_returns_batch.max()

        state_batch = torch.Tensor([s for (s,a,r) in transitions]) 
        action_batch = torch.Tensor([a for (s,a,r) in transitions]) 

        pred_batch = model(state_batch) 
        prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() 
        
        loss = - torch.sum(torch.log(prob_batch) * expected_returns_batch) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if trajectory % 50 == 0 and trajectory>0:
                print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, np.mean(score[-50:-1])))


    score = np.array(score)
    avg_score = running_mean(score)

    plt.figure(figsize=(15,7))
    plt.ylabel("Trajectory Duration",fontsize=12)
    plt.xlabel("Training Epochs",fontsize=12)
    plt.plot(score, color='gray' , linewidth=1)
    plt.plot(avg_score, color='blue', linewidth=3)
    plt.scatter(np.arange(score.shape[0]),score, color='green' , linewidth=0.3)

    plt.show()
    env.close()


def running_mean(x):
    N=50
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


if __name__ == "__main__":
    main()