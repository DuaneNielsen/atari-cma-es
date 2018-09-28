import gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def top_25_percent(scores, higher_is_better=True):
    """
    Calculates the top 25 best scores
    :param scores: a list of the scores
    :return: a longtensor with indices of the top 25 scores
    """
    indexed = [(i, s) for i, s in enumerate(scores)]
    indexed = sorted(indexed, key=lambda score: score[1], reverse=higher_is_better)
    best = [indexed[i][0] for i in range(len(indexed)//4)]
    rest = [indexed[i][0] for i in range(len(indexed)//4+1, len(indexed))]
    return torch.tensor(best), torch.tensor(rest)


env = gym.make('CartPole-v1')
policy_nets = []
weights = []
scores = []
sample_size = 40
epochs = 20
rollouts = 1

for net in range(sample_size):
    policy_nets.append(nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()).double())
    scores.append(0.0)
    weight = policy_nets[net]._modules['0'].weight.data
    weights.append(weight)

weights_t = torch.stack(weights)
mu = weights_t.mean(0)
sigma = torch.sqrt(weights_t.var(0))

for epoch in range(epochs):
    for net in range(sample_size):
        distrib = Normal(mu, sigma)
        weights[net] = distrib.sample((1,)).view((1,4))
        policy_nets[net]._modules['0'].weight.data = weights[net].data

    for i, net in enumerate(policy_nets):
        for i_episode in range(rollouts):
            raw_observation = env.reset()
            for t in range(100):
                obs = torch.tensor(raw_observation)
                action = net(obs)
                action = int(round(action.item()))
                env.render()
                raw_observation, reward, done, info = env.step(action)
                scores[i] += reward
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

    print(scores)
    best, rest = top_25_percent(scores)
    for i, elem in enumerate(scores):
        scores[i] = 0.0
    weights_t = torch.stack(weights)
    mu = weights_t[best].mean(0)
    sigma = torch.sqrt(weights_t.var(0))
    with open('best_model', 'wb') as f:
        torch.save(policy_nets[best[0].item()], f)


