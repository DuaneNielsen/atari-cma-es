import gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from mentalitystorm import config, Storeable
import numpy as np

device = config.device()
visuals = Storeable.load('D:\data\models\GM53H301W5YS38XH').to(device)

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


env = gym.make('SpaceInvaders-v4')
policy_nets = []
weights = []
scores = []
sample_size = 200
epochs = 200
rollouts = 1

for net in range(sample_size):
    policy_nets.append(nn.Sequential(nn.Linear(16, 6), nn.Softmax(dim=1)).double())
    scores.append(0.0)
    weight = policy_nets[net]._modules['0'].weight.data
    weights.append(weight)

weights_t = torch.stack(weights)
mu = weights_t.mean(0)
sigma = torch.sqrt(weights_t.var(0))

for epoch in range(epochs):
    for net in range(sample_size):
        distrib = Normal(mu, sigma)
        weights[net] = distrib.sample((1,)).view((6, 16))
        policy_nets[net]._modules['0'].weight.data = weights[net].data

    for i, net in enumerate(policy_nets):
        for i_episode in range(rollouts):
            env.reset()
            for t in range(5000):
                screen = env.render(mode='rgb_array')
                obs = torch.tensor(screen).cuda().float()
                obs = obs.permute(2, 0, 1).unsqueeze(0)
                latent, sigma = visuals.encoder(obs)
                latent = latent.cpu().double().squeeze(3).squeeze(2)
                action = net(latent)
                _, the_action = action.max(1)
                env.render()
                raw_observation, reward, done, info = env.step(the_action.item())
                scores[i] += reward
                if done:
                    #print("Episode finished after {} timesteps".format(t+1))
                    break

    scores_np = np.array(scores)
    score_mean = scores_np.mean()
    score_var = scores_np.var()
    best, rest = top_25_percent(scores)
    best_score = scores[best[0].item()]
    for i, elem in enumerate(scores):
        scores[i] = 0.0
    weights_t = torch.stack(weights)
    mu = weights_t[best].mean(0)
    sigma = torch.sqrt(weights_t.var(0))
    print('SCORE: mean %f, variance %f, best %f, CME: mean %f, sigma %f' % (score_mean, score_var, best_score, mu.view(6*16).mean(0), sigma.view(6*16).mean()))
    filename = 'best_model%d'%epoch
    with open(filename, 'wb') as f:
        torch.save(policy_nets[best[0].item()], f)
    filename = 'musigma_%d' % epoch
    with open(filename, 'wb') as f:
        import pickle
        pickle.dump((mu.numpy(), sigma.numpy()), f)



