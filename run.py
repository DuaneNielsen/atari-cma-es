import gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from mentalitystorm import config, Storeable, Run, ImageViewer
from mentalitystorm.basemodels import MultiChannelAE
import torchvision.transforms as TVT
import mentalitystorm.transforms as tf
import numpy as np
from models import PolicyNet
from pathlib import Path
from collections import namedtuple

device = config.device()
#visuals = Storeable.load('D:\data\models\GM53H301W5YS38XH').to(device)
shot_encoder = Run.load_model(r'c:\data\runs\549\shots_v1\epoch0060.run').to(device=config.device())
player_encoder = Run.load_model(r'c:\data\runs\580\shots_v1\epoch0081.run').to(device=config.device())
visuals = MultiChannelAE()
visuals.add_ae(shot_encoder, [0, 2, 3])
visuals.add_ae(player_encoder, [1, 2, 3])

transforms = TVT.Compose([tf.CoordConv()])

view_latent = ImageViewer('latent', (320, 480))


def view_image(model, input, output):
    view_latent.update(output[0].data)

decode_viewer = ImageViewer('decoded', (320, 480))

def view_decode(model, input, output):
    image = model.decode(output)
    decode_viewer.update(image)

visuals.decode_ch_l = [[0],[1]]
visuals.register_forward_hook(view_decode)

def top_25_percent(scores, higher_is_better=True):
    """
    Calculates the top 25 best scores
    :param scores: a list of the scores
    :return: a longtensor with indices of the top 25 scores
    """
    indexed = [(i, s) for i, s in enumerate(scores)]
    indexed = sorted(indexed, key=lambda score: score[1], reverse=higher_is_better)
    num_winners = len(indexed) // 4
    num_winners = num_winners if num_winners > 0 else 1
    best = [indexed[i][0] for i in range(num_winners)]
    rest = [indexed[i][0] for i in range(num_winners+1, len(indexed))]
    return torch.tensor(best), torch.tensor(rest)


def flatten(net):
    w = []

    def _capture(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            w.append(m.weight.data)
            w.append(m.bias.data)

    net.apply(_capture)

    t = list(map(lambda x: x.view(-1), w))
    return torch.cat(t)


def restore(net, t):
    start = 0

    def _restore(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nonlocal start

            length = m.weight.data.numel()
            chunk = t[range(start, start + length)]
            m.weight.data = chunk.view(m.weight.data.shape)
            start += length

            length = m.bias.data.numel()
            chunk = t[range(start, start + length)]
            m.bias.data = chunk.view(m.bias.data.shape)
            start += length

    net.apply(_restore)


class CMA:
    def __init__(self, higher_is_better=True):

        self.weights = []
        self.scores = []
        self.higher_is_better = higher_is_better
        self.stats = []
        self.distrib = None

    def add(self, net, score, stats=None):
        self.weights.append(flatten(net))
        self.scores.append(score)
        if stats is not None:
            self.stats.append(stats)

    def rank_and_compute(self):
        w_t = torch.stack(self.weights)
        best, rest = top_25_percent(self.scores, self.higher_is_better)
        mu = w_t[best].mean(0)
        stdv = torch.sqrt(w_t.var(0))
        self.distrib = Normal(mu, stdv)

        self.print_scores()

        self.weights = []
        self.scores = []
        self.stats = []
        return best, rest

    def set_sample_weights(self, net):
        sample = self.distrib.sample((1,)).squeeze(0)
        restore(net, sample)

    def print_scores(self):
        if self.scores is not None:
            best, rest = top_25_percent(self.scores, self.higher_is_better)
            scores_np = np.array(self.scores)
            score_mean = scores_np.mean()
            score_var = scores_np.var()
            best_score = self.scores[best[0].item()]
            episode_steps = [d['episode_steps'] for d in self.stats]
            episode_steps_np = np.array(episode_steps)
            print('SCORE: mean %f, variance %f, best %f, ' \
                   'epi mean length %f, epi max len %f, ' \
                   'CME: mean %f, sigma %f' % (
                       score_mean, score_var, best_score,
                       episode_steps_np.mean(), episode_steps_np.max(),
                       self.distrib.mean.mean(), self.distrib.stddev.mean()))


env = gym.make('SpaceInvaders-v4')
policy_nets = []
cma = CMA()

episode_steps = []
sample_size = 300
epochs = 200
rollouts = 1

z_size = 32


for net in range(sample_size):
    policy_nets.append(PolicyNet((11, 8), 6).double())
    #policy_nets.append(nn.Sequential(nn.Linear(4, 2), nn.Softmax(dim=1)))
    episode_steps.append(0)


for epoch in range(epochs):
    for net in policy_nets:
        score = 0
        stats = {'episode_steps': 0}
        for i_episode in range(rollouts):
            raw_observation = env.reset()
            for t in range(5000):
                screen = env.render(mode='rgb_array')
                obs = torch.tensor(screen).cuda().float()
                obs = obs.permute(2, 0, 1)
                obs = transforms(obs)
                obs = obs.unsqueeze(0)
                latent = visuals(obs)
                #todo implement decoding
                #visuals.decoder(latent)
                latent = latent.cpu().double().squeeze(3).squeeze(2)
                action = net(latent)
                #action = net(torch.tensor(raw_observation).float().unsqueeze(0))
                _, the_action = action.max(1)
                env.render()
                raw_observation, reward, done, info = env.step(the_action.item())
                score += reward
                stats['episode_steps'] += 1
                if done:
                    break

        cma.add(net, score, stats)

    best, rest = cma.rank_and_compute()

    save_path = Path('modelzoo/run2')
    save_path.mkdir(parents=True, exist_ok=True)

    filename = save_path / ('best_model%d' % epoch)
    with filename.open('wb') as f:
        torch.save(policy_nets[best[0].item()], f)

    filename = save_path / ('cma_%d' % epoch)
    with filename.open('wb') as f:
        import pickle
        pickle.dump(cma, f)

    for net in policy_nets:
        cma.set_sample_weights(net)





