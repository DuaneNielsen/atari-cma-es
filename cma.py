import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal


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
                  'CME: mean %f, sigma %f, parameters %f' % (
                      score_mean, score_var, best_score,
                      episode_steps_np.mean(), episode_steps_np.max(),
                      self.distrib.mean.mean(), self.distrib.stddev.mean(),
                      self.distrib.mean.numel()))
