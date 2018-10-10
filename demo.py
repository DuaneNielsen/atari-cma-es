import models
import mentalitystorm.transforms as tf
import torchvision.transforms as TVT
from mentalitystorm.basemodels import MultiChannelAE
from mentalitystorm.runners import Run
from mentalitystorm.config import config
from mentalitystorm.policies import VCPolicyMultiAE, RolloutGen
from mentalitystorm.data_containers import ActionEmbedding
import torch
import gym
from viewer import *

shots = tf.ColorMask(lower=[128, 128, 128], upper=[255, 255, 255], append=True)
player = tf.ColorMask(lower=[30, 100, 40], upper=[70, 180, 70], append=True)
cut_player = tf.SetRange(0, 60, 0, 210, [4])
invader = tf.ColorMask(lower=[120, 125, 25], upper=[140, 140, 130], append=True)
cut_invader = tf.SetRange(0, 30, 0, 210, [5])
barrier = tf.ColorMask(lower=[120, 74, 30], upper=[190, 100, 70], append=True)
select = tf.SelectChannels([3, 4, 5, 6])

observe = tf.ViewChannels('transform', (320, 480), channels=[0, 1, 2])

segmentor = TVT.Compose([shots, player, cut_player, invader, cut_invader,
                         barrier, select, TVT.ToTensor(), tf.CoordConv()])

device = config.device()
shot_encoder = Run.load_model(r'.\modelzoo\vision\shots.run').eval().to(device=config.device())
player_encoder = Run.load_model(r'.\modelzoo\vision\player.run').eval().to(device=config.device())
invaders_encoder = Run.load_model(r'.\modelzoo\vision\invaders.run').eval().to(device=config.device())
barrier_encoder = Run.load_model(r'.\modelzoo\vision\barrier.run').eval().to(device=config.device())
visuals = MultiChannelAE()
visuals.add_ae(shot_encoder, [0, 4, 5], [0])
visuals.add_ae(player_encoder, [1, 4, 5], [1])
visuals.add_ae(invaders_encoder, [2, 4, 5], [2])
visuals.add_ae(barrier_encoder, [3, 4, 5], [3])

visuals.register_forward_hook(view_decode)
visuals.register_forward_hook(view_image)


controller = torch.load(r'C:\data\SpaceInvaders-v4\policy_runs\629\best_model0')

env = gym.make('SpaceInvaders-v4')
policy = VCPolicyMultiAE(visuals, controller, segmentor, ActionEmbedding(env), device)


for screen, observation, reward, done, info, action in RolloutGen(env, policy, render_to_window=True, populate_screen=True):
    pass
