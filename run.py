import gym
import torch
import pickle
from cma import CMA
from mentalitystorm.config import config
from mentalitystorm.runners import Run
from mentalitystorm.observe import ImageViewer
from mentalitystorm.basemodels import MultiChannelAE
import torchvision.transforms as TVT
import mentalitystorm.transforms as tf
from models import PolicyNet
from pathlib import Path
from tqdm import tqdm
import gym_wrappers


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

view_latent1 = ImageViewer('latent1', (320, 480), channels=[0, 1, 2])
view_latent2 = ImageViewer('latent2', (320, 480), channels=[3])
view_input1 = ImageViewer('input1', (320, 480), channels=[0, 1, 2])
view_input2 = ImageViewer('input2', (320, 480), channels=[3, 4, 5])


def view_image(model, input, output):
    view_input1.update(input[0].data)
    view_input2.update(input[0].data)
    view_latent1.update(output[0].data)
    view_latent2.update(output[0].data)


decode_viewer1 = ImageViewer('decoded1', (320, 480), channels=[0, 1, 2])
decode_viewer2 = ImageViewer('decoded2', (320, 480), channels=[3])


def view_decode(model, input, output):
    image = model.decode(output)
    decode_viewer1.update(image)
    decode_viewer2.update(image)

env = gym.make('SpaceInvaders-v4')
env = gym_wrappers.StepReward(env, step_reward=1)

policy_nets = []
cma_file = r'C:\data\SpaceInvaders-v4\policy_runs\603\cma_8'

with Path(cma_file).open('rb') as f:
    cma = pickle.load(f)

episode_steps = []
sample_size = 400
epochs = 200
rollouts = 1

z_size = 32

viewers = False

if viewers:
    visuals.register_forward_hook(view_decode)
    visuals.register_forward_hook(view_image)


for net in range(sample_size):
    policy_nets.append(PolicyNet((11, 8), 4, 6).double())
    episode_steps.append(0)

run_id = config.increment_run_id()

for epoch in range(epochs):
    for net in tqdm(policy_nets):
        score = 0
        stats = {'episode_steps': 0}
        for i_episode in range(rollouts):
            raw_observation = env.reset()
            for t in range(5000):
                #screen = env.render(mode='rgb_array')

                obs = segmentor(raw_observation)
                obs = obs.unsqueeze(0).to(device)
                latent = visuals(obs)

                latent = latent.cpu().double().squeeze(3).squeeze(2)
                action = net(latent)
                _, the_action = action.max(1)

                raw_observation, reward, done, info = env.step(the_action.item())
                score += reward
                stats['episode_steps'] += 1
                if done:
                    break

        cma.add(net, score, stats)

    best, rest = cma.rank_and_compute()

    save_path = config.basepath() / 'SpaceInvaders-v4' / 'policy_runs' / run_id
    save_path.mkdir(parents=True, exist_ok=True)

    filename = save_path / ('best_model%d' % epoch)
    with filename.open('wb') as f:
        torch.save(policy_nets[best[0].item()], f)

    filename = save_path / ('cma_%d' % epoch)
    with filename.open('wb') as f:
        pickle.dump(cma, f)

    for net in policy_nets:
        cma.set_sample_weights(net)
