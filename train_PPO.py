#!/usr/bin/python3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from quadrotor_env import QuadrotorEnv
from gym.envs.registration import register
import rospy
import gym
import rospkg
from stable_baselines3.common.callbacks import CheckpointCallback

register(id='Quadrotor-v0', entry_point='quadrotor_env:QuadrotorEnv')
rospy.init_node('quadrotor_gym', anonymous=True)
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('hector_quadrotor_gazebo')
log_path = pkg_path + '/Training/Logs'

Alpha = 0.0003
Gamma = 0.99
nepisodes = 5 
nsteps = 2048 # Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch.
nepochs= 10 # Number of epochs of interaction (equivalent to number of policy updates) to perform.
batch_size = 64

env = gym.make('Quadrotor-v0')
#check_env(env)

#TEST THE ENVIROMENT
'''for episode in range(1,nepisodes+1):
    rospy.loginfo('Inizio Episodio: '+str(episode))
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        #rospy.loginfo('obs:{} reward:{}'.format(obs, reward))
        total_reward += reward           
    rospy.loginfo('Episode:{} Score:{}'.format(episode,total_reward))'''


# Save a checkpoint every 10000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path= log_path,
  name_prefix="PPO_model",
  save_replay_buffer=True,
  save_vecnormalize=True
)
#TRAIN THE MODEL
#policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, learning_rate=Alpha, n_steps=nsteps, n_epochs=nepochs,batch_size=batch_size)
#model = PPO.load(pkg_path + '/Training/SavedModels/PPO_Navigation3', env)
model.learn(
    total_timesteps=50000, 
    progress_bar=True, 
    callback=checkpoint_callback)

(mean_reward, std_reward) = evaluate_policy(model, env, n_eval_episodes=2)
rospy.loginfo('mean_reward:{} std_reward:{}'.format(mean_reward,std_reward))

'''for episode in range(1, nepisodes+1):
    obs = env.reset() 
    score = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action) 
        score += reward 
    rospy.loginfo('Episode:{} Score:{}'.format(episode, score))'''