rm -rf policy/ q/
python run_dqn_critic_cartpole.py &bg
tensorboard --logdir .
