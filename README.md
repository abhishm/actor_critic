# Project Title
Implementation of actor-critic algorithm.

# Features
- Using a neural network based policy as the actor
- Using a Q-network as the critic
- Using Policy Gradient Theorem to update critic
- Using a variation of a Q-learning updates to update Q-network
- ![equation](http://www.sciweavers.org/download/Tex2Img_1495325239.jpg)

>Note that the above equation is similar as in the Q-learning update except that instead of using the max action-values, we are using the averaged action-values. The rationale for using the above update is that this update converges to the action-values of the present policy while the max update (Q-learning update) converges to the action-values of the optimal policy. We need the action-values of the present policies for policy gradient updates that is why we used the above updates.
