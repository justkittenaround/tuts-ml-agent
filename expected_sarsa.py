import sys
import random
import numpy as np
from typing import Dict
from collections import deque
import matplotlib.pyplot as plt
from collections import defaultdict

import mlagents
from mlagents_envs.environment import UnityEnvironment



env = UnityEnvironment(file_name='Basic_eSARSA', seed=1, side_channels=[])

env.reset()

behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")

spec = env.behavior_specs[behavior_name]
print("Number of observations : ", len(spec.observation_shapes))
print("Observation vector shape: ", spec.observation_shapes)

if spec.is_action_continuous():
  print("The action is continuous")

decision_steps, terminal_steps = env.get_steps(behavior_name)



#-------------------------------------------------------------------------

def e_greedy(Q, state, epsilon, nA):
    policy_s = np.ones(nA) * epsilon / nA[0]
    best = np.argmax(Q[state])
    policy_s[best] = 1-epsilon + (epsilon/nA[0])
    if np.sum(Q[state]) > 0:
        action = np.random.choice(np.arange(nA[0]), p=policy_s)
    else:
        action = env.behavior_specs[behavior_name].create_random_action(len(decision_steps))
    return action


def update_Q_expsarsa(alpha, gamma, nA, eps, Q, state, action, reward, next_state=None):
    """Returns updated Q-value for the most recent experience."""
    current = Q[state][action]         # estimate in Q-table (for current state, action pair)
    policy_s = np.ones(nA) * eps / nA[0]  # current policy (for next state S')
    policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA[0]) # greedy action
    Qsa_next = np.dot(Q[next_state], policy_s)         # get value of state at next time step
    target = reward + (gamma * Qsa_next)               # construct target
    new_value = current + (alpha * (target - current)) # get updated value
    return new_value



def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
    nA = env.behavior_specs[behavior_name].action_shape
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        episode_rewards = 0
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon = 1/i_episode
        env.reset() # begin the episode
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1 # -1 indicates not yet tracking
        done = False # For the tracked_agent
        state = tuple(map(tuple, o) for o in decision_steps.obs)
        while not done:
            if tracked_agent == -1 and len(decision_steps) >= 1:
              tracked_agent = decision_steps.agent_id[0]
            #choose next action under e-greedy Q
            action = e_greedy(Q, state, epsilon, nA)
            #take action and observe r,s
            env.set_actions(behavior_name, action)
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name) # agent performs internal updates based on sampled experience
            next_state = tuple(map(tuple, o) for o in decision_steps.obs)
            if tracked_agent in decision_steps:
                reward = decision_steps[tracked_agent].reward
                episode_rewards += decision_steps[tracked_agent].reward
                #Q(s,a) <-- Q(s,a) + a(R1 + g*sum(expected prob a|next state *Q(next_state, action)) - Q(s,a))
                prev_Q = Q[state][action]
                eps_probs = np.ones(nA) * (1-epsilon) + (epsilon/nA[0])
                Q[state][action] = update_Q_expsarsa(alpha, gamma, nA, epsilon, Q, state, action, reward, next_state)
                state = next_state
            if tracked_agent in terminal_steps:
                done = True
                episode_rewards += terminal_steps[tracked_agent].reward
                print(f"Total rewards for episode {i_episode} is {episode_rewards}")
                break
    return Q


Q_table = expected_sarsa(env, 10, .7, .7)
