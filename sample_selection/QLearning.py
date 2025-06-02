import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


class QLSS():
    def __init__(self, N_STATES=6, ACTIONS=None, EPSILON=0.9, ALPHA=0.1, GAMMA=0.9, MAX_EPISODES=13, FRESH_TIME=0.3):
        if ACTIONS is None:
            self.ACTIONS = ['left', 'right']
        self.N_STATES = N_STATES
        self.EPSILON = EPSILON
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.MAX_EPISODES = MAX_EPISODES
        self.FRESH_TIME = FRESH_TIME

    def build_q_table(self, n_states, actions):
        table = pd.DataFrame(
            np.zeros((n_states, len(actions))),     # q_table initial values
            columns=actions,    # actions's name
        )
        # print(table)    # show table
        return table

    def choose_action(self, state, q_table):
        # This is how to choose an action
        state_actions = q_table.iloc[state, :]
        if (np.random.uniform() > self.EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(self.ACTIONS)
        else:   # act greedy
            action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
        return action_name

    def get_env_feedback(self, S, A):
        # This is how agent will interact with the environment
        if A == 'right':    # move right
            if S == self.N_STATES - 2:   # terminate
                S_ = 'terminal'
                R = 1
            else:
                S_ = S + 1
                R = 0
        else:   # move left
            R = 0
            if S == 0:
                S_ = S  # reach the wall
            else:
                S_ = S - 1
        return S_, R

    def update_env(self, S, episode, step_counter):
        # This is how environment be updated
        env_list = ['-']*(self.N_STATES-1) + ['T']   # '---------T' our environment
        if S == 'terminal':
            interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
            print('\r{}'.format(interaction), end='')
            time.sleep(2)
            print('\r                                ', end='')
        else:
            env_list[S] = 'o'
            interaction = ''.join(env_list)
            print('\r{}'.format(interaction), end='')
            time.sleep(self.FRESH_TIME)

    def rl(self):
        q_table = self.build_q_table(self.N_STATES, self.ACTIONS)
        for episode in range(self.MAX_EPISODES):
            step_counter = 0
            S = 0
            is_terminated = False
            self.update_env(S, episode, step_counter)
            while not is_terminated:
                A = self.choose_action(S, q_table)
                S_, R = self.get_env_feedback(S, A)  # take action & get next state and reward
                q_predict = q_table.loc[S, A]
                if S_ != 'terminal':
                    q_target = R + self.GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
                else:
                    q_target = R     # next state is terminal
                    is_terminated = True    # terminate this episode
                q_table.loc[S, A] += self.ALPHA * (q_target - q_predict)  # update
                S = S_  # move to next state
                self.update_env(S, episode, step_counter+1)
                step_counter += 1
        return q_table


if __name__ == "__main__":
    ql = QLSS()
    q_table = ql.rl()
    print('\r\n Q-table:\n')
    print(q_table)