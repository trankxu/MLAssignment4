import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
class ValueIteration:
    def __init__(self, goal = 100, proba_h = 0.4, theta=1e-9, gamma = 1):
        self.ph = proba_h
        self.gamma = gamma
        self.goal = goal
        self.theta = theta
        self.states = np.arange(self.goal + 1)
        self.state_values = np.zeros(self.goal + 1)
        self.state_values[self.goal] = 1.0
        self.policy = np.zeros(self.goal + 1)
        self.sweeps_history = []

    def truncated_policy_evaluation(self):
        while True:
            old_state_values = self.state_values.copy()
            self.sweeps_history.append(old_state_values)
            for s in self.states[1:self.goal]:
                actions = np.arange(min(s, self.goal - s))+1
                action_returns = []
                for a in actions:
                    ret = self.ph * (self.gamma * self.state_values[s + a]) + \
                          (1 - self.ph) * (self.gamma * self.state_values[s - a])
                    action_returns.append(ret)
                self.state_values[s] = np.max(action_returns)   # max_a
            delta = abs(self.state_values - old_state_values).max()
            if delta <= self.theta:
                break

    def optimal_policy(self):
        for s in self.states[1:self.goal]:
            actions = np.arange(min(s, self.goal - s))+1
            old_a = self.policy[s]
            action_returns = []
            for a in actions:
                ret = self.ph * (self.gamma * self.state_values[s + a]) + \
                      (1 - self.ph) * (self.gamma * self.state_values[s - a])
                action_returns.append(ret)
            argmax_a = actions[np.argmax(np.round(action_returns,5))]
            self.policy[s] = argmax_a

final_score=list()
gammaa=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
probab=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
'''
for gamm in gammaa:

    value1=ValueIteration(gamma=gamm)
    value1.truncated_policy_evaluation()
    value_sta=value1.optimal_policy()
    for i in range(0,len(value1.sweeps_history)):
        plt.plot(range(0,101), value1.sweeps_history[i],label=i)
        plt.legend()
        plt.xlabel('state')
        plt.grid()
        plt.title('state-score curve when gamma={}'.format(gamm))
        plt.ylabel('Average Reward')
    plt.show()
    final_score.append(value1.sweeps_history[len(value1.sweeps_history)-1])
for i in range(len(final_score)):
    plt.plot(range(0,101), final_score[i],label="gamma={}".format((i+1)/10))
    plt.legend()
    plt.xlabel('state')
    plt.grid()
    plt.title('State-value score curve for different Gamma')
    plt.ylabel('Average Reward')
plt.show()

for prob in probab:
    policy2 = ValueIteration(proba_h=prob,gamma=1)
    policy2.truncated_policy_evaluation()
    policy_sta = policy2.optimal_policy()
    plt.plot(range(0, 101), policy2.sweeps_history[len(policy2.sweeps_history)-1], label="probab={}".format(prob))
    plt.legend()
    plt.xlabel('state')
    plt.grid()
    plt.title('state-score curve for different Ph when gamma=1')
    plt.ylabel('Average Reward')
plt.show()
'''

value1=ValueIteration(proba_h=0.6,gamma=1)
value1.truncated_policy_evaluation()
value_sta=value1.optimal_policy()
for i in range(0,len(value1.sweeps_history)):

    if i<5:
        plt.plot(range(0,101), value1.sweeps_history[i],label=i)
    if (5<=i<100):

        if i%15 != 0:
            continue
        else:
            plt.plot(range(0,101), value1.sweeps_history[i],label=i)
    if i>99:
        if i%100 != 0:
            continue
        else:
            plt.plot(range(0,101), value1.sweeps_history[i],label=i)

    plt.legend()
    plt.xlabel('state')
    plt.grid()
    plt.title('state-score curve when gamma={} Ph=0.6'.format(1))
    plt.ylabel('Average Reward')
plt.show()