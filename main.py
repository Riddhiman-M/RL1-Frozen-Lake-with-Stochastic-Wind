import gymnasium as gym
import numpy as np  
import time

from MCEval import MonteCarlo
from policy_eval import evaluatePolicy, simul, getBestPolicy


import seaborn as sns
import matplotlib.pyplot as plt 


desc=["SFFF", "FHFF", "FFFF", "HFFG"]

# To see the visualisation/rendering of the trials uncomment the line below... this would be very slow
# env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True,render_mode="human")

# If using visualisation use the above rendering else use this one below 
env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True)

numState = env.observation_space.n
method = "MonteCarlo"


def grid_print(valueFunction, fName):
    ax = sns.heatmap(valueFunction,
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    plt.title('Final Value Function')
    plt.savefig(fName,dpi=600)
    plt.show()


if method == "MonteCarlo":
    # number of simulation episodes
    numEpisodes=10000
    discountRate=0.7

    # estimate the state value function by using the Monte Carlo method
    MonteCarloValues = MonteCarlo(env,stateNumber=numState,numberOfEpisodes=numEpisodes,discountRate=discountRate)
    
    ValFinal = MonteCarloValues.reshape(4,4)
    ValFinal[3][3] = 1
    print(ValFinal)
    grid_print(ValFinal, fName='./results/monteCarloResults.png')
    states, actions = getBestPolicy(env,  ValFinal)
    print(actions)

    simul(desc, actions)

elif method == "PolicyEval" or method == "PolicyIteration":
    initialPolicy = (1/4)*np.ones((16,4))
    valInitial = np.zeros(env.observation_space.n)
    num_max_iterations = 1000
    convergenceLimit = 10**(-6)
    discountRate = 0.7
    updatePolicy = True if method=="PolicyIteration" else False            # True for Policy Iteration, False for Policy Evaluation
    fname = './results/PolicyIteration.png' if method=="PolicyIteration" else './results/PolicyEvaluation.png'

    PolicyValues = evaluatePolicy(env, valInitial, initialPolicy, discountRate, num_max_iterations, convergenceLimit, updatePolicy)

    
    # print(valueFunctionIterativePolicyEvaluation)
    ValFinal = np.reshape(PolicyValues, (4,4))
    ValFinal[3][3] = 1
    print(ValFinal)
    grid_print(ValFinal, fName=fname)

    states, actions = getBestPolicy(env, ValFinal)
    print(actions)


    simul(desc, actions)


