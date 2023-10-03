import time
import numpy as np


def MonteCarlo(env,stateNumber,numberOfEpisodes,discountRate):
    
    sumReturnForEveryState=np.zeros(stateNumber)
    numberVisitsForEveryState=np.zeros(stateNumber)
    
    # estimate of the state value function vector
    valueFunctionEstimate=np.zeros(stateNumber)
    
    for episode in range(numberOfEpisodes):
        # this list stores visited states in the current episode
        visitedStatesInEpisode=[]
        # this list stores the return in every visited state in the current episode
        rewardInVisitedState=[]
        (currentState,prob)=env.reset()
        visitedStatesInEpisode.append(currentState)
        
        print("Simulating episode {}".format(episode))
        
        # Below is a single episode simulation
        while True:
            
            # select a random action
            randomAction= env.action_space.sample()
            # 0 - Left
            # 1 - Down
            # 2 - Right
            # 3 - Up
            
            # here we step and return the state, reward, and boolean denoting if the state is a terminal state
            # The action that the we choose above is what the agent will try to take but the stochastic wind can push it in a different direction
            currState, currReward, terminalState, info, _ = env.step(randomAction)          
            
            rewardInVisitedState.append(currReward)
            
            # if the current state is NOT terminal state 
            if not terminalState:
                visitedStatesInEpisode.append(currState)   
            # if the current state IS terminal state 
            else: 
                break

        
        
        # how many states we visited in an episode    
        numberOfVisitedStates=len(visitedStatesInEpisode)
        print(visitedStatesInEpisode)
        print(rewardInVisitedState)
            
        # Gt is the return from the current state to the end of the episode
        Gt=0
        
        print(visitedStatesInEpisode)
        for currIndex in range(numberOfVisitedStates-1,-1,-1):
                
            currState = visitedStatesInEpisode[currIndex] 
            returnTmp = rewardInVisitedState[currIndex]
            # print("State {} has return {}".format(stateTmp,returnTmp))
              
              
            Gt=discountRate*Gt+returnTmp
              
            # First Visit MC
            # checking if it hasn't occured anywhere previously in the episode, so that the reward is counted only once for that state in that episode
            if currState not in visitedStatesInEpisode[0:currIndex]:  
                numberVisitsForEveryState[currState]=numberVisitsForEveryState[currState]+1 # so this is basically number of episodes in which the current state occured
                sumReturnForEveryState[currState]=sumReturnForEveryState[currState]+Gt
        
        # print(sumReturnForEveryState)
            
        # if indexEpisode==10:
        #     exit(0)
            
    
    
    # finally we need to compute the final estimate of the state value function vector
    for indexSum in range(stateNumber):
        if numberVisitsForEveryState[indexSum] !=0:
            valueFunctionEstimate[indexSum]=sumReturnForEveryState[indexSum]/numberVisitsForEveryState[indexSum]
        
    return valueFunctionEstimate
            



        