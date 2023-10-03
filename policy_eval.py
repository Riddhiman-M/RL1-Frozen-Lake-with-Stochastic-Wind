from math import sqrt
import numpy as np
import time
import gymnasium as gym


def checkPolicyConvergence(valueFunctionVector,valueFunctionVectorNextIteration):
    dim = valueFunctionVector.shape[0]

def checkValid(curr_state, dim):
    if curr_state[0]>=0 and curr_state[0]<dim and curr_state[1]>=0 and curr_state[1]<dim:
        return True
    else:
        return False

def getNewPolicy(val_Next, policy):
    dim = val_Next.shape[0]
    val_Next[dim-1][dim-1] = 1
    # print(val_Next)
    new_policy = np.zeros((dim*dim, dim))
    
    ## Calculate the new policy based on the new value function
    for i in range(dim):
        for j in range(dim):
            state = np.array([i,j])
            curr_best = 0
            all_zero = True
            best_action_list = []
            for delta in [[-1,0],[0,1],[1,0],[0,-1]]:
                new_state = np.add(state, delta)
                if not checkValid(new_state, dim):
                    continue
                if val_Next[new_state[0]][new_state[1]]>=curr_best:
                    if val_Next[new_state[0]][new_state[1]]>curr_best:
                        best_action_list = []
                    if (delta == np.array([-1,0])).all():
                        best_action_list.append(3) # Up
                    elif (delta == np.array([0,1])).all():
                        best_action_list.append(2) # Right
                    elif (delta == np.array([1,0])).all():
                        best_action_list.append(1) # Down
                    else:
                        best_action_list.append(0) # Left
                    curr_best = val_Next[new_state[0]][new_state[1]]
            
            for k in range(len(policy[i*dim+j])):
                new_policy[i*dim+j][k] = 1.0/len(best_action_list) if k in best_action_list else 0
    
    # print(new_policy)

    
    return new_policy


def evaluatePolicy(env,valueFunctionVector,policy,discountRate,maxNumberOfIterations,convergenceTolerance, updatePolicy):
    convergenceTrack=[]
    print('Policy: ',policy)
    for iterations in range(maxNumberOfIterations):
        convergenceTrack.append(np.linalg.norm(valueFunctionVector,2))
        valueFunctionVectorNextIteration=np.zeros(env.observation_space.n)
        # if iterations==0:
        #     print('Iteration 0 State 0: ', env.P[0])
        # elif iterations==100:
        #     print('Iteration 100 State 0: ', env.P[0])
        #     exit(0)
        
        for state in env.P:
            outerSum=0
            for action in env.P[state]:
                innerSum=0
                for probability, nextState, reward, isTerminalState in env.P[state][action]:
                    #print(probability, nextState, reward, isTerminalState)
                    innerSum=innerSum+ probability*(reward+discountRate*valueFunctionVector[nextState])
                outerSum=outerSum+policy[state,action]*innerSum
            valueFunctionVectorNextIteration[state]=outerSum

        print('Iteration no.: ',iterations)
        if updatePolicy:
            new_policy = getNewPolicy(valueFunctionVectorNextIteration.reshape(4,4), policy)
            if (new_policy == policy).all():
                print('Policy converged!')
                break
            policy = new_policy
        # if(iterations == 15):
        #     exit(0)
        
        if(np.max(np.abs(valueFunctionVectorNextIteration-valueFunctionVector))<convergenceTolerance):
            valueFunctionVector=valueFunctionVectorNextIteration
            print('Iterative policy evaluation algorithm converged!')
            print(f'Iterations {iterations}')
            break
        
        flag = checkPolicyConvergence(valueFunctionVector,valueFunctionVectorNextIteration)
        valueFunctionVector=valueFunctionVectorNextIteration   
    
    print('Final Policy\n',policy)
    
    return valueFunctionVector


def getBestPolicy(env, stateValues):
    action_list, state_list = [], []
    dim = int(sqrt(env.observation_space.n))
    curr_state = np.array([0,0])
    num_state = len(stateValues)
    goal_state = np.array([dim-1,dim-1])

    while True:
        curr_best = 0
        # print('Curr state: ',curr_state)
        state_list.append(curr_state)
        if (curr_state == goal_state).all():
            break
        for delta in [[-1,0],[0,1],[1,0],[0,-1]]:
            new_state = np.add(curr_state, delta)
            # print('New state: ',new_state)
            # time.sleep(5)
            if not checkValid(new_state, dim):
                continue
            # print(new_state)
            # print(new_state[0], "...", new_state[1])
            # print(stateValues[0][1])
            if stateValues[new_state[0]][new_state[1]]>curr_best:
                if (delta == np.array([-1,0])).all():
                    best_action = 3 # Up
                elif (delta == np.array([0,1])).all():
                    best_action = 2 # Right
                elif (delta == np.array([1,0])).all():
                    best_action = 1 # Down
                else:
                    best_action = 0 # Left

                best_state = new_state
                curr_best_action = best_action
                curr_best = stateValues[new_state[0]][new_state[1]]
            # print('Best state: ',best_state)

        curr_state = best_state
        action_list.append(curr_best_action)
                    
    return state_list, action_list

def simul(desc, action_list):
    env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=False,render_mode="human")
    t = env.reset()
    for action in action_list:
        print('Taking action ',action)
        obs, reward, terminal, trunc, info = env.step(action)
        time.sleep(0.5)

# import numpy as np
# curr_state = (0,0)
# print(curr_state != (3,3))
# # while curr_state != (3, 3):
# for delta in [(-1,0),(0,1),(1,0),(0,-1)]:
#     new_state = np.add(curr_state,delta)
#     print(new_state)
