# RL1-Frozen-Lake-with-Stochastic-Wind
Using Monte Carlo Learning and Policy Evaluation-Iteration methods to chart a path for an agent to reach a goal through an icy terrain while also avoiding intermittent holes and dealing with stochastic wind which make the agents' actions non-correspondent with it's observed state transition
  

<!--Original Policy (a random policy when agent is still exploring the environment)

![RL1_raw_gif](https://github.com/Riddhiman-M/RL1-Frozen-Lake-with-Stochastic-Wind/assets/89708853/97d2a7c5-6e9e-41e6-82ec-024f20204f7f)-->

Original Policy | Policy after 30 episodes
:-: | :-:
<img src='https://github.com/Riddhiman-M/RL1-Frozen-Lake-with-Stochastic-Wind/assets/89708853/7e8e96be-53cf-4541-b921-99fa9b06aab8' height=240/> | <img src='https://github.com/Riddhiman-M/RL1-Frozen-Lake-with-Stochastic-Wind/assets/89708853/80506f71-c3cf-475e-a16f-9580e0141a45' height=240/>

The policy is found to converge within 30 episodes while the state values take ~1000 episodes to converge by Monte-Carlo method

Final state values:

<img src='https://github.com/Riddhiman-M/RL1-Frozen-Lake-with-Stochastic-Wind/assets/89708853/a5c2dee5-543a-4e43-b769-0e261b018e49' height=240/>
