# highway-env

Introduce highway-env

## Custom modifications
### Previous modifications
- State transistions
- Custom merging environment
- include `n_a` and `n_s` to define number of variables in action space and statespace.

[//]: # (TODO: Check and confim if there are any more!)

### Current modifications 
- Add `heading` \($\phi$\) in the state variable by defining new controller class, action class and observation class.
- Modified initial spawn points (mod2) in merging env.
- Control leteral movement of vehicle using steering velocity control instead of steering control.

## Usage

```python
import gym
import highway_env

env = gym.make("merge-multi-agent-v0")

done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, info = env.step(action)
    env.render()
```

## Acknowledgements

This version of highway-env is derived from:
```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```
