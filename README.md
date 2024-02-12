# PPO-based Quadrotors Controller

Reinforcement Learning for Robust Locomotion Control of Quadrotors

## Description

* Hard Coded Version: MujocoEnv, [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) are used
* Mujoco built-in fluid dynamics is used

## Getting Started

### Dependencies

* gymnasium==0.29.1
* matplotlib==3.8.2
* mpi4py==3.1.5
* numba==0.58.1
* numpy==1.26.3
* scipy==1.12.0
* stable_baselines3==2.2.1
* tensorflow==2.15.0
* tensorflow_macos==2.15.0


### Executing program

* Execute train/run.py
* Logs and models will be saved at train/logs and train/saved_models respectively
```
cd train
python run.py
```


## Authors

Mintae Kim  
Jiaze Cai

## Version History

* 0.1.0
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE file for details
