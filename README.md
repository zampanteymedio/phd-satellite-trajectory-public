# Agents to solve the gym envs for Satellite Trajectory Optimization

[![Build Status](https://travis-ci.com/zampanteymedio/phd-satellite-trajectory.svg?token=5u83STK3Ceb1MuJDeDoy&branch=master)](https://travis-ci.com/zampanteymedio/phd-satellite-trajectory)
[![codecov](https://codecov.io/gh/zampanteymedio/phd-satellite-trajectory/branch/master/graph/badge.svg?token=YKKWPZOOOT)](https://codecov.io/gh/zampanteymedio/phd-satellite-trajectory)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.6&nbsp;|&nbsp;3.7&nbsp;|&nbsp;3.8](https://img.shields.io/badge/python-3.6&nbsp;|&nbsp;3.7&nbsp;|&nbsp;3.8-blue.svg)](https://www.python.org/downloads/release/python-360/)

The [Phd Satellite Trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) is a set of agents
that solve the [Satellite Trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) environments.

# Installation

You can follow the directions in the [.travis.yml](.travis.yml) file, which describes the necessary steps
to install all the dependencies and this package in a clean environment.

# Working with this package

## Train agents

If you work in the workspace folder, everything will be saved in place for you.

Each training module has an entry point defined in setup.py file. If you want to run a case, simply use those entry points, e.g.:

```
perigeeraising_discrete1d_dqn
```

## Tensorboard

If you want to run a tensorboard server, simply type from the workspace folder:

```
launch_tensorboard.sh
```
