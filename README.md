# Large-Scale Collision-Free Path Planning

## Getting started

Install package in editable mode:

* For users: 		`pip install -e `
  `pre-commit install`
* For contributors:	`pip install -e “.[dev]”`

## Entrypoints

* `compute-trajectories:` generates a collision-free trajectories for an randomized scenario of initial and final positions (and visualizes). serves as an example on how to generate trajectories.
* `compute-trajectories-batch:` generates a batch of collision-free trajectories and saves the data in a results folder.
  From an architecture point of view, this could be improved. Currently it is implemented as a "standalone" file that needs to be executed to generate the data. It would be better to write it as a batch unit which we can pass the parameters to from a config and it is called inside a main function.
* `plot-runtime-boxplot:` generates a boxplot of scp computations. It takes all the data in a folder. output dir and input dir can be specified.
