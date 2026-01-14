# Benchmarking code for Imitation Learning

A (shortened) single trial for the Mushroom, Warfarin, and Wheel problems can be run using example.py via:

```
python example.py
```

Results in the paper can be reproduced using the number of trials and number of time steps specified in the paper. For exact reproduction, each method must be run using the same random seed individually, in order to ensure that the initial random actions are the same for each method in each trial.

## Dependencies/Requirements

- Python >= 3.6
- botorch >= 0.2.5
