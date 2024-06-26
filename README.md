# KernelDE


- See KernelDEE.ipynb for notebook with implementation
- `calculate_DE.py` shows a simple example how to use the program.


To make more simulations and reproduce grid_search_session2 results:

```
python -m DE_main 0 4 1 #FQK
python -m DE_main 0 5 1 #RBF
python -m DE_main 0 6 1 #PQK
```

and load them following grid_search_session2



List of things I still want to improve:

- Quantum Kernels:

    - Adapt to use pure squlearn
    - Implement Coupled DE
    - Implement PDE
    - Include and benchmark more types of projected kernel
    - Think of systematic simulation to compare methods
    - ~~Accept arbitrary squlearn circuits (DONE)~~

- QNNs:
    - Reproduce Paper Fig 6.
    - Sucessfully run any ode with shots
    - Sucessfully use variance regularization
    - Implement more freedom in choose of optimizer, perhaps this solves shot problems
    - Implement callback function for all optimizer to save mse and loss history (currently only works for adam, because no callback is used)
    - Implement Coupled DE
    - Implement PDE
