# Dynamic Programming for MDPs

In this class, we discussed the Dynamic Programming and different algorithms used for solving MDPs.

We showed how the Bellman equations can be used to do:

* Policy Evaluation (Prediction) 
* Policy Iteration/Improvement
* Value Iteration

This exercise is meant to allow you implement the value iteration algorithm.

The empty code is found in `value_iteration.py` with several `TODO` blocks to help you implement the solution

The full solution is implemented in `value_iteration_solved.py`.

Notes:
1. The code here is implemented to be more understandable in an academic fashion and to correlate to what we described in class and not necessarily the cleanest or most optimal.

2. The solution may have a slightly different structure or more methods, this is meant to clarify the steps

3. Important for usage, the gridlworld environment was isolated into a separate files (`value_iteration_agent.py` and `gridworld.py`) to allow for more flexibility of using it in your work.



## Directions

It is recommended that you start with the `value_iteration.py` and try to implement the algorithm yourself.
You can then compare the results with those from `value_iteration_solved.py` and see what you can improve.

There are several TODOs in `value_iteration_solved.py` as well to help you explore more aspects discussed in class and get a better feel for the problem in hand.

## Contribution

If you have a different implementation that you'd like to share with the class, please create a Pull Request describing what you added.
This may be something like exploring some of the asynchronous methods.