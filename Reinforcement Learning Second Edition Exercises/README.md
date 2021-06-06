# Reinforcement Learning Second Edition Exercises

## Exercise 2.5:
Design and conduct an experiment to demonstrate the
diculties that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the q*(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean 0
and standard deviation 0.01 to all the q*(a) on each step). Prepare plots like Figure 2.2
for an action-value method using sample averages, incrementally computed, and another
action-value method using a constant step-size parameter, alpha = 0.1. Use epsilon = 0.1 and
longer runs, say of 10,000 steps.


### Summary
- Create a bandit with 10 arms;
- Instantiate q*(a) = 0 for all a, then take random walks for each a (normal distribution, mean=0, deviation=0.01);
- Use the sample average technique and a bandit with alpha = 0.1;
- Create all the tests with epsilon = 1;

## Exercise 2.11:
Make a figure analogous to Figure 2.6 for the nonstationary
case outlined in Exercise 2.5. Include the constant-step-size epsilon-greedy algorithm with
alpha= 0.1. Use runs of 200,000 steps and, as a performance measure for each algorithm and
parameter setting, use the average reward over the last 100,000 steps.

### Summary
Make a graph with these bandits:
- Sample average ε-greedy;
- Constant step size ε-greedy with alpha = 0.1;
- Upper Bound Confidence (UBC);
- Gradient bandit;
- ε-greedy with optimistic initialization (e.g. q*(a) = 5 for all a) with alpha = 0.1;

#### Implementation
I used NumPy arrays for the bandits, so each bandit has it's own Q(a),
I did this because then I would only need one object and calculating stuff with arrays is a bit faster
than having to loop over all the Bandit objects and change their values seperatly.
