# Code Repository for Paper: Risk-Aware Congested Link Diagnosis with CVaR Enhanced Network Boolean Tomography

## Environment Setup

The code requires a Python environment as specified in `requirements.txt`. Since the code leverages GPU for computation, you need to install the appropriate GPU drivers for your system. Additionally, Gurobi is used for solving optimization problems, so you will need to install Gurobi as well.

## File Descriptions and Usage

### Algorithm Files
Files prefixed with `alg` contain various network tomography algorithms, including our CENBT algorithm and other benchmarks.

The input format for the algorithms is as follows:

1. Observation Information: The observations of the links. For example, the second row `[0, 0, 1]` in the code snippet below indicates that in the second observation, paths 1 and 2 are clear (observation result is 0), while path 3 is congested (observation result is 1).
```python
y = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]], dtype=np.int8).T
```
2. Routing Matrix: This represents the topology structure. The second row `[1, 0, 1, 1, 0]` in the code snippet below indicates that path 1 passes through links 1, 3, and 4 (positions with 1), but not through links 2 and 5 (positions with 0).
```python
A_rm = np.array([
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1]], dtype=np.int8)
```
3. Link Prior Congestion Probability: This indicates the prior congestion probability of the links in order.
```python
x_pc = np.array([0.1] * 5)
```
4. Other input parameters such as α, β, and θ can be input at the corresponding algorithm interfaces.

### Iterative Fault Repair Process Files
`diagnoser.py` and `diagnose_robot.py` are files for simulating the iterative fault repair process.

`diagnose_robot` is a class used to simulate the repair process of a network for a given observation. Its inputs include: `y` for the initial observation state, `x` for the true link state, `A_rm` for the routing matrix, `method` for the prediction method, `x_pc` for the parameters of the prediction method (if any), and `Fn` representing which F-measure is being calculated (default is 1 for F1). More specific inputs and outputs can be found in the comments and code.

`diagnoser` integrates multiple observations using `diagnose_robot`, accepting multiple observation inputs and invoking multiple parallel processes created by `diagnose_robot`, and finally aggregating the information for output.

`diagnose_method_compare_v2.py` is a file for conducting Iterative Congestion Troubleshooting.

```python
methods=['CENBT_98%','SAT_98%','map_98%','G-CALS_98%']
```
This line in the code represents the methods to be compared, with the percentage after the "_" indicating the reliability β value.

```python
    tp_name = 'Chinanet'
    lb=0
    ub=0.1
    alpha=1.9
    theta=0.5
```
In the code, these lines set the topology and experimental parameters. `lb` represents the lower bound of the prior congestion probability of the links, `ub` represents the upper bound, `alpha` is the penalty coefficient in CENBT, and `theta` is the decision threshold in CENBT. Note that these settings can call different topologies and link prior congestion probabilities used in our experiments, but you can also write your own input data according to our input format and conduct experiments.

```python
diagnose_method_compare(scenario_prob=scenarios_prob,observe_times=5000,topology_name=topology_name,source_nodes=source_nodes,lb=lb,ub=ub,alpha=alpha,theta=theta)
```
In this line, the main adjustable parameter is `observe_times`. By changing the integer value after it, you can modify the number of simulation runs.

The experimental results after running `diagnose_method_compare_v2.py` should be placed in a newly generated folder named `result_diagnose_method_v2` in the same directory.

### Other Files
The `topology_zoo` folder contains code for initializing the network and data for various topologies.

The `topology_probs` folder contains the prior congestion probabilities of network links set during our experiments. You can call the data in this folder for experiments, or write your own experimental data according to the format and your needs.