# 这是论文 Risk-Aware Congested Link Diagnosis with CVaR Enhanced Network Boolean Tomography 的代码库

## 运行环境

代码对python环境的要求如requirements.txt所示，由于代码使用了GPU进行了计算，故而为了正常运行代码，还需要安装你的电脑对应的显卡驱动，同时代码还使用了Gurobi进行求解，故而你也需要安装Gurobi。

## 文件作用以及使用

### 算法文件
以alg开头的是各种网络层析成像的算法文件，这包含了我们的CENBT算法和其他的benchmark

算法的输入格式如下:

1. 观测信息，对链路的观测，如下列代码第二横行`[0, 0, 1]`就指在第二次观测之中路径1和路径2是通畅的(观测结果为0)，而路径三是拥塞的(观测结果为1)。
```python
y = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]], dtype=np.int8).T
```
2. 路由矩阵, 表示对拓扑的结构，代码第二行的`[1, 0, 1, 1, 0]`就表示了路径1经过链路1，3，4(对应的位置为1)，而不经过链路2，5(对应的位置为0)。
```python
A_rm = np.array([
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1]], dtype=np.int8)
```
3. 链路先验拥塞概率，表示对应顺序链路的先验拥塞概率
```python
x_pc = np.array([0.1] * 5)
```
4. 其他的输入参数如α，β，θ都可以在对应的算法接口处输入

### 多轮迭代故障修复过程相关文件
diagnoser.py和diagnose_robot.py是多轮迭代故障修复过程模拟程序的文件

其中dignose_robot是一个类，用于模拟对某次观测的网络的修复过程，其输入有: y为初始的观测状态, x为真实的链路状态, A_rm为机器人的状态转移矩阵,method为预测方法, x_pc为预测方法的参数(如有), Fn代表算的是F几，默认为1计算F1, 更加的具体输入输出可以看注释

dignoser是对dignose_robot的多次观测集成使用，可以接受多次的观测信息输入，并调用多个并行的使用dignose_robot创建的进程，最后再将信息集中输出。

diagnose_method_compare_v2.py是一个进行多轮故障检测实验的文件

```python
methods=['CENBT_98%','SAT_98%','map_98%','G-CALS_98%']
```
代码中这一行代表需要进行对比的方法，"_"后面的百分比数代表可靠性β的数值。

```python
    tp_name = 'Chinanet'
    lb=0
    ub=0.1
    alpha=1.9
    theta=0.5
```
在代码之中，上述这些行的数值用于选择实验的拓扑和实验的参数设置，lb表示链路先验拥塞概率的下限，ub表示链路先验拥塞拥塞的上限，alpha即CENBT中的惩罚系数，theta就是CENBT中的决定阈值。需要注意的是，这里的设置可以调用到我们在实验中用到的不同拓扑的不同的链路先验拥塞概率，但实际上，你也可以根据我们的输入格式编写你自己的输入数据并进行实验。

```python
diagnose_method_compare(scenario_prob=scenarios_prob,observe_times=5000,topology_name=topology_name,source_nodes=source_nodes,lb=lb,ub=ub,alpha=alpha,theta=theta)
```
这一行之中主要的可设置参数是`observe_times`，通过更改其后的整数数值，你可以修改自己的模拟次数。

diagnose_method_compare_v2.py运行后的实验结果应该会被放到一个同路径下新生成的文件夹
result_diagnose_method_v2之中。

### 其他文件
topology_zoo文件夹存放着我们对网络进行初始化的代码，以及各个拓扑的数据。

topology_probs文件夹存放着我们实验时设置的网络链路先验拥塞概率，你可以调用其中的数据进行实验，也可以根据其格式和你的需要编写自己的实验数据。

