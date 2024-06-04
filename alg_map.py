import numpy as np

import mip
from mip import Model, CBC, GUROBI, BINARY, xsum


def alg_map(y: np.ndarray, A_rm: np.ndarray, x_pc: np.ndarray):

    num_paths, num_links = A_rm.shape
    if y.ndim == 1:  # 如果是横向量，则转为列向量
        Y_obs = np.reshape(y, (num_paths, -1))
    else:
        Y_obs = y

    num_times = Y_obs.shape[-1]

    x_identified = np.zeros((num_links, num_times), dtype=object)
    x_map_pr = np.zeros(num_times, dtype=object)

    try:
        Model(solver_name=GUROBI)  # 优先使用 gurobi 作为求解引擎
        MIP_engine = GUROBI
    except:
        MIP_engine = CBC

    for t in range(num_times):  # 逐个时刻进行诊断
        model = Model(name='MAP_Boolean_Tomography', sense='MAX', solver_name=MIP_engine)  # 重新生成模型以清空上一时刻的路径状态约束
        model.verbose = False  # 不打印 mip 的统计信息

        x = [model.add_var(name="x_" + str(i + 1), var_type=BINARY) for i in range(num_links)]  # 声明布尔变量：链路状态
        model.objective = mip.maximize(xsum(np.log(1 - x_pc[i]) * (1 - x[i]) +
                                            np.log(x_pc[i]) * x[i] for i in range(num_links)))

        for j in range(num_paths):  # 将路径状态观测值转化为约束
            if Y_obs[j, t]:
                model.add_constr(xsum(x[i] for i in range(num_links) if A_rm[j, i]) >= Y_obs[j, t])
            else:
                model.add_constr(xsum(x[i] for i in range(num_links) if A_rm[j, i]) == Y_obs[j, t])

        if model.optimize() == mip.OptimizationStatus.OPTIMAL:
            temp_x_identified = np.array([np.int8(model.vars[i].x) for i in range(num_links)])
            temp_x_map_pr = np.exp(model.objective_value)
        else:
            temp_x_identified = np.array([None] * num_links)
            temp_x_map_pr = None

        x_identified[:, t] = temp_x_identified
        x_map_pr[t] = temp_x_map_pr

    try:
        if num_times == 1 and y.ndim == 1:  # 在单时刻计算时，保持链路状态输出的维度形式与路径状态输入的维度形式一致
            x_identified = x_identified.reshape((-1))

        x_identified = np.int8(x_identified)  # 节约存储空间
    except:
        pass

    return x_identified, x_map_pr


if __name__ == "__main__":
    #routing_matrix = np.genfromtxt('10_16_routing_matrix.csv', delimiter=",")
    #num_paths, num_links = routing_matrix.shape

    #link_cong_pro = np.array([0.1] * num_links)
    # link_cong_pro = np.random.uniform(low=0.1, high=0.3, size=num_links)  # 随机指定链路的先验拥塞概率
    #Y_obs = np.array([[1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
                      #[1, 1, 1, 1, 1, 0, 0, 1, 1, 1]], dtype=np.int8).T
    
    y = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=np.int8).T

    A_rm = np.array([
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1]], dtype=np.int8)
    
    #print(A_rm[0,:])
    # A_rm = np.array([
    #     [1,0,1,1,0],
    #     [1,0,1,0,1],
    #     [0,1,1,0,1]
    #     ])

    x_pc = np.array([0.1] * 5)

    X_identified, X_map_pr = alg_map(y, A_rm, x_pc)

    #print('Given the piror link congestion probabilities:\n{}\n\nand the status observations of paths:'.format(
        #link_cong_pro))
    #print('\n'.join("y_{} = {}".format(j, Y_obs[j]) for j in range(num_paths)))

    #print('\nwe will have the following MAPs and MAP diagnoses:\n{}'.format(X_map_pr))
    #print('\n' + '\n'.join("x_{} = {}".format(i, X_identified[i]) for i in range(num_links)))
    print(X_identified)
