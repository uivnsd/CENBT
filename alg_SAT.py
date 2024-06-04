import numpy as np
from z3 import *
from copy import copy


def alg_SAT(y: np.ndarray, A_rm: np.ndarray):

    if np.ndim(y) <= 1:  # 强制转换为列向量
        y = y.reshape((-1, 1))

    _, n = A_rm.shape
    _, num_time = y.shape

    x_identified = np.zeros((n, num_time))

    for i in range(num_time):
        path_state_obv = y[:, i]  # 按照行来取
        links_state_inferred = SAT_z3(path_state_obv, A_rm)

        for j in range(links_state_inferred.shape[0]):
            x_identified[j][i] = links_state_inferred[j]

    return x_identified

def SAT_z3(y, A_rm):
    y_new = copy(y)
    _, n = A_rm.shape
    
    links_state_inferred = np.zeros(n, dtype=np.int8)
    if np.where(y_new > 0)[0].size <= 0:
        return np.zeros(n, dtype=np.int8)
    
    # 根据 路径的测量结果对 链路进行统计
    path_good_index,  = np.where(y_new < 1)
    link_good_index,  = np.where(np.sum(A_rm[path_good_index, :], axis = 0)  > 0)
    path_fault_index, = np.where(y_new > 0)
    
    # 将所有的路径分为两类，一类是经过故障链路的，一类是没有经过故障链路的
    link_fault_sets = []
    for index in path_fault_index:
        link_fault_set =  np.where(A_rm[index, :] > 0)
        link_fault_set = link_fault_set[0].tolist()
        
        link_fault_sets.append([x for x in link_fault_set if x not in link_good_index])

    
    # 对故障链路组成的集和 去重
    U_ =  [x for set_ in link_fault_sets for x in set_] 
    U = list(set(U_))

    
    # 若两者皆为空集 ，则返回
    if len(U) <= 0 or len(link_fault_sets) <= 0:
        return np.full(n, np.nan)
        
    solver = Optimize()

    # 对所有故障链路集和 中的元素，创建布尔变量
    elements = {u: Bool(u) for u in U}

    # 加入限制，使得 最小一致故障集 中的每个子集至少包含 故障链路集和 中的一个元素
    for s in link_fault_sets:
        subset_contains_element = Or([elements[u] for u in s])
        solver.add(subset_contains_element)
        
    # 最小化被选中的元素的数量
    num_selected_elements = sum([If(elements[u], 1, 0) for u in U])
    solver.minimize(num_selected_elements)

    if solver.check() == sat:
        model = solver.model()
        selected_elements = [u for u in U if is_true(model[elements[u]])]
        links_state_inferred[selected_elements] = 1
        return np.array(links_state_inferred)
        
    else:
        return np.full(n, np.nan)

def get_paths(link: int, route_matrix: np.ndarray):
    assert link >= 0
    paths, = np.where(route_matrix[:, link] > 0)
    return paths.tolist()
    
def test_sat():
    #y = np.array([[0, 0, 0],
                  #[0, 0, 1],
                  #[1, 0, 0],
                  #[0, 1, 1],
                  #[1, 0, 0],
                  #[0, 0, 1],
                  #[1, 1, 0],
                  #[1, 1, 1]], dtype=np.int8).T
    
    y=np.array(
        [[1,1,1]],dtype=np.int8).T

    A_rm = np.array([
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1]], dtype=np.int8)
    
    links_state_inferred = alg_SAT(y, A_rm)
    print(links_state_inferred)

if __name__ == '__main__':
    test_sat()