import numpy as np
import copy


def alg_G_CALS(y: np.ndarray, A_rm: np.ndarray, x_pc: np.ndarray):
    if np.ndim(y) <= 1:  # 强制转换为列向量
        y = y.reshape((-1, 1))

    m, n = A_rm.shape
    _, num_time = y.shape

    x_identified = np.zeros((n, num_time))

    for i in range(num_time):
        paths_state_obv = y[:, i]
        links_state_infered = g_cals_a_groub(paths_state_obv, A_rm, x_pc)
        x_identified[:, i] = links_state_infered

    return np.int8(x_identified)


def g_cals_a_groub(y, A_rm, x_pc):

    tree_vector = rm_to_vector(A_rm)
    num_links = len(tree_vector)
    links_congest_pro = list(x_pc.flatten())

    route_matrix = A_rm

    paths_state_obv = copy.deepcopy(y)  # 观测路径状态数组

    links_state_inferred = diagnose(paths_state_obv, route_matrix, num_links, links_congest_pro)
    return links_state_inferred


def diagnose(paths_state_obv, route_matrix: np.ndarray, num_links, links_cong_pro: list):
    paths_cong_obv, paths_no_cong_obv = cal_cong_path_info(paths_state_obv)
    # print('链路的拥塞概率:', links_cong_pro)
    congested_path = (paths_cong_obv - 1).tolist()
    un_congested_path = (paths_no_cong_obv - 1).tolist()
    # print("congested_path",congested_path)
    # print("un_congested_path",un_congested_path)

    # 生成正常链路和不确定链路
    good_link, uncertain_link = get_link_state_class(un_congested_path, route_matrix, num_links)
    # print('位于不拥塞路径中的链路:', good_link)
    # print('不确定拥塞状态的链路:', uncertain_link)

    # 获取经过一条链路的所有路径domain
    domain_dict = {}
    for i in uncertain_link:
        domain_dict[i] = [j for j in get_paths(i + 1, route_matrix) if j in congested_path]
    # print("domain_dict")
    # print(domain_dict)

    links_state_inferred = np.zeros(num_links)
    links_cong_inferred = []
    # 计算所有的链路
    temp_state = [1e8 for _ in range(len(uncertain_link))]
    # print('temp_state:', temp_state)

    if not temp_state and len(congested_path):  # 如果存在无解的情形
        links_state_inferred = links_state_inferred + np.nan

    while temp_state and len(congested_path) > 0:
        # 找到最小的值对应的链路
        for index, i in enumerate(uncertain_link):
            a = np.log((1 - links_cong_pro[i]) / (links_cong_pro[i]))
            b = len(domain_dict[i])
            if b == 0:
                temp_state[index] = 1e8
            else:
                temp_state[index] = a / b

        index = temp_state.index(min(temp_state))
        links_state_inferred[uncertain_link[index]] = 1
        links_cong_inferred.append(uncertain_link[index] + 1)

        removed_path = []
        for item in domain_dict[uncertain_link[index]]:
            if item in congested_path:
                congested_path.remove(item)
                removed_path.append(item)           #记录移除的路径
        domain_dict.pop(uncertain_link[index])
        uncertain_link.remove(uncertain_link[index])
        temp_state.remove(temp_state[index])

        for p in removed_path:
            for l in range(num_links):
                if route_matrix[p][l] == 1 and l in uncertain_link:
                    temp_state.remove(temp_state[uncertain_link.index(l)])
                    uncertain_link.remove(l)


        for k, v in domain_dict.items():
            temp = []
            for i in v:
                if i in congested_path:
                    temp.append(i)
            domain_dict[k] = copy.deepcopy(temp)

    return links_state_inferred


def get_paths(link: int, route_matrix):
    """
    获取经过指定链路的所有路径。

    在路由矩阵中，第 0 列代表链路 1，第 1 列代表链路 2。依次类推。
    第 0 行代表路径 1，第 1 行代表路径 2。依次类推。
    :param link: 链路的编号
    :return:
    """
    assert link > 0
    paths, = np.where(route_matrix[:, link - 1] > 0)
    return paths.tolist()


def get_link_state_class(un_congested_path: list, route_matrix, num_links):
    """
    根据非拥塞路径，返回正常链路列表，和拥塞链路列表
    :param un_congested_path:list
    :return:good_link:list ,uncertain_link:list   存储链路下标
    """
    # 所有经过了不拥塞路径的链路
    good_link = []

    for i in un_congested_path:
        for index, item in enumerate(route_matrix[i]):
            if int(item) == 1 and index not in good_link:
                good_link.append(index)
    #print('good_link:',good_link)
    all_links = [i for i in range(num_links)]
    # 排除那些肯定不拥塞的链路
    uncertain_link = []
    for item in all_links:
        if item not in good_link:
            uncertain_link.append(item)
    return good_link, uncertain_link


def cal_cong_path_info(paths_state_obv):
    """
    根据路径的观测信息，计算拥塞路径和非拥塞路径
    :param paths_state_obv:
    :return:
    """
    paths_cong = []
    paths_no_cong = []
    for index in range(len(paths_state_obv)):
        if int(paths_state_obv[index]) == 1:
            # if int(self.path_states[index]) == 1:
            paths_cong.append(index + 1)
        else:
            paths_no_cong.append(index + 1)
    return np.array(paths_cong), np.array(paths_no_cong)


def rm_to_vector(A_rm: np.ndarray):
    """
    将路由矩阵转换为树向量
    :param A_rm:
    :return:
    """

    tree_vector = [0] * (A_rm.shape[1])

    for i in range(A_rm.shape[0]):
        path = A_rm[i]
        pre_node = 0
        for j in range(path.shape[0]):
            if path[j] == 1:
                tree_vector[j] = pre_node
                pre_node = j + 1

    return tree_vector


def test_G_cals_alg():
    y = np.array([[0, 0, 0],
                  [0, 0, 1],
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

    links_state_inferred = alg_G_CALS(y, A_rm, x_pc)
    print(links_state_inferred)


if __name__ == '__main__':
    test_G_cals_alg()
