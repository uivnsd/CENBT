import itertools
import sys
import types
import warnings
from copy import deepcopy
from functools import reduce
from itertools import product

import matplotlib.pylab as plt
import networkx as nx
import numpy as np
from ordered_set import OrderedSet
from tqdm import trange
from time import time


# noinspection NonAsciiCharacters
class 链路基类:
    def __init__(self, 名称: tuple):
        self.名称: tuple = 名称  # 元组类型的名称
        self.状态: bool = True  # 默认处于正常状态
        self.当前时刻: int = 0  # 记录当前运行的时刻

        self.先验拥塞概率: float = 0.0  # 默认不发生拥塞

    def __str__(self):
        return f'链路基类对象 {self.名称}'

    def __repr__(self):
        return f'链路基类对象 {self.名称}'

    # def __eq__(self, other):
    #     return self.名称 == other.名称

    def 生成状态(self, 运行时刻: int = 0):
        self.当前时刻 = deepcopy(运行时刻)  # 更新当前时刻

        self.状态 = False if np.random.rand() <= self.先验拥塞概率 else True
        return deepcopy(self.状态)


# noinspection NonAsciiCharacters
class 路径基类(链路基类):
    def __init__(self, 名称: tuple):
        super().__init__(名称)

        self.路由: list = []  # 集和元素为 链路基类 实例

    def __str__(self):
        return f'路径基类对象 {self.名称}'

    def __repr__(self):
        return f'路径基类对象 {self.名称}'

    def 计算先验拥塞概率(self):
        self.先验拥塞概率 = 1.0 - reduce(lambda i, j: i * j, [1.0 - link.先验拥塞概率 for link in self.路由])
        return deepcopy(self.先验拥塞概率)

    def 生成状态(self, 运行时刻: int = 0):
        self.当前时刻 = deepcopy(运行时刻)  # 更新当前时刻

        self.状态 = reduce(lambda i, j: i and j, [link.状态 for link in self.路由])  # True 为好；False 为坏
        return deepcopy(self.状态)


# noinspection NonAsciiCharacters
class 网络基类:
    def __init__(self):
        self.图文件名 = ''
        self.图: nx.DiGraph = nx.DiGraph([])

        self.名称: str = ''

        self.节点字典: dict = {
            '所有节点集和': OrderedSet([]),
            '边缘节点集和': OrderedSet([]),
            '内部节点集和': OrderedSet([])
        }

        self.链路集和: OrderedSet = OrderedSet([])
        self.路径集和: OrderedSet = OrderedSet([])

        self.路由矩阵: np.ndarray = np.array([])

        self.运行日志: dict = {
            '链路状态': np.empty_like([]),
            '路径状态': np.empty_like([]),
            '运行总时间': 0
        }

    def __str__(self):
        return f'网络基类对象 {self.名称}'

    def 配置拓扑(self, graph_gml: str):
        # 通过 topology zoo 提供的 gml 网络文件格式读取网络拓扑信息
        self.图文件名 = graph_gml + '.gml' if graph_gml[-4:] != '.gml' else graph_gml
        self.图 = nx.read_gml(self.图文件名)  # 最开始按照无向图来读取，最后再转成有向图

        # 读取拓扑名称
        try:
            self.名称 = self.图.graph['label'] if 'label' in self.图.graph.keys() else self.图.graph['Network']
        except KeyError:
            self.名称 = self.图文件名[:-4]

        # 移除度为 2 的点：会造成该点相邻的两条链路，不具有"端到端"可区分性
        link_src_list = [link[0] for link in self.图.edges.keys()]
        link_dst_list = [link[1] for link in self.图.edges.keys()]
        for node, degree in dict(self.图.degree()).items():
            if not self.图._node[node]['Internal']:  # 移除外部的虚拟节点
                self.图.remove_node(node)
            elif degree == 2 and node in link_src_list and node in link_dst_list:
                self.图.add_edge(*self.图.degree._nodes[node])
                self.图.remove_node(node)
            else:
                pass

        # 判断是否为连通图
        if not nx.is_connected(self.图):
        #if not nx.is_weakly_connected(self.图):
            warnings.warn("注意: 当前裁剪得到的图属于一个非连通图！", category=Warning)

        # 获取节点字典
        self.节点字典['所有节点集和'] = OrderedSet([node for node in self.图.nodes])
        self.节点字典['边缘节点集和'] = OrderedSet([node for node, degree in self.图.degree if degree <= 2])
        self.节点字典['内部节点集和'] = self.节点字典['所有节点集和'] - self.节点字典['边缘节点集和']

        # 转为有向图
        self.图 = self.图.to_directed()

        self.图.paths = []  # 向图实例中，新添加一个 paths 的列表属性

    def 部署测量路径(self, 源节点列表: list = [], 目的节点列表: list = []):
        """
        - 根据源目的节点列表，选定测量路径，以执行端到端测量
        - 输出测量覆盖的链路集和
        - 输出路由矩阵
        """
        if not OrderedSet(源节点列表).issubset(self.节点字典['边缘节点集和']):
            try:
                源节点列表 = [self.节点字典['所有节点集和'][i] for i in 源节点列表]
            except:
                warnings.warn("注意: 请检查所指定的源节点列表！", category=Warning)
                return
        else:
            warnings.warn("注意: 请检查所指定的源节点列表！", category=Warning)
            return

        if 目的节点列表 == []:
            目的节点列表 = list(self.节点字典['边缘节点集和'] - set(源节点列表))

        # 获取路径与链路集和
        path_set, self.图.paths, link_ref = dict(nx.all_pairs_shortest_path(self.图)), [], OrderedSet([])
        self.链路集和, self.路径集和 = OrderedSet([]), OrderedSet([])
        for src, dst in product(源节点列表, 目的节点列表):
            self.图.paths += [(src, dst)]
            self.路径集和.append(路径基类((src, dst)))

            route = path_set[src][dst]
            for link in [(i, j) for i, j in zip(route[:-1], route[1:])]:
                if link in link_ref:
                    no = link_ref.index(link)
                    self.路径集和[-1].路由.append(self.链路集和[no])
                else:
                    link_ref.append(link)
                    self.链路集和.append(链路基类(link))

                    self.路径集和[-1].路由.append(self.链路集和[-1])

        # 获取路径矩阵
        self.路由矩阵 = np.zeros((len(self.路径集和), len(self.链路集和)), dtype=bool)
        for i, path in enumerate(self.路径集和):
            for j, link in enumerate(self.链路集和):
                self.路由矩阵[i, j] = True if link in path.路由 else False

        index = 0  # 去掉"不具有NT可识别性"的链路，即对于承载了相同路径集和的所有链路而言，仅保留其中一条
        while index < self.路由矩阵.shape[1] - 1:
            to_remove = []
            for i in range(index + 1, self.路由矩阵.shape[1]):
                if all(self.路由矩阵[:, index] == self.路由矩阵[:, i]):
                    to_remove.append(i)

            self.路由矩阵 = np.delete(self.路由矩阵, to_remove, 1)
            for j in to_remove[::-1]:  # 通过倒序的方式来进行链路删除
                self.链路集和.pop(j)

            index += 1

    def 导出拓扑(self, img_name: str = None, with_path=False, with_name=False, with_geo=True):
        """
        - 默认只画拓扑，导出为 PNG 格式；若指定画出路径，则将所有的"端到端"路径全部高亮画出，并导出为 GIF 格式
        - 默认用节点对应的数字编号而非名称来标示每个节点
        - 默认使用节点的经纬度来画网络拓扑
        """
        img_name = deepcopy(self.名称) if img_name is None else img_name

        pos = {}
        if with_geo:
            for node, attr in self.图._node.items():
                pos[node] = np.array([attr['Longitude'], attr['Latitude']])
        else:
            pos = nx.kamada_kawai_layout(self.图)

        if with_path and self.图.paths != []:  # 画网络拓扑，并高亮测量路径
            import os, shutil, imageio

            if os.path.exists('./img'):
                shutil.rmtree('./img')
            os.makedirs('./img')

            frames = []
            for i, path in enumerate(self.图.paths):
                plt.figure()
                plt.title(f'{self.名称}\nPATH$_{{{i}}}$: {path[0]} --> {path[1]}')  # 使用LaTeX语句实现下标，注意有3层大括号

                nx.draw(self.图, pos,
                        with_labels=True,
                        node_color='white',
                        node_size=100,
                        edge_color='black',
                        labels=None if with_name else {node: idx for idx, node in enumerate(self.图.nodes.keys())})

                nx.draw_networkx_nodes(self.图, pos, nodelist=[path[0]], node_color='gold')
                nx.draw_networkx_nodes(self.图, pos, nodelist=[path[1]], node_color='grey', node_shape='s')

                path_index = self.图.paths.index(path)
                route = [link.名称 for link in self.路径集和[path_index].路由]
                nx.draw_networkx_edges(self.图, pos, edgelist=route, edge_color='red', width=2)

                plt.axis('tight')
                plt.savefig(f'./img/img_{i}.png')
                plt.close('all')

                image = imageio.v2.imread(f'./img/img_{i}.png')
                frames.append(image)

            imageio.mimsave('./' + img_name + '.gif', frames, duration=500)  # 间隔单位: ms
            shutil.rmtree('./img')

        else:
            plt.figure()
            plt.title(f'{self.名称}')

            nx.draw(self.图, pos,
                    with_labels=True,
                    node_color='white',
                    node_size=100,
                    edge_color='black',
                    arrows=False,  # 按无向边来做图
                    labels=None if with_name else {node: idx for idx, node in enumerate(self.图.nodes.keys())})

            plt.axis('tight')
            plt.savefig('./' + img_name + '.png')
            plt.close('all')

    def 配置参数(self, **kwargs):
        """
        可扩展，以支持配置更多的参数类别
        """
        if '异构链路先验拥塞概率' in kwargs.keys():
            if len([kwargs['异构链路先验拥塞概率']]) == 1:
                given_c_prob = deepcopy(kwargs['异构链路先验拥塞概率'])
                kwargs['异构链路先验拥塞概率'] = []
                np.random.seed(int(time()))
                for _ in self.链路集和:
                    c_prob = np.float16(np.random.rand())
                    kwargs['异构链路先验拥塞概率'].append(given_c_prob if c_prob > given_c_prob
                                                       else deepcopy(c_prob))

            for link, c_prob in zip(self.链路集和, kwargs['异构链路先验拥塞概率']):
                link.先验拥塞概率 = deepcopy(c_prob)

            for path in self.路径集和:
                path.计算先验拥塞概率()

        elif '同构链路先验拥塞概率' in kwargs.keys():
            if len([kwargs['同构链路先验拥塞概率']]) != 1:
                warnings.warn("注意: 当前只能指定同一先验拥塞概率值！", category=Warning)
                sys.exit(0)

            c_prob = kwargs['同构链路先验拥塞概率'] if 0.0 <= kwargs['同构链路先验拥塞概率'] <= 1.0 \
                else (1.0 if kwargs['同构链路先验拥塞概率'] > 1.0 else 0.0)
            for link in self.链路集和:
                link.先验拥塞概率 = deepcopy(c_prob)

            for path in self.路径集和:
                path.计算先验拥塞概率()

    def 运行网络(self, 运行的总时间: int = 1):
        self.运行日志['运行总时间'] = 运行的总时间

        self.运行日志['链路状态'] = np.zeros((len(self.链路集和), 运行的总时间), dtype=bool)
        self.运行日志['路径状态'] = np.zeros((len(self.路径集和), 运行的总时间), dtype=bool)
        for t in trange(运行的总时间, desc=f'网络-{self.名称}', leave=True):
            self.运行日志['链路状态'][:, t] = np.array([link.生成状态(运行时刻=t) for link in self.链路集和])
            self.运行日志['路径状态'][:, t] = np.array([path.生成状态(运行时刻=t) for path in self.路径集和])


# noinspection NonAsciiCharacters
class 攻击者基类:
    def __init__(self):
        self.被入侵的路径对象集和: OrderedSet = OrderedSet([])

        self.攻击日志: dict = {
            '攻击位置': [],
            '攻击时刻': []
        }

    @staticmethod
    def 随机攻击策略函数(路径状态_受攻击前: bool, 攻击概率: float = 0.1):
        """
        - 攻击概率，通过 __defaults__ 可以动态修改
        """

        路径状态_受攻击后 = 路径状态_受攻击前 ^ True if np.random.rand() <= 攻击概率 else deepcopy(路径状态_受攻击前)

        return 路径状态_受攻击后

    @staticmethod
    def 生成状态(self: 路径基类, 运行时刻: int = 0, 攻击策略函数: callable = lambda x: x, 攻击日志: dict = None):
        """
        - 通过"反转"路径的观测结果，完成攻击动作
        - 攻击策略函数，默认不攻击；通过 __defaults__ 可以动态修改
        - 攻击日志，默认不记录；通过 __defaults__ 可以动态修改
        """
        self.当前时刻 = deepcopy(运行时刻)  # 更新当前时刻

        路径状态_受攻击前 = reduce(lambda i, j: i and j, [link.状态 for link in self.路由])
        路径状态_受攻击后 = 攻击策略函数(路径状态_受攻击前)

        if 攻击日志 is not None and 路径状态_受攻击后 != 路径状态_受攻击前:
            攻击日志['攻击位置'].append(self.名称)
            攻击日志['攻击时刻'].append(运行时刻)

        self.状态 = deepcopy(路径状态_受攻击后)  # 回写攻击结果
        return deepcopy(self.状态)

    def 执行入侵(self, 受攻击的路径对象: 路径基类 = None, **kwargs):
        if 受攻击的路径对象 not in self.被入侵的路径对象集和:
            self.被入侵的路径对象集和.add(受攻击的路径对象)

        受攻击的路径对象.生成状态 = types.MethodType(self.生成状态, 受攻击的路径对象)

    def 清空(self, **kwargs):
        if "被入侵的路径对象集和" in kwargs.keys():
            self.被入侵的路径对象集和 = OrderedSet([])

        elif "攻击日志" in kwargs.keys():
            for i in self.攻击日志.keys():
                self.攻击日志[i] = []

        else:
            pass


def test():
    net = 网络基类()

    net.配置拓扑("./topology_zoo数据集/Chinanet.gml")
    net.部署测量路径(源节点列表=[2, 3])
    net.导出拓扑(with_path=True)  # 默认只画拓扑，不高亮任何路径

    net.配置参数(异构链路先验拥塞概率=0.3)

    attacker = 攻击者基类()
    # 采用随机攻击
    attacker.随机攻击策略函数.__defaults__ = (0.5,)  # 设置随机攻击机率为 0.5
    attacker.生成状态.__defaults__ = (attacker.随机攻击策略函数, attacker.攻击日志,)  # 植入随机攻击策略

    for path_id in [0, 1, 2]:
        attacker.执行入侵(net.路径集和[path_id])  # 入侵路径 0

    net.运行网络(运行的总时间=5)

    pass


if __name__ == '__main__':
    test()
