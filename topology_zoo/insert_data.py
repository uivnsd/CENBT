from TopologyZoo import *
from pymongo import MongoClient

remote_uri="mongodb://nr525:nr525@10.21.220.37:27017"


def insert_data(topology_name:str,先验概率类型:bool,run_time:int,先验概率:float,源节点列表:list):
    client=MongoClient(remote_uri)

    db=client['net']
    collection=db['net']
    collection.create_index([('ID', 1)], unique=True)

    net = 网络基类()
    net.配置拓扑(f"./topology_zoo数据集/{topology_name}.gml")
    net.部署测量路径(源节点列表=源节点列表)
    if 先验概率类型:
        #先验概率类型为True代表异构
        net.配置参数(异构链路先验拥塞概率=先验概率)
    else:
        net.配置参数(同构链路先验拥塞概率=先验概率)
    net.运行网络(run_time)

    y=np.where(net.运行日志['路径状态'],0,1)
    A_rm=np.where(net.路由矩阵,1,0)
    links_real_state=np.where(net.运行日志['链路状态'],0,1)
    link_congestion_prob=[]
    for link in net.链路集和:
        link_congestion_prob.append(float(link.先验拥塞概率))

    A_rm_new=[]
    for i in range(len(A_rm)):
        A_rm_new.append([])
        for j in range(len(A_rm[i])):
            A_rm_new[i].append(int(A_rm[i,j]))
    
    links_real_state_new=[]
    for i in range(len(links_real_state)):
        links_real_state_new.append([])
        for j in range(len(links_real_state[i])):
            links_real_state_new[i].append(int(links_real_state[i,j]))

    y_new=[]
    for i in range(len(y)):
        y_new.append([])
        for j in range(len(y[i])):
            y_new[i].append(int(y[i,j]))

    result = collection.aggregate([
    {
        '$group': {
            '_id': None,
            'maxField': { '$max': '$ID' }
        }
    }
    ])
    for doc in result:
        new_ID=doc['maxField']
    data={
        "ID":new_ID+1,
        "拓扑名称":topology_name,
        "源节点列表":源节点列表,
        "是异构的先验概率?":先验概率类型,
        "先验概率":先验概率,
        "链路先验概率集合":link_congestion_prob,
        "运行时间":run_time,
        "路由矩阵":A_rm_new,
        "链路真实状态":links_real_state_new,
        "路径状态":y_new
    }
    #print(data)
    result=collection.insert_one(data)
    print(f"插入成功，插入的ID为:{new_ID+1}")
    #print(result)
    client.close()

if __name__=='__main__':
    for i in range(100):
        insert_data('Chinanet',True,10,0.3,[2,3])