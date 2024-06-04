from diagnoser import *
import pickle
import time as ttt
def diagnose_method_compare(scenario_prob:list,observe_times:int,topology_name:str,source_nodes:list,lb:float,ub:float,alpha:float,theta:float):
    net=网络基类()
    net.配置拓扑(f"./topology_zoo/topology_zoo数据集/{topology_name}.gml")
    net.部署测量路径(源节点列表=source_nodes)

    diagnose_mathods=['all']
    methods=['CENBT_98%','SAT_98%','map_98%','G-CALS_98%']
    data={}
    if os.path.exists("./result_diagnose_method_v2")==False:
        os.mkdir("./result_diagnose_method_v2")
    keys=['DRs','FPRs','F100s','accuracys','observed_links_nums','costs','rounds','CVaRs','first_cvars','TPs','FPs','FNs','TNs','first_vars']
    draw_keys=['DRs','FPRs','F100s','accuracys','observed_links_nums','costs','rounds','CVaRs','first_cvars','first_vars']

    for s in range(len(scenario_prob)):
        print('#####################################################')
        #print(congestion_prob)
        #input()
        # net.配置参数(异构链路先验拥塞概率=congestion_prob)
        net.配置参数(指定先验拥塞概率=scenario_prob[s])

        A_rm = net.路由矩阵.astype(int)

        link_congestion_prob = []
        for link in net.链路集和:
            #print(link.先验拥塞概率)
            #input()
            link_congestion_prob.append(link.先验拥塞概率)
        link_congestion_prob_array=np.array(link_congestion_prob)

        with open(f"./result_diagnose_method_v2/scenarios.txt",'a') as f:
            f.write(f'index{s}:{link_congestion_prob}\n')

        net.运行网络(运行的总时间=observe_times)

        y = np.logical_not(net.运行日志['路径状态']).astype(int)
        #print(y)
        links_real_state = np.logical_not(net.运行日志['链路状态']).astype(int)
        #print(links_real_state)

        for method in methods:
            for diagnose_method in diagnose_mathods:

                print(f"topology:{topology_name},scenario:{s},method:{method},diagnose_method:{diagnose_method}")
                random_seed=np.random.randint(0,2**20)
                #print(y)
                #print(links_real_state)
                diagnoser_tmp=diagnoser(topology_name=topology_name,randomseed=random_seed,y=y,x=links_real_state,A_rm=A_rm,method=method,x_pc=link_congestion_prob_array,Fn=100,diagnose_method=diagnose_method,alpha=alpha,theta=theta)
                diagnoser_tmp.run_diagnose()

                #print(diagnoser_tmp.data)
                for key in keys:
                    if key in draw_keys:
                        data[method+'_'+diagnose_method+'_'+key]=data.get(method+'_'+diagnose_method+'_'+key,[])+[np.mean(diagnoser_tmp.data[key])]
                    else:
                        data[method+'_'+diagnose_method+'_'+key]=data.get(method+'_'+diagnose_method+'_'+key,[])+[diagnoser_tmp.data[key]]

                diagnoser_tmp=None
                #回收内存


        
        #保存data变量为pickle
        with open(f"./result_diagnose_method_v2/{topology_name}_data_{lb}_{ub}_{alpha}_{theta}_{s}.pickle",'wb') as f:
            pickle.dump(data,f)
        if s!=0:
            os.remove(f"./result_diagnose_method_v2/{topology_name}_data_{lb}_{ub}_{alpha}_{theta}_{s-1}.pickle")
        
        for key in draw_keys:
            plt.figure()
            for method in methods:
                for diagnose_method in diagnose_mathods:
                    #为每种key画出图像
                    x=list(range(s+1))
                    y=data[method+'_'+diagnose_method+'_'+key]
                    plt.plot(x,y,label=method+'_'+diagnose_method)
            plt.title(key)
            plt.xlabel('scenario_index')
            plt.ylabel('ave_'+key)
            plt.legend()
            plt.savefig(f"./result_diagnose_method_v2/{topology_name}_{key}_{lb}_{ub}.png")
            plt.clf()
             


if __name__=='__main__':
    start_time=ttt.time()
    test_sum_cvar=False
    tp_name = 'Chinanet'
    lb=0
    ub=0.1
    alpha=1.9
    theta=0.5
    if lb==0:
        lb+=1e-6
    with open(f'topology_probs/new_Probs_{tp_name}_{lb}_{ub}.pickle', 'rb') as fp:
        data = pickle.load(fp)

    scenarios_prob = data['Probs']
    topology_name = data['topology_name']
    source_nodes = data['source_nodes']
    assert tp_name == topology_name
    diagnose_method_compare(scenario_prob=scenarios_prob,observe_times=5000,topology_name=topology_name,source_nodes=source_nodes,lb=lb,ub=ub,alpha=alpha,theta=theta)

    end_time=ttt.time()
    duration=end_time-start_time
    print(f"duration:{duration}")
    