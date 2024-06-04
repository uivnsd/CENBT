import gurobipy as gp
from gurobipy import GRB, LinExpr
from numba import cuda
import numpy as np
import multiprocessing as mp
import os
from copy import deepcopy
import time
from numba import cuda
import warnings

warnings.filterwarnings("ignore")

def comb_self_define(n: int, k: int):
    #计算n选k的组合数
    if n < k:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    c = 1
    for i in range(k):
        c = c * (n - i)
    for i in range(k):
        c = c // (i + 1)
    return c

def binlist_to_int(lst):
    # 将二进制列表转换为整数
    return int(''.join(map(str, lst[::-1])), 2)

@cuda.jit
def kernel_generate_and_verify_combinations(state_prob_t_array,state_array,x_pc,y_t,combination_array,index_offset,A_rm,undetermin_link_array):
    #根据切分的组合计算可能性
    
    tx=cuda.threadIdx.x
    #线程在block内的编号

    block_index=cuda.blockIdx.x
    #block的编号

    block_size=cuda.blockDim.x
    #block的大小

    grid_size=cuda.gridDim.x
    #grid的大小

    thread_index=tx+block_index*grid_size
    #计算进程在所有进程里的编号

    num_process=block_size*grid_size
    #总的进程数量

    k, m = divmod(len(combination_array), num_process)

    start=thread_index*k+min(thread_index, m)
    end=(thread_index+1)*k+min(thread_index+1, m)
    #计算进程需要处理的范围

    for i in range(start,end):
        #对于其范围内的所有状态

        comb_index=i+index_offset
        #计算需要处理的组合的索引

        #get_comb(len(undetermin_link_array),len(combination_array[i]),comb_index,combination_array[i])
        n=len(undetermin_link_array)
        k=len(combination_array[i])
        pre_n=n-1
        pre_k=k-1
        num=0
        tmp=comb_index
        for j in range(k-1):
            #进入第j层,确认第j个数

            #c=comb_self_define(pre_n,pre_k)
            tmp_n=pre_n
            tmp_k=pre_k
            if tmp_n<tmp_k:
                c=0
            elif tmp_k==0 or tmp_k==tmp_n:
                c=1
            else:
                if tmp_k>tmp_n-tmp_k:
                    tmp_k=tmp_n-tmp_k
                c=1
                for l in range(tmp_k):
                    c = c * (tmp_n - l)
                for l in range(tmp_k):
                    c = c // (l + 1)
                
            while tmp>=c:
                tmp-=c
                pre_n-=1
                num+=1
                #c=comb_self_define(pre_n,pre_k)
                tmp_n=pre_n
                tmp_k=pre_k
                if tmp_n<tmp_k:
                    c=0
                elif tmp_k==0 or tmp_k==tmp_n:
                    c=1
                else:
                    if tmp_k>tmp_n-tmp_k:
                        tmp_k=tmp_n-tmp_k
                    c=1
                    for l in range(tmp_k):
                        c = c * (tmp_n - l)
                    for l in range(tmp_k):
                        c = c // (l + 1)
                
            combination_array[i][j]=num
            num+=1
            pre_k-=1
            pre_n=n-num-1
        combination_array[i][k-1]=num+tmp
        #根据索引计算组合,并将组合存入combination_array,此时combination_array内的组合只是元素序号而已

        for j in range(len(combination_array[i])):
            #将序号转换为真正的组合
            combination_array[i][j]=undetermin_link_array[combination_array[i][j]]


    for i in range(start,end):
        #对于其范围内的所有状态

        flag=0

        for j in range(len(y_t)):
            #即对于每一个路由而言
            if flag:
                break
            if y_t[j]:
                #如果该路由有拥塞
                tmp=0
                for k in combination_array[i]:
                    #对于所有的在场景中被认为是拥塞状态的链路
                    tmp+=A_rm[j][k]
                if tmp==0:
                    #当前的scenario无法解释该路由的拥塞
                    flag=1
        
        if flag==0:
            #代表状态通过了检测
            prob=1
            for j in range(len(undetermin_link_array)):
                #prob*=(1-x_pc[j])
                prob*=(1-x_pc[undetermin_link_array[j]])
            for j in range(len(combination_array[i])):
                prob/=(1-x_pc[combination_array[i][j]])
                prob*=x_pc[combination_array[i][j]]

            state_prob_t_array[i]=prob
            #记录发生概率

            for j in range(len(combination_array[i])):
                state_array[i][combination_array[i][j]]=1
            #根据组合数组记录场景

def gen_state_prob_purning(y:np.ndarray,A_rm:np.ndarray,x_pc:np.ndarray,max_state:int,t:int):
    #生成第 t 次的状态和其可能性
    #y是观测矩阵,A_rm是路由矩阵,x_pc是先验概率,t是第t次,max_state为最大状态数
    num_paths, num_links = A_rm.shape
    if y.ndim == 1:  # 如果是横向量,则转为列向量
        Y_obs = np.reshape(y, (num_paths, -1))
    else:
        Y_obs = y

    link_state_obs=[0 for i in range(num_links)]

    for i in range(num_paths):
        if Y_obs[i,t]==0:
            for j in range(num_links):
                if A_rm[i,j]:
                    link_state_obs[j]=1
    undetermin_link_index=[]
    for l in range(len(link_state_obs)):
        if link_state_obs[l]==0:
            undetermin_link_index.append(l)
    #构建在第t次观察下的链路状态预测数组,分出未知状态的链路

    state_prob_t=[]
    link_state_t=[]

    if len(undetermin_link_index)==0:
        #如果没有未知状态的链路,直接返回
        state_prob_t=[1]
        link_state_t=[[0 for i in range(num_links)]]
        return state_prob_t,link_state_t
    
    
    if 0:
        pass
    else:
        #代表场景数量过大，需要剪枝
        #print(f"本t{t}要生成的情况总数为:",2**len(undetermin_link_index))

        calculated_scenario_num=0

        undetermin_link_array=np.array(undetermin_link_index)
        #将未知状态的链路的索引转换为nparray

        y_t=Y_obs[:,t]
        #一维nparray，代表第t时刻的观测数据

        y_t=np.ascontiguousarray(y_t)
        #将y_t变为连续的内存

        for choosen_link_num in range(len(undetermin_link_index)):
            if calculated_scenario_num>max_state and len(state_prob_t)>0:
                #如果已经超过了最大场景数量限制且已经有了场景,则停止
                break

            #用gpu生成组合
            start=0
            end=comb_self_define(len(undetermin_link_index),choosen_link_num+1)
            while start<end:
                #当未到达最后一种选择的组合时

                if start+2**20<end:
                    end_this_time=start+2**20
                else:
                    end_this_time=end
                #一次最多生成2**20个组合

                num_com=end_this_time-start
                #本轮循环要处理的组合数量

                combinations_array=np.zeros((num_com,choosen_link_num+1),dtype=np.int32)
                #开辟空间记录组合

                threads_per_block=128      #2**7
                blocks_per_grid=128

                while num_com<threads_per_block*blocks_per_grid:
                    threads_per_block/=2
                    blocks_per_grid/=2

                if threads_per_block<1:
                    threads_per_block=1
                    blocks_per_grid=1

                sub_state_prob_t_array=np.zeros(num_com)
                #开辟空间记录可能性

                sub_state_array=np.zeros((num_com,num_links),dtype=np.int8)
                #开辟空间记录场景

                kernel_generate_and_verify_combinations[int(blocks_per_grid),int(threads_per_block)](sub_state_prob_t_array,sub_state_array,x_pc,y_t,combinations_array,start,A_rm,undetermin_link_array)
                #生成组合并验证

                for i in range(len(sub_state_prob_t_array)):
                    if sub_state_prob_t_array[i]>0:
                        #代表该组合可能

                        state_prob_t.append(sub_state_prob_t_array[i])
                        #记录当前组合的可能性

                        link_state_t.append(list(sub_state_array[i]))
                        #记录该场景
                
                start=end_this_time
                #更新下一轮循环的开始位置
                
            calculated_scenario_num+=comb_self_define(len(undetermin_link_index),choosen_link_num+1)
            #增加完成判断的场景数量

        #print('生成场景完成')
        return state_prob_t,link_state_t
    
def alg_CENBT(y:np.ndarray,A_rm:np.ndarray,x_pc:np.ndarray,alpha:float=-1,beta:float=-1,theta:float=-1,return_cvar=False):
    num_paths, num_links = A_rm.shape
    if y.ndim == 1:  # 如果是横向量,则转为列向量
        Y_obs = np.reshape(y, (num_paths, -1))
    else:
        Y_obs = y
    
    if theta==-1:
        print('theta is not set, using default value 0.5')
        theta=0.5

    num_times = Y_obs.shape[-1]
    x_identify= np.zeros((num_links, num_times), dtype=np.int8)
    origin_x_identified = np.zeros((num_links, num_times), dtype=np.float64)
    cvars=np.zeros(num_times)

    if num_times==1:
        if not return_cvar:
            _,temp_origin_x_identified,temp_x_identified,_=alg_var_new_per_t(Y_obs[:,0],A_rm,x_pc,alpha,beta,theta=theta,t=0)
        else:
            _,temp_origin_x_identified,temp_x_identified,cvars[0]=alg_var_new_per_t(Y_obs[:,0],A_rm,x_pc,alpha,beta,theta=theta,t=0)
        x_identify[:,0]=np.int8(temp_x_identified)
        origin_x_identified[:,0]=temp_origin_x_identified
    else:
        already_identified = {} #小数据库
        next_start=0
        while next_start<num_times:
            to_do=[]
            for t in range(next_start,num_times):
                if binlist_to_int(Y_obs[:,t]) not in already_identified:
                    to_do.append(t)
                    if len(to_do)>=32:
                        break
                else:
                    print(f'{t}查表')
                    x_identify[:,t]=np.int8(already_identified[binlist_to_int(Y_obs[:,t])][1])
                    origin_x_identified[:,t]=already_identified[binlist_to_int(Y_obs[:,t])][0]
                    if return_cvar:
                        cvars[t]=already_identified[binlist_to_int(Y_obs[:,t])][2]
            
            next_start=t+1

            pool=mp.Pool()
            res=pool.starmap_async(alg_var_new_per_t,[(Y_obs[:,t],A_rm,x_pc,alpha,beta,theta,t) for t in to_do])
            pool.close()
            pool.join()

            for result in res.get():
                x_identify[:,result[0]]=np.int8(result[2])
                origin_x_identified[:,result[0]]=result[1]
                if return_cvar:
                    cvars[result[0]]=result[3]
                if binlist_to_int(Y_obs[:,result[0]]) not in already_identified.keys():
                    if return_cvar:
                        already_identified[binlist_to_int(Y_obs[:,result[0]])]=[result[1],result[2],result[3]]
                    else:
                        already_identified[binlist_to_int(Y_obs[:,result[0]])]=[result[1],result[2]]
    if not return_cvar:
        return x_identify,origin_x_identified
    else:
        return x_identify,origin_x_identified,cvars

@cuda.jit
def normalize_with_numba(states_probs, result,sum_state_prob):
    pos = cuda.grid(1)
    if pos < states_probs.size:  # 确保不会越界
        result[pos] = states_probs[pos] / sum_state_prob


def alg_var_new_per_t(y:np.ndarray,A_rm:np.ndarray,x_pc:np.ndarray,alpha:float=-1,beta:float=-1,theta:float=0.5,t=0):
    #print('进入算法',time.time())
    
    #处理全部的路径观测都通畅的情况
    if np.sum(y)==0:
        return t,np.zeros(len(x_pc),dtype=np.int8),np.zeros(len(x_pc),dtype=np.int8),0

    num_path,num_link=A_rm.shape
    #首先根据y删掉无用的链路和路径
    good_paths=np.where(y==0)[0]
    good_links=[]
    for path in good_paths:
        good_links+=np.where(A_rm[path,:]==1)[0].tolist()
    good_links=np.unique(good_links)

    #print(good_links)
    
    #记录剩余的链路序号
    good_links=np.array(good_links,dtype=int)
    remain_link_index=np.delete(np.arange(num_link),good_links)

    #print('remain_link_index: ',remain_link_index)

    A_rm_tmp=deepcopy(A_rm)
    x_pc_tmp=deepcopy(x_pc)
    y_tmp=deepcopy(y)
    A_rm_tmp=np.delete(A_rm_tmp,good_paths,axis=0)     #删除路径对应的行
    A_rm_tmp=np.delete(A_rm_tmp,good_links,axis=1)      #删除链路对应的列
    y_tmp=np.delete(y_tmp,good_paths,axis=0)      #删除路径对应的元素
    x_pc_tmp=np.delete(x_pc_tmp,good_links,axis=0)      #删除链路对应的元素

    if alpha==-1:
        alpha=1+np.mean(x_pc_tmp)
    print('alpha:',alpha)
    print('beta:',beta)

    states_probs,link_states=gen_state_prob_purning(y_tmp,A_rm_tmp,x_pc_tmp,max_state=2**32,t=0)

    #根据观测的概率从大到小进行重排序,截取到概率最大的场景的1/100或者1/10部分
    arg=np.argsort(states_probs)[::-1]
    break_point=-1
    for i in range(len(arg)):
        if states_probs[arg[i]]<states_probs[arg[0]]/10:
            break_point=i
            break
    if break_point==-1:
        break_point=len(arg)
    indices=np.array(arg[:break_point],dtype=int)
    states_probs=np.array(states_probs,dtype=np.float64)
    states_probs=states_probs[indices]
    link_states=[link_states[i] for i in indices]

    states_probs_gpu = cuda.to_device(states_probs)
    result_gpu = cuda.device_array_like(states_probs_gpu)
    sum_state_prob=np.sum(states_probs)
    

    threads_per_block = 256
    blocks_per_grid = (states_probs_gpu.size + (threads_per_block - 1)) // threads_per_block
    normalize_with_numba[blocks_per_grid, threads_per_block](states_probs_gpu, result_gpu,sum_state_prob)
    states_probs = result_gpu.copy_to_host()
    #print('归一化后:',list(states_probs))
    #归一化

    remain_path_nums=len(y_tmp)
    remain_link_nums=len(A_rm_tmp[0])

    #算每个链路的路由权重
    link_weights=np.zeros(remain_link_nums)
    for l in range(remain_link_nums):
        link_weights[l]=np.sum(A_rm_tmp[:,l])
    #print('link_weights: ',link_weights)
    

    #创建模型
    model=gp.Model('alg_var_new_per_t')
    model.setParam('OutputFlag',0)
    #print(os.cpu_count())
    model.setParam('Threads',os.cpu_count())
    model.setParam('MIPFocus', 1)  # 更加关注找到可行解
    model.setParam('Cuts', 2)      # 使用更激进的剪枝策略


    #print('开始求解',time.time())

    #创建变量
    X_inferred=model.addVars(remain_link_nums,lb=0,ub=1,vtype=GRB.CONTINUOUS,name='X')      #准备暗示的链路状态
    L_link=model.addVars(len(link_states),remain_link_nums,lb=0,vtype=GRB.CONTINUOUS,name='L_link')      #链路的损失
    L_s=model.addVars(len(link_states),lb=0,vtype=GRB.CONTINUOUS,name='L_s')      #场景的损失
    VaR=model.addVar(lb=0,vtype=GRB.CONTINUOUS,name='VaR')      #VaR

    #创建约束
    for p in range(remain_path_nums):
        model.addConstr(gp.quicksum(A_rm_tmp[p,l]*X_inferred[l] for l in range(remain_link_nums))>=y_tmp[p],name=f'path_{p}')        #相当于A_rm*X>=y,满足观测
    
    for s in range(len(link_states)):
        for l in range(remain_link_nums):
            model.addConstr(L_link[s,l]>=X_inferred[l]+(1-X_inferred[l])*link_states[s][l]*alpha,name=f'link_{s}_{l}')        #链路的损失
        model.addConstr(L_s[s]>=gp.quicksum(L_link[s,l] for l in range(remain_link_nums))-VaR,name=f'scene_{s}')        #场景的损失

    model.setObjective(VaR+(1/(1-beta))*gp.quicksum(states_probs[s]*L_s[s] for s in range(len(link_states))),GRB.MINIMIZE)      #目标函数

    model.optimize()
    #获取结果
    if model.status == GRB.OPTIMAL:
        result_X=[X_inferred[l].X for l in range(remain_link_nums)]
        cvar=model.ObjVal
        var=VaR.X
    else:
        raise Exception('Gurobi error')
    
    #将结果转为原始的链路序号
    result_X_origin=np.zeros(num_link,dtype=np.float64)
    result_X_origin[remain_link_index]=result_X

    print('theta:',theta)
    #根据theta进行判断
    temp_x_identified=deepcopy(result_X_origin)
    for l in range(num_link):
        if temp_x_identified[l]>theta:
            temp_x_identified[l]=1
        else:
            temp_x_identified[l]=0

    
    link_weights_all= np.zeros(num_link)
    link_weights_all[remain_link_index]=link_weights
    #重新扩充链路权重

    #判断是否有链路状态无法解释路径上的拥塞
    for p in range(num_path):
        if y[p]>0 and np.sum(A_rm[p,:]*temp_x_identified)==0:
            #路径p无法解释,选取路由权重乘以x_pc最大的链路

            l_max=-1
            l_max_index=-1
            for l in range(num_link):
                if A_rm[p,l]>0 and l not in good_links and x_pc[l]*link_weights_all[l]>l_max:
                    l_max=x_pc[l]*link_weights_all[l]
                    l_max_index=l
                    #寻找最大的链路权重乘以x_pc

            temp_x_identified[l_max_index]=1
            #将最大的链路权重乘以x_pc的链路设置为1，使得路径p可以解释
    return t,result_X_origin,temp_x_identified,cvar
            

def test_CENBT():
    y = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=np.int8).T

    #y=np.array([[0,0,0]],dtype=np.int8).T

    A_rm = np.array([
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1]], dtype=np.int8)
    
    x_pc = np.array([0.1] * 5)

    link_state_inferred,origin_info= alg_CENBT(y,A_rm,x_pc,alpha=1.5,beta=0.9,theta=0.5)
    print(link_state_inferred),print(origin_info)

    link_state_inferred,origin_info,cvars = alg_CENBT(y,A_rm,x_pc,alpha=1.5,beta=0.9,theta=0.5,return_cvar=True)
    print(link_state_inferred),print(origin_info),print(cvars)

def test_CENBT_specific_case():
    y=np.array([[1,1]],dtype=np.int8).T
    A_rm=np.array([
        [1,0,1],
        [1,1,0]
    ],dtype=np.int8)
    x_pc=np.array([0.1,0.2,0.25])
    link_state_inferred,origin_info= alg_CENBT(y,A_rm,x_pc,alpha=2,beta=0.9,theta=0.5)
    print(link_state_inferred),print(origin_info)

    state_inferred_1=np.array([1,1,1],dtype=np.int8).reshape(-1,1)
    print(state_inferred_1)
    cvar=calculate_cvar_new_per_t(y,A_rm,x_pc,state_inferred_1,2,0.9)
    print(cvar)

def calculate_cvar_new_per_t(y,A_rm,x_pc,target_inferred_result,alpha,beta,return_var=False):
    if np.sum(y)==0:
        return 0
    print('alpha:',alpha)
    print('beta:',beta)
    num_path,num_link=A_rm.shape
    #首先根据y删掉无用的链路和路径
    good_paths=np.where(y==0)[0]
    good_links=[]
    for path in good_paths:
        good_links+=np.where(A_rm[path,:]==1)[0].tolist()
    good_links=np.unique(good_links)

    #记录剩余的链路序号
    good_links=np.array(good_links,dtype=int)
    remain_link_index=np.delete(np.arange(num_link),good_links)

    A_rm_tmp=deepcopy(A_rm)
    x_pc_tmp=deepcopy(x_pc)
    y_tmp=deepcopy(y)
    A_rm_tmp=np.delete(A_rm_tmp,good_paths,axis=0)     #删除路径对应的行
    A_rm_tmp=np.delete(A_rm_tmp,good_links,axis=1)      #删除链路对应的列
    y_tmp=np.delete(y_tmp,good_paths,axis=0)      #删除路径对应的元素
    x_pc_tmp=np.delete(x_pc_tmp,good_links,axis=0)      #删除链路对应的元素

    if alpha==-1:
        alpha=1+np.mean(x_pc_tmp)

    states_probs,link_states=gen_state_prob_purning(y_tmp,A_rm_tmp,x_pc_tmp,max_state=2**32,t=0)

    #根据观测的概率从大到小进行重排序,截取到概率最大的场景的1/100或1/10部分
    arg=np.argsort(states_probs)[::-1]
    break_point=-1
    for i in range(len(arg)):
        if states_probs[arg[i]]<states_probs[arg[0]]/10:
            break_point=i
            break
    if break_point==-1:
        break_point=len(arg)
    indices=np.array(arg[:break_point],dtype=int)
    states_probs=np.array(states_probs,dtype=np.float64)
    states_probs=states_probs[indices]
    link_states=[link_states[i] for i in indices]

    states_probs_gpu = cuda.to_device(states_probs)
    result_gpu = cuda.device_array_like(states_probs_gpu)
    sum_state_prob=np.sum(states_probs)

    threads_per_block = 256
    blocks_per_grid = (states_probs_gpu.size + (threads_per_block - 1)) // threads_per_block
    normalize_with_numba[blocks_per_grid, threads_per_block](states_probs_gpu, result_gpu,sum_state_prob)
    states_probs = result_gpu.copy_to_host()

    remain_path_nums=len(y_tmp)
    remain_link_nums=len(A_rm_tmp[0])

    link_weights=np.zeros(remain_link_nums)
    for l in range(remain_link_nums):
        link_weights[l]=np.sum(A_rm_tmp[:,l])

    model=gp.Model('cal_cvar_new_per_t')
    model.setParam('OutputFlag',0)
    model.setParam('Threads',os.cpu_count())
    model.setParam('MIPFocus', 1)  # 更加关注找到可行解
    model.setParam('Cuts', 2)      # 使用更激进的剪枝策略

    #暗示的链路状态就是target_inferred_result,但需要按照remain_link_index的顺序变形
    target_inferred_result_tmp=np.zeros(remain_link_nums,dtype=np.float64)
    target_inferred_result_tmp=target_inferred_result[remain_link_index]
    L_link=model.addVars(len(link_states),remain_link_nums,lb=0,vtype=GRB.CONTINUOUS,name='L_link')      #链路的损失
    L_s=model.addVars(len(link_states),lb=0,vtype=GRB.CONTINUOUS,name='L_s')      #场景的损失
    VaR=model.addVar(lb=0,vtype=GRB.CONTINUOUS,name='VaR')      #VaR

    for s in range(len(link_states)):
        for l in range(remain_link_nums):
            model.addConstr(L_link[s,l]>=target_inferred_result_tmp[l]+(1-target_inferred_result_tmp[l])*link_states[s][l]*alpha,name=f'link_{s}_{l}')        #链路的损失
        model.addConstr(L_s[s]>=gp.quicksum(L_link[s,l] for l in range(remain_link_nums))-VaR,name=f'scene_{s}')        #场景的损失

    model.setObjective(VaR+(1/(1-beta))*gp.quicksum(states_probs[s]*L_s[s] for s in range(len(link_states))),GRB.MINIMIZE)      #目标函数

    model.optimize()
    #获取结果
    if model.status == GRB.OPTIMAL:
        cvar=model.ObjVal
        var=VaR.X
    else:
        raise Exception('Gurobi error')
    if not return_var:
        return cvar
    else:
        return cvar,var

if __name__=='__main__':
    test_CENBT()
    test_CENBT_specific_case()