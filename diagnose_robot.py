import numpy as np
from alg_G_CALS import *
from alg_CENBT import *
from alg_map import *
from alg_SAT import *
from itertools import combinations
import multiprocessing as mp
from numba import cuda
import time as ttt
sys.path.append(os.path.join(os.path.dirname(__file__), 'topology_zoo'))
from TopologyZoo import *

def count_Fn_FPR_DR(real_array,inferred_array,n,return_TP=False):
    #inferred_array由算法得出,real_array即真实信息,n即表示计算的是F几,如计算F1就填1,计算F2就填2
    ave_FPR=0
    ave_DR=0
    ave_Fn=0
    rounds=len(real_array.T)
    #print('rounds:',rounds)
    for round in range(rounds):
        TP=0
        FP=0
        FN=0
        TN=0
        for link_index in range(len(real_array.T[round])):
            if real_array.T[round][link_index]==0 and inferred_array.T[round][link_index]==0:
                TN+=1
            elif real_array.T[round][link_index]==0 and inferred_array.T[round][link_index]==1:
                FP+=1
            elif real_array.T[round][link_index]==1 and inferred_array.T[round][link_index]==1:
                TP+=1
            elif real_array.T[round][link_index]==1 and inferred_array.T[round][link_index]==0:
                FN+=1
        if TP+FN==0:
            DR=1
        else:
            DR=TP/(TP+FN)
        if FP+TN==0:
            FPR=0
        else:
            FPR=FP/(FP+TN)
        if TP+FP==0:
            precision=1
        else:
            precision=TP/(TP+FP)
        if TP+FN==0:
            recall=1
        else:
            recall=TP/(TP+FN)
        if TP==0:
            if sum(real_array.T[round])==0:
                Fn=1
            else:
                Fn=0
        else:
            Fn=(1+n**2)*precision*recall/(n**2*precision+recall)
        ave_FPR+=FPR
        ave_DR+=DR
        ave_Fn+=Fn
    ave_FPR/=rounds
    ave_DR/=rounds
    ave_Fn/=rounds   
    if return_TP:
        return TP,FP,FN,TN
    return ave_Fn,ave_FPR,ave_DR

class diagnose_robot():
    #调度机器人
    def __init__(self,y:np.ndarray,x:np.ndarray,A_rm:np.ndarray,method:str,x_pc=None,n:int=1,diagnose_method:str='all',t:int=0,alpha:float=2,theta:float=0.5) -> None:
        #y为初始的观测状态,x为真实的链路状态,A_rm为机器人的状态转移矩阵,method为预测方法,x_pc为预测方法的参数(如有),Fn代表算的是F几，默认为1
        #认为传入的x,y,x_pc应该是一维数组

        #print('y:',y)
        #print('x:',x)
        self.congestion_link_num=np.sum(x)
        #记录总共有多少条链路是拥塞的

        if np.ndim(y) <= 1:  # 强制将观察转换为列向量
            y = y.reshape((-1, 1))
        
        if np.ndim(x) <= 1:  # 强制将真实链路状态转换为列向量
            x = x.reshape((-1, 1))
        self.y = y
        #print('y:',y)
        #print('y[0]:',y[0])
        #print('y[0]+10:',y[0]+10)
        #print('if y[0]==1:',y[0]==1)
        self.x = x
        self.A_rm = A_rm
        self.method = method
        self.x_pc = x_pc
        self.first_cvar=-1
        self.first_var=-1
        self.t=t
        self.alpha=alpha
        self.theta=theta
        
        self.DR=[]
        #DR为机器人的每次判断的dr的数组

        self.FPR=[]
        #FPR为机器人的每次判断的fpr的数组

        self.Fn=[]
        #F1为机器人的每次判断的f1的数组

        self.TP=[]
        self.FP=[]
        self.FN=[]
        self.TN=[]

        self.CVaRs=[]
        #CVaRs为机器人的每次判断的cvar的数组

        self.VaRs=[]
        #vars为机器人的每次判断的var的数组

        self.observed_links_num=0
        #记录诊断过程共诊断了多少个链路

        self.rounds=0
        #记录诊断过程共进行了多少轮诊断

        self.accuracy=0
        #记录总体诊断的准确性
        
        self.n=n
        #记录算的是F几
        #同时计算F0.5，1，2则传12

        self.diagnose_method=diagnose_method
        #记录使用的诊断方法,可选项有‘all’代表直接按照参考信息进行诊断,‘path’代表按照单条拥塞路径进行诊断,‘link_i’代表按照i条链路进行诊断

    def update(self) -> None:
        #根据输入确认预测方法更新机器人的状态,需要计算本次的dr,fpr,f1并存入DR,FPR,F1中
        #一轮结束需要更新A_rm,x_pc,y,去掉已经诊断的链路,并将rounds+1,observed_links_num加上这轮诊断的链路的数量
        check=np.where(self.y==1)[0]
        if len(check)==0:
            return -1
        #如果没有需要诊断的链路,则返回-1

        elif 'CENBT' in self.method:
            #'CENBT_90%'的格式
            beta=float(self.method.split('_')[-1][:-1])/100
            inferred_link_state,origin_info,cvar = alg_CENBT(self.y,self.A_rm,self.x_pc,alpha=self.alpha,beta=beta,theta=self.theta,return_cvar=True)
        elif '_' in self.method:
            #即如SAT_90%的格式
            method=self.method.split('_')[0]
            beta=float(self.method.split('_')[-1][:-1])/100
            if method=='map':
                inferred_link_state,_=alg_map(self.y,self.A_rm,self.x_pc)
            elif method=='SAT':
                inferred_link_state=alg_SAT(self.y,self.A_rm)
            elif method=='G-CALS':
                inferred_link_state=alg_G_CALS(self.y,self.A_rm,self.x_pc)
        else:
            print("invalid diagnose method")
            exit(-1)
        #区分要使用的诊断方法
        if self.diagnose_method=='all':
            #如果使用的是all方法,则直接按照参考信息进行诊断
            if self.n!=12 and self.n!=100:
                #一般
                round_Fn,round_FPR,round_DR=count_Fn_FPR_DR(self.x,inferred_link_state,self.n)
                TP=-1
                FP=-1
                FN=-1
                TN=-1
            #计算本轮的dr,fpr,fn
            elif self.n!=100:
                #self.n=12
                round_Fn,_,_= count_Fn_FPR_DR(self.x,inferred_link_state,1) #Fn用来记1
                round_FPR,_,_= count_Fn_FPR_DR(self.x,inferred_link_state,0.5) #FPR用来记0.5
                round_DR,_,_= count_Fn_FPR_DR(self.x,inferred_link_state,2) #DR用来记2
                TP=-1
                FP=-1
                FN=-1
                TN=-1
            else:
                #self.n=100
                TP,FP,FN,TN=count_Fn_FPR_DR(self.x,inferred_link_state,1,return_TP=True)
                round_Fn,round_FPR,round_DR=count_Fn_FPR_DR(self.x,inferred_link_state,2)

            self.TP.append(TP)
            self.FP.append(FP)
            self.FN.append(FN)
            self.TN.append(TN)
            self.DR.append(round_DR)
            self.FPR.append(round_FPR)
            self.Fn.append(round_Fn)

            round_cvar,round_var=calculate_cvar_new_per_t(self.y,self.A_rm,self.x_pc,inferred_link_state,alpha=self.alpha,beta=beta,return_var=True)
            print(f'{self.t} 计算cvar:{round_cvar}')

            self.CVaRs.append(round_cvar)
            if self.first_cvar==-1:
                self.first_cvar=round_cvar
            
            self.VaRs.append(round_var)
            if self.first_var==-1:
                self.first_var=round_var

            checked_links=np.where(inferred_link_state.T[0]==1)[0]
            
            self.observed_links_num+=len(checked_links)
            
            self.change_info(inferred_link_state)
        
        self.rounds+=1
        return self.rounds


    def change_info(self,inferred_link_state:np.ndarray) -> None:
        #根据输入的推断链路状态更新机器人的状态,更改A_rm,y,x,x_pc
        #完成两个步骤:1.去掉已经诊断的链路,2.将y内观察为通畅的路由上的链路都删掉(确认为通畅),3.根据更新后的A_rm和x推断出下一轮的y

        #1.去掉已经诊断的链路

        checked_links=np.where(inferred_link_state.T[0]==1)[0]
        #找到本轮检测的链路
        
        working_routers=np.where(self.y.T[0]==0)[0]
        #2.将y内观察为通畅的路由上的路由都删掉(确认为通畅)
        
        working_links=np.array([])
        for router in working_routers:
            working_links=np.append(working_links,np.where(self.A_rm[router,:]==1)[0])
        #寻找确认正常工作的链路

        to_delete_columns=np.unique(np.append(checked_links,working_links,axis=None))
        #找到需要删除的列,即状态已确认的链路

        to_delete_rows=working_routers
        #找到需要删除的行,即正常工作的路由

        to_delete_columns=np.array(to_delete_columns,dtype=int)
        to_delete_rows=np.array(to_delete_rows,dtype=int)
        #将序号转换为整数

        self.A_rm=np.delete(self.A_rm,to_delete_columns,axis=1)
        self.A_rm=np.delete(self.A_rm,to_delete_rows,axis=0)
        #删除A_rm内的对应行列

        self.x=np.delete(self.x,to_delete_columns,axis=0)
        #删除x内的对应链路

        self.x_pc=np.delete(self.x_pc,to_delete_columns,axis=0)
        #删除x_pc内的对应链路

        self.y=np.dot(self.A_rm,self.x)
        #根据更新后的A_rm和x推断出下一轮的y
        self.y = np.where(self.y > 1, 1, self.y)
        #将y内大于1的值置为1,防止一条拥塞路由内有多条拥塞链路
        

    def activate_robot(self):
        #激活机器人,开始诊断
        a=-2
        while(a!=-1):
            a=self.update()
        #print('断点3',ttt.time())
        if self.congestion_link_num==0:
            #如果本身就没有拥塞链路,则准确率为1
            self.accuracy=1
        else:
            self.accuracy=self.observed_links_num/self.congestion_link_num

if __name__=='__main__':
    rounds=10
    net=网络基类()
    net.配置拓扑("./topology_zoo/topology_zoo数据集/Chinanet.gml")
    net.部署测量路径(源节点列表=[2, 3])
    net.配置参数(异构链路先验拥塞概率=0.5)
    net.运行网络(运行的总时间=rounds)
    y = np.logical_not(net.运行日志['路径状态']).astype(int)
    A_rm = net.路由矩阵.astype(int)
    links_real_state = np.logical_not(net.运行日志['链路状态']).astype(int)

    link_congestion_prob = []
    for link in net.链路集和:
        link_congestion_prob.append(link.先验拥塞概率)
    link_congestion_prob = np.array(link_congestion_prob)
    
    for i in range(rounds):
        y_t=y[:,i]
        x_t=links_real_state[:,i]
        robot_t=diagnose_robot(y_t,x_t,A_rm,'SAT_all_98%',link_congestion_prob,2,'all',i,alpha=1.9,theta=0.5)
        robot_t.activate_robot()
        #if robot_t.rounds>=1:
            #print(robot_t.rounds,robot_t.observed_links_num,robot_t.accuracy)

    