from diagnose_robot import *
import math
import time as ttt
from multiprocessing import Lock,Manager,Process
import pickle
import sys

class diagnoser():
    def __init__(self,topology_name:str,randomseed:int,y:np.ndarray,x:np.ndarray,A_rm:np.ndarray,method:str,x_pc=None,Fn:int=1,alpha:float=1.5,diagnose_method:str='all',theta:float=0.5) -> None:
        #y为初始的观测状态,x为真实的链路状态,A_rm为机器人的状态转移矩阵,method为预测方法,x_pc为预测方法的参数(如有),Fn代表算的是F几，默认为1
        #diagnose_method为调度方法，可选'all'
        #认为传入的x,y,x_pc应该是一维数组
        #如果要同时计算F0.5，1，2，则传12

        if np.ndim(y) <= 1:  # 强制将观察转换为列向量
            y = y.reshape((-1, 1))
        
        if np.ndim(x) <= 1:  # 强制将真实链路状态转换为列向量
            x = x.reshape((-1, 1))

        self.y = y
        self.x = x
        self.A_rm = A_rm
        self.method=method
        self.x_pc=x_pc
        self.n=Fn
        self.topology_name=topology_name
        self.random_seed=randomseed
        self.to_do=[]
        self.diagnose_method=diagnose_method
        self.alpha=alpha
        self.theta=theta
        #记录数据

        self.observe_times=self.y.shape[1]
        #记录总共观测了多少次

        self.info={}
        #记录已经计算完成的数据，防止重复计算

        self.manager=Manager()
        self.lock=Lock()

        self.DRs=self.manager.list([0 for _ in range(self.observe_times)])
        #记录每次观测的dr

        self.FPRs=self.manager.list([0 for _ in range(self.observe_times)])
        #记录每次观测的fpr

        self.Fns=self.manager.list([0 for _ in range(self.observe_times)])
        #记录每次观测的fn

        self.rounds=self.manager.list([0 for _ in range(self.observe_times)])
        #记录每次观测的rounds

        self.observed_links_nums=self.manager.list([0 for _ in range(self.observe_times)])
        #记录每次观测的诊断链路数量

        self.accuracys=self.manager.list([0 for _ in range(self.observe_times)])
        #记录每次观测的准确性

        self.CVaRs=self.manager.list([0 for _ in range(self.observe_times)])
        #记录每次观测的CVaR

        self.first_cvars=self.manager.list([0 for _ in range(self.observe_times)])

        self.VaRs=self.manager.list([0 for _ in range(self.observe_times)])
        #记录每次观测的VaR

        self.first_vars=self.manager.list([0 for _ in range(self.observe_times)])

        self.TPs=self.manager.list([[] for _ in range(self.observe_times)])
        self.FPs=self.manager.list([[] for _ in range(self.observe_times)])
        self.FNs=self.manager.list([[] for _ in range(self.observe_times)])
        self.TNs=self.manager.list([[] for _ in range(self.observe_times)])

        #此时TPs,FPs,FNs,TNs的格式为
        #datas[TP或FP或FN对应的key][第几个场景][第几个t]
        #查询出来的是一个list，如datas['SAT_98%_all_TPs'][0][1]=[2, 1]
        #代表SAT算法在第0个场景的第1个t中进行了两轮诊断,TP分别是2和1

        self.data={}


    def diagnose(self) ->None:
        #进行诊断
        #set_start_method('spawn')

        already_done=0
        while already_done<self.observe_times:
            #如果还有观测没有进行诊断
            self.to_do=[]
            #记录本次要并发处理的t
            
            for t in range(already_done,self.observe_times):
                if binlist_to_int(self.x[:,t]) not in self.info.keys():
                    #如果这个t的链路状态没有计算过
                    self.to_do.append(t)
                    #将t加入到本次要并发处理的t中
                    if len(self.to_do)>=32:
                        break
                    #每次最多同时计算64个t的数据

                else:
                    #如果场景已经被计算过:
                    print(f'{t}查表')
                    self.DRs[t]=self.info[binlist_to_int(self.x[:,t])][0]
                    self.FPRs[t]=self.info[binlist_to_int(self.x[:,t])][1]
                    self.Fns[t]=self.info[binlist_to_int(self.x[:,t])][2]
                    self.rounds[t]=self.info[binlist_to_int(self.x[:,t])][3]
                    self.observed_links_nums[t]=self.info[binlist_to_int(self.x[:,t])][4]
                    self.accuracys[t]=self.info[binlist_to_int(self.x[:,t])][5]
                    self.CVaRs[t]=self.info[binlist_to_int(self.x[:,t])][6]
                    self.first_cvars[t]=self.info[binlist_to_int(self.x[:,t])][7]
                    self.VaRs[t]=self.info[binlist_to_int(self.x[:,t])][8]
                    self.first_vars[t]=self.info[binlist_to_int(self.x[:,t])][9]
                    self.TPs[t]=self.info[binlist_to_int(self.x[:,t])][10]
                    self.FPs[t]=self.info[binlist_to_int(self.x[:,t])][11]
                    self.FNs[t]=self.info[binlist_to_int(self.x[:,t])][12]
                    self.TNs[t]=self.info[binlist_to_int(self.x[:,t])][13]

            already_done=t+1
            #更新已经诊断的次数

            processes=[]
            processes=[Process(target=diagnose_t,args=(t,self.DRs,self.FPRs,self.Fns,self.CVaRs,self.first_cvars,self.VaRs,self.first_vars,self.rounds,self.observed_links_nums,self.accuracys,self.method,self.x_pc,self.A_rm,self.x[:,t],self.y[:,t],self.n,self.diagnose_method,self.TPs,self.FPs,self.FNs,self.TNs,self.alpha,self.theta,self.lock)) for t in self.to_do]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            #并发处理
                

            for t in self.to_do:
                if binlist_to_int(self.x[:,t]) not in self.info.keys():
                    self.info[binlist_to_int(self.x[:,t])]=[self.DRs[t],self.FPRs[t],self.Fns[t],self.rounds[t],self.observed_links_nums[t],self.accuracys[t],self.CVaRs[t],self.first_cvars[t],self.VaRs[t],self.first_vars[t],self.TPs[t],self.FPs[t],self.FNs[t],self.TNs[t]]
            #将本次计算的数据加入到info中
    
    def save_info_and_calculate_cost(self):
        self.costs=[self.rounds[t]*self.accuracys[t] for t in range(self.observe_times)]
        self.scores=[self.Fns[t]*math.exp(-self.costs[t]) for t in range(self.observe_times)]

        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists(f'data/{self.topology_name}'):
            os.mkdir(f'data/{self.topology_name}')

        #将所有的数据打包成一个dict
        data={}
        data['name']=self.topology_name
        data['random_seed']=self.random_seed
        data['x_pc']=self.x_pc
        data['DRs']=list(self.DRs)
        data['FPRs']=list(self.FPRs)
        data[f'F{self.n}s']=list(self.Fns)
        data['rounds']=list(self.rounds)
        data['observed_links_nums']=list(self.observed_links_nums)
        data['accuracys']=list(self.accuracys)
        data['costs']=list(self.costs)
        data['scores']=list(self.scores)
        data['CVaRs']=list(self.CVaRs)
        data['first_cvars']=list(self.first_cvars)
        data['VaRs']=list(self.VaRs)
        data['first_vars']=list(self.first_vars)
        data['TPs']=list(self.TPs)
        data['FPs']=list(self.FPs)
        data['FNs']=list(self.FNs)
        data['TNs']=list(self.TNs)

        #将数据保存
        self.data=data
    
    def run_diagnose(self):
        #print('开始诊断')
        self.diagnose()
        #print('诊断完成')
        self.save_info_and_calculate_cost()




def diagnose_t(t:int,DRs,FPRs,Fns,CVaRs,first_cvars,VaRs,first_vars,rounds,observed_links_nums,accuracys,method,x_pc,A_rm,x_t,y_t,n,diagnose_method,TPs,FPs,FNs,TNs,alpha,theta,lock):
    #排查第t次观测的数据
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(t%2))
    #根据计算的t来决定使用哪张显卡

    robot=diagnose_robot(y_t,x_t,A_rm,method,x_pc,n,diagnose_method,t=t,alpha=alpha,theta=theta)
    #定义排查信息

    robot.activate_robot()
    #开始排查

    lock.acquire()
    try:
        if robot.rounds!=0:
            DRs[t]=sum(robot.DR)/robot.rounds
            FPRs[t]=sum(robot.FPR)/robot.rounds
            Fns[t]=sum(robot.Fn)/robot.rounds
            CVaRs[t]=sum(robot.CVaRs)/robot.rounds
            first_cvars[t]=robot.first_cvar
            VaRs[t]=sum(robot.VaRs)/robot.rounds
            first_vars[t]=robot.first_var
            TPs[t]=robot.TP
            FPs[t]=robot.FP
            FNs[t]=robot.FN
            TNs[t]=robot.TN
        else:
            DRs[t]=1
            FPRs[t]=0
            Fns[t]=1
            CVaRs[t]=0
            first_cvars[t]=0
            VaRs[t]=0
            first_vars[t]=0
            TPs[t]=[0]
            FPs[t]=[0]
            FNs[t]=[0]
            TNs[t]=[0]
        rounds[t]=robot.rounds
        observed_links_nums[t]=robot.observed_links_num
        accuracys[t]=robot.accuracy
    finally:
        lock.release()
        sys.exit(0)
    


