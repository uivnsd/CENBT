from TopologyZoo import *
import random, os, imageio
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

def repair_and_get_data(W_next,Z,graph):
    # 修复链路与计算FPR和DR
    
    #计算每轮的FPR和dR
    #total_bad_link记录当前总的坏链路的数量
    total_bad_link=0
    for i in range(len(Z)):
        if Z[i]==0:
            total_bad_link+=1
    #inspected记录本轮被检查的链路的数量
    inspected=len(W_next)
    #originally_good记录本轮检测前就为good的链路的数量,bad_link记录本轮检测的链路里有多少条是bad的
    originally_good=0
    bad_link=0
    for j in range(len(W_next)):
        W_index = next((i for i, x in enumerate(list(graph.nodes)) if x == W_next[j]), None)
        if Z[W_index]==1:
            #本来就是good的
            originally_good+=1
        else:
            #确实是bad的链路
            bad_link+=1
            Z[W_index]=1    #修复链路
    if inspected!=0:
        FPR=originally_good/inspected
        detection_rate=bad_link/total_bad_link
    else:
        FPR=None
        detection_rate=None
    return FPR,detection_rate,Z

def generate_Z(graph,root,proportion=1):
    #产生一个随机的Z数组,现在还是完全随机，以后可以修改,其中proportion为比例，0.8，0.6之类的
    Z=[]
    for i in range(len(list(graph.nodes))):
        tmp = random.choices([0, 1], weights=[1-proportion,proportion], k=1)[0]
        Z.append(tmp)
    root_index=next((i for i,x in enumerate(list(graph.nodes)) if x==root), None)
    Z[root_index] = 1     #设置根节点的链路一定是好的
    return Z
    #return [0,1,1,1, 1,0,1, 1,0 ,0,0,0,0]

def compute_X(graph, Z, root):
    #根据生成的Z数组计算X

    leaves = [node for node in list(graph.nodes) if graph.out_degree(node) == 0]      #找到所有的叶子节点

    X = np.full(len(list(graph.nodes)), -1)     #创建一个全为-1的1维数组为X
    for leaf in leaves:
        path = nx.shortest_path(graph, source=root, target=leaf)
        tmp = 1
        for node in path:
            node_index = next((i for i, x in enumerate(list(graph.nodes)) if x == node), None)
            tmp *= Z[node_index]      #计算X
        leaf_index = next((i for i, x in enumerate(list(graph.nodes)) if x == leaf), None)
        X[leaf_index] = tmp
    return X

def recurse(graph,k,Y,X,W,root):
    #graph为图，k为当前进行到的节点
    #如示例中国网中k应该是节点的名称，而不是单纯的序号
    k_index=next((i for i,x in enumerate(list(graph.nodes)) if x==k),None)
    if graph.out_degree(k) == 0:    #判断是不是叶子节点
       Y[k_index] = X[k_index]
       
    else:   #不是叶子节点就遍历孩子节点
        children = graph.successors(k)
        
        for child in children:
            # print("child1", child)
            recurse(graph,child,Y,X,W,root)
            
        children = graph.successors(k)
        children = list(children)
        
        if k != root:
            for child in children:
                child_index = next((i for i,x in enumerate(list(graph.nodes)) if x==child),None)
                Y[k_index] = max(Y[k_index], Y[child_index])
        for child in children:
            child_index = next((i for i,x in enumerate(list(graph.nodes)) if x==child),None)
            if Y[k_index]==1 and Y[child_index]==0:     #认为这个child节点有问题
                W.append(child)
                #W_index.append(child_index)
    return
                
def SCFS_on_tree(graph, X, root):
    #graph为nx类的图，X[k_index]即为X_k变量，Y[k_index]为Y_k变量
    #root为根节点的代号
    is_tree = nx.is_tree(graph)
    if not is_tree:
        print("错误，检测图是否为树状")
        nx.draw(graph,with_labels=True)
        print(graph.nodes)
        
        isolated_nodes = []
        for node in graph.nodes():
            if graph.in_degree(node) == 0 and graph.out_degree(node) == 0:
                isolated_nodes.append(node)
        print("隔离的点：",isolated_nodes)
        error_node=[]
        for node in graph.nodes():
            if graph.in_degree(node) >1:
                error_node.append(node)
        print("入度大于1的点：",error_node)
        plt.show()
        exit(-1)
    
    #初始化部分
    nodes = list(graph.nodes)
    root_index = nodes.index(root) 
    Y = np.zeros(len(list(graph.nodes)))      #创建一个全为0的1维数组为Y
    Y[root_index] = 1   #设置根节点一定是通畅的
    W = []        #创建空的W数组

    #从根开始递归
    recurse(graph, root, Y, X, W, root)
    #
    return W

# 可视化树状图
def visualize_tree(graph, W, round_num,dir):
 
    # 绘制图形（以树状排列）
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')  # 使用Graphviz的dot排列节点
    nx.draw(graph, pos, with_labels=True, node_size=500, font_size=8, font_weight='bold', edge_color='gray', node_color='green')

    # 将最小一致故障机中的节点 设为红色
    nx.draw_networkx_nodes(graph, pos, nodelist = W, node_color='red', node_size=500)

    plt.savefig(f'{dir}/round{round_num}.png')

def create_gif_from_images(image_folder, gif_filename):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.png') and filename.startswith('round'):  # 仅处理PNG图像，且以round开头
            file_path = os.path.join(image_folder, filename)
            images.append(Image.open(file_path))
    gif_filepath = os.path.join(image_folder, gif_filename)
    imageio.mimsave(gif_filepath, images, duration=2)  # 设置每帧间隔的时间，单位为秒



def simulation(graph, root,Data_, output=True, proportion=1):
    #对SCFS进行模拟

    round_num = 0
    # 生成Z数组
    Z = generate_Z(graph, root,proportion=proportion)
    if output:
        print("生成的Z数组:", Z) #展示生成的Z数组，后面可以弄得更直观点
        print("fail 的链路:",end='')
        failed_links=[]
        for i in range(len(Z)):
            if Z[i]==0:
                failed_links.append(list(graph.nodes)[i])
        print(failed_links)
    if output:
        dir='./scfs'
        if not os.path.exists(dir):    #运行之前检测一下路径存不存在，不存在就创建
            os.makedirs('./scfs')
        if not os.path.isfile(os.path.join(dir, 'counter.txt')):
            with open(os.path.join(dir, 'counter.txt'), 'w') as f:  #记录counter文件保存次数
                f.write('1')
        with open(os.path.join(dir, 'counter.txt'), 'r') as f:      #存在就直接读文件
            word = ""
            while True:
                char=f.read(1)
                if char==" "or char=="\n"or char=='':
                    break
                word+=char
            c=int(word)
        with open(os.path.join(dir, 'counter.txt'), 'w') as f:
            f.write(str(c + 1))
        newdir = os.path.join(dir, str(c))      #创建第n次模拟的路径和文件夹
        os.mkdir(newdir)


    W = [root]
    while W or round_num == 0:
        X = compute_X(graph, Z, root)         #计算X
        W_next = [] 
        for node in W:
            tmp = SCFS_on_tree(graph, X, node)        
            W_next.extend(tmp)
        
        

        # 如果 W' 不为空, 修复链路
        FPR,detection_rate,Z=repair_and_get_data(W_next, Z,graph)

        if output and isinstance(FPR,(int, float, complex, np.number)):
            print("本轮的FPR为:",FPR,"本轮的dR为:",detection_rate)
        if round_num==0 and isinstance(FPR,(int, float, complex, np.number)):
            Data_[proportion] = {'FPR': FPR, 'DR': detection_rate}      #根据图五，应该只画第一轮的数据
        #elif round_num==0:
            #Data_[proportion] = {'FPR': 0, 'DR': 1}

        W = W_next                      # 将此轮求得的最小一致故障集(下一轮的根节点)赋值给 W
        if output:
            print("W", W)
        round_num += 1
        if output:
            visualize_tree(graph, W, round_num,newdir) # 进行可视化处理
    if output:
        create_gif_from_images(newdir, 'scfs.gif') # 生成GIF动图
        print("complete")
    return 


#--------------------------下为为一般拓扑编写部分-------------------------
'''
src数组存放发送信号的节点的名称,graphs数组存放以各个节点为根生成的树状图
,des数组存放接受端的节点的名称,默认路由选路都使用nx.shortest_path的算法
'''

def generate_random_src_and_des(graph,num,max_node_num):
    #传入初始的图像以生成算法所需的src和des数组,num为需要发信节点的数量
    #max_node_num为每个src对应的最大的目的地数量

    if num>len(list(graph.nodes)):
        #防止src数量超过总的节点数量
        num=len(list(graph.nodes))
    
    #产生num个随机数作为发送节点的序号
    src_index = random.sample(range(len(list(graph.nodes))), num)     
    src=[]
    for i in src_index:
        src.append(list(graph.nodes)[i])    #生成src数组
    
    des=[]      #生成des数组
    for node in src:
        node_index = next((i for i, x in enumerate(list(graph.nodes)) if x == node), None)
        top=[]
        time=0
        while time<=max_node_num:
            time+=1
            target=random.randint(0,len(list(graph.nodes))-1)
            target_node=list(graph.nodes)[target]
            if target==node_index or  target_node in top:
                #避免选的目标点是发送点或者已经存在
                continue
            #找到路径，确保为端到端测量
            path=nx.shortest_path(graph, source=node, target=target_node)
            top=[x for x in top if x not in path]
            top.append(target_node)
        for destination in top:
            path=nx.shortest_path(graph,source=node, target=destination)
            path=path[1:-1]
            top=[x for x in top if x not in path]
        des.append(top)
    
    '''
    graphs=[]       #构建graphs函数，包含各个src节点为开始的树状图
    for i in range(len(src)):
        node=src[i]
        G=nx.DiGraph()
        for target_node in des[i]:
            path=nx.shortest_path(graph, source=node, target=target_node)
            for n in path:
                G.add_node(n)
            for j in range(len(path)-1):
                G.add_edge(path[j],path[j+1])
        graphs.append(G)'''
    return src,des

def generate_Z_on_general_topology(graph):
    #随机产生Z数组，和树形拓扑不太一样，树形拓扑的Z是跟着点产生的，一般拓扑得跟着边产生
    #那么就是说在一般拓扑里，就算是发信点对应的第一条边也可能会故障
    Z=[]
    links=list(graph.edges)
    for i in range(len(links)):
        Z.append(random.randint(0,1))
    return Z


#更新每一个小树的x数组z,组成集合为X
def update_X(ori_graph,Z,graphs,src):
    #ori_graph就是原来的图，Z即Z数组，graphs就是每一个小型的树状图
    X=[]
    j=0
    for graph in graphs:
        top=[]
        z=[]    #创建每一个小树的z
        links=list(graph.edges)
        for link in links:
            #找到新的link在原图里对应的序号
            link_index=next((i for i,x in enumerate(list(ori_graph.edges())) if x==link),None)
            z.append(Z[link_index])
        leaves = [node for node in list(graph.nodes) if graph.out_degree(node) == 0]
        x = np.full(len(list(graph.nodes)), -1)     #创建一个全为-1的1维数组为x
        for leaf in leaves:
            paths = nx.shortest_path(graph, source=src[j], target=leaf)
            tmp = 1
            for i in range(len(paths)-1):
                path=(paths[i],paths[i+1])
                path_index=next((i for i,x in enumerate(list(graph.edges)) if x==path),None)
                tmp*=z[path_index]
            leaf_index = next((i for i, a in enumerate(list(graph.nodes)) if a == leaf), None)
            x[leaf_index] = tmp
        X.append(x)
        j+=1
    return X

def generate_graphs(graph,src,des):
    #根据图和src和des数组生成graphs数组
    graphs=[]       #构建graphs函数，包含各个src节点为开始的树状图
    for i in range(len(src)):
        node=src[i]
        G=nx.DiGraph()
        paths=[]
        for target_node in des[i]:
            path=nx.shortest_path(graph, source=node, target=target_node)
            paths.append(path)
        for j in range(len(paths)):
            for k in range(0,j):
                length=min(len(paths[j]),len(paths[k]))
                for l in range(length):
                    if paths[k][l]==paths[j][l]:
                        paths[j][1:l]=paths[k][1:l]         #确保到同一个节点的路径是一样的，不然可能每个分别的检测不是树状的
        for path in paths:
            for n in path:
                G.add_node(n)
            for j in range(len(path)-1):
                G.add_edge(path[j],path[j+1])
        graphs.append(G)
    return graphs
    
def SCFS_on_general_topology(graphs,X,src):
    W=[]
    #print(src)
    for i in range(len(src)):
        tmp=[]
        for root in src[i]:
            tmp1=SCFS_on_tree(graphs[i],X[i],root)     #对每个树单独调用标准的SCFS算法
            tmp.extend(tmp1)    #每棵树这一轮的结果保留在tmp里
        W.append(tmp)
    return W

def update_Z_and_get_data(graph,graphs,Z,W):

    inspected=[]    #记录检查过的边
    #originally_good记录本轮检测前就为good的链路的数量,bad_link记录本轮检测的链路里有多少条是bad的,total_bad记录在本轮的覆盖范围内有多少的链路是bad的
    originally_good=0
    bad_link=0
    total_bad=0
    total_link=[]   #记录在graphs即检测范围内的所有link
    for i in range(len(graphs)):
        links=list(graphs[i].edges)
        links=[link for link in links if link not in total_link]
        total_link.extend(links)
    for link in total_link:
        link_index= next((i for i,x in enumerate(list(graph.edges)) if x==link),None)   #找到边在Z里的序号
        if Z[link_index]==0:
            total_bad+=1    #找到所有检测范围内bad的边的数量
        
    for i in range(len(W)):
        links=list(graphs[i].edges)
        for node in W[i]:
            link=next((x for i,x in enumerate(links) if x[1]==node),None)   #找到边
            inspected.append(link)
            link_index= next((i for i,x in enumerate(list(graph.edges)) if x==link),None)    #找到边在Z里的序号
            if Z[link_index]==1:
                #链路本来就是好的
                originally_good+=1
            else:
                #链路本来就是坏的
                bad_link+=1
                Z[link_index]=1
    if len(inspected)!=0:
        FPR=originally_good/len(inspected)
        detection_rate=bad_link/total_bad
    else:
        FPR=0
        detection_rate=1

    return Z,inspected,FPR,detection_rate

def plot_topology(graph,Z,round_num,dir):
    #根据Z画图，参考另一边的visualize_tree函数
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')  # 使用Graphviz的dot排列节点
    nx.draw(graph, pos, with_labels=True, node_size=500, font_size=8, font_weight='bold', edge_color='gray', node_color='green')

    error_node=[]
    for i in range(len(Z)):
        if Z[i]==0:
            link=list(graph.edges)[i]
            node=link[1]
            if node not in error_node:
                error_node.append(node)
    nx.draw_networkx_nodes(graph, pos, nodelist = error_node, node_color='red', node_size=500)
    plt.savefig(f'{dir}/round{round_num}.png')


def simulation_on_general_topology(graph,num,max_node_num,Z=[],round_num=-1,graphs=[],src=[],des=[]):
    #如果是第一次进入模拟，Z应该传入空，如果不是，那就传入上一次的Z
    if round_num==-1:       #看是不是第一次模拟
        dir='./scfs'
        if not os.path.exists(dir):    #运行之前检测一下路径存不存在，不存在就创建
            os.makedirs('./scfs')
        if not os.path.isfile(os.path.join(dir, 'counter.txt')):
            with open(os.path.join(dir, 'counter.txt'), 'w') as f:  #记录counter文件保存次数
                f.write('1')
        with open(os.path.join(dir, 'counter.txt'), 'r') as f:      #存在就直接读文件
            word = ""
            while True:
                char=f.read(1)
                if char==" "or char=="\n"or char=='':
                    break
                word+=char
            c=int(word)
        with open(os.path.join(dir, 'counter.txt'), 'w') as f:
            f.write(str(c + 1))
        newdir = os.path.join(dir, str(c))      #创建第n次模拟的路径和文件夹
        os.mkdir(newdir)
    else:
        dir="./scfs"
        with open(os.path.join(dir, 'counter.txt'), 'r') as f:  
            word = ""
            while True:
                char=f.read(1)
                if char==" "or char=="\n"or char=='':
                    break
                word+=char
            c=int(word)-1
        newdir = os.path.join(dir, str(c))
        
    if Z==[]:   #如果是第一次模拟才生成Z
        Z=generate_Z_on_general_topology(graph)
    recurse=0   #判断是否是在随机循环模拟，是就是1，不是就是0
    if graphs==[]:         #随机循环模拟，graphs应该传入[]
        src,des=generate_random_src_and_des(graph,num,max_node_num)
        graphs=generate_graphs(graph,src,des)
        recurse=1

    print("-------------------------------------------------------")
    print("src:",src)
    print("des:",des)
    print("fail 的链路:",end='')
    failed_links=[]
    for i in range(len(Z)):
        if Z[i]==0:
            failed_links.append(list(graph.edges)[i])
    print(failed_links)

    W=[]
    for i in src:
        W.append([i])
    while 1:
        end=1
        for nodes in W: #查看W内元素是否全为空列表
            if nodes:
                end=0
        if end==1:
            break
        round_num+=1
        X=update_X(graph,Z,graphs,src)
        W=SCFS_on_general_topology(graphs,X,W)
        plot_topology(graph,Z,round_num,newdir)
        Z,inspected,FPR,detection_rate=update_Z_and_get_data(graph,graphs,Z,W)
        if inspected!=[]:
            print("本轮检查的链路：",end='')
            print(inspected)
            print("本轮的FPR:",FPR,"本轮的DR:",detection_rate)
    remain,error=check(graph,graphs,Z)
    if error:
        print("----------算法出错.请检查数据-----------")
        exit(-1)
    if remain and recurse:  #还有链路完全没被检测，因为没经过
        simulation_on_general_topology(graph,num+1,max_node_num+1,Z,round_num,[],[],[])        #再次运行算法，同时增加src的数量和最大des的数量
    else:
        create_gif_from_images(newdir, 'scfs.gif')

def check(graph,graphs,Z):
    #检查算法是否正确
    remain=[]
    failed_to_check=[]
    for i in range(len(Z)):
        if Z[i]==0:
            failed_link=list(graph.edges)[i]
            remain.append(failed_link)
            for G in graphs:
                if failed_link in list(G.edges):
                    failed_to_check.append(failed_link)
    print("未被修复的链路:",remain)
    if failed_to_check:
        print("应该被修复但没有被检测到的链路:",failed_to_check)
        return remain,1 #如果发现算法出错就返回第二位为1
    return remain,0

def get_general_topology_graphs(route_matrix,graph,links):
    #根据一般拓扑的图和路由矩阵找出src数组，des数组，和graphs数组
    src=[]
    des=[]
    for i in range(len(route_matrix)):
        #对每一个paths
        paths=[]
        for j in range(len(route_matrix[i])):
            if route_matrix[i][j]==1:
                paths.append(links[j])      #组成paths
            start_node,end_node=get_start_and_end(paths)
            if start_node in src:
                #如果开始节点已在src中
                src_index=src.index(start_node)
                des[src].append(end_node)
            else:
                #开始节点未在src中
                src.append(start_node)
                des.append([end_node])
    graphs=generate_graphs(graph,src,des)
    return src,des,graphs       #返回生成好的src，des和graphs数组

def get_start_and_end(paths):
    #paths为一个乱序的路径数组如[('4','6'),('2','4'),('1','2')]
    in_degree = {}
    out_degree = {}
    for path in paths:
        in_degree[path[0]] = in_degree.get(path[0], 0) + 0
        out_degree[path[0]] = out_degree.get(path[0], 0) + 1
        in_degree[path[1]] = in_degree.get(path[1], 0) + 1
        out_degree[path[1]] = out_degree.get(path[1], 0) + 0
    start = [node for node, degree in in_degree.items() if degree == 0]
    end = [node for node, degree in out_degree.items() if degree == 0]
    if len(start)!=1 or len(end)!=1:
        print("出错，路径结尾或开头不唯一")
        exit(-1)
    return start[0],end[0]      #返回'1'和'6'

#------------------------------------------------------------




def parsers(route_matrix,links=[],root=None):
    #route_matrix即路由矩阵
    if links==[]:
        #树状
        graph=nx.DiGraph()
        for i in range(len(route_matrix)):
            from_node=root
            for j in range(len(route_matrix[i])):
                if route_matrix[i][j]==1:
                    graph.add_edge(from_node,str(j))
                    from_node=str(j)
    else:
        #一般
        graph=nx.DiGraph()
        #网状无法直接通过路由矩阵添加，而是通过links数组包含所有的边的信息添加
        graph.add_edges_from(links)
    return graph,route_matrix

def start_simulation(route_matrix,links=[]):
    
    if links==[]:
        #树状
        graph,route_matrix=parsers(route_matrix,links,root='0')
        simulation(graph,'0',True)
    else:
        #一般
        graph,route_matrix=parsers(route_matrix,links)
        src,des,graphs=get_general_topology_graphs(route_matrix,graph,links)
        simulation_on_general_topology(graph,10,5,[],-1,graphs,src,des)     #src,graph,des内有内容代表不是循环模拟
    
    
def generate_tree(graph, parent, depth, max_depth, branch_ratio):
    if depth >= max_depth:
        return
        
    for _ in range(branch_ratio):
        child = graph.number_of_nodes()  # 使用当前节点数作为新节点的编号
        graph.add_node(child, Internal=1)
        graph.add_edge(parent, child)
        generate_tree(graph, child, depth + 1, max_depth, branch_ratio)
          
          
def generate_gml(max_depth, branch_ratio):
    # 创建一个空的无向图
    G = nx.DiGraph() 

    # 添加根节点
    root = 0
    G.add_node(root, Internal=1)

    child = G.number_of_nodes() 
    G.add_node(child, Internal=1)
    G.add_edge(0, child)

    # 生成树状结构

    generate_tree(G, 1, 1, max_depth, branch_ratio)
    # 保存为GML文件
    nx.write_gml(G, "tree_topology.gml")

    # 可视化网络拓扑
    #pos = nx.spring_layout(G)
    #nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10)
    #plt.show()
def get_data(Data_, FPR, DR, alp):
    Data_[alp] = {'FPR': FPR, 'DR': DR}
    
def get_averages(Data):

    # print("data", Data)
    # 数据转换，列表转字典
    Data_ = []
    for item in Data:
        key = list(item.keys())[0]
        value = item[key]
        Data_.append((key, (value['FPR'], value['DR'])))
    # print('last', Data_)
    averaged_data = defaultdict(lambda: {'FPR': 0, 'DR': 0, 'count': 0})

    # 遍历 Data_，累加具有相同键的值
    for item in Data_:
        # print(item)
        averaged_data[item[0]]['FPR'] += item[1][0]
        averaged_data[item[0]]['DR'] += item[1][1]
        averaged_data[item[0]]['count'] += 1
    
    # debug
    # for key, values in averaged_data.items():
    #     print(key, values, values['DR']/values['count'])
    
    # 计算平均值并创建新的字典
    result = {key: {'FPR': value['FPR'] / value['count'],
                    'DR': value['DR'] / value['count']}
            for key, value in averaged_data.items()}
    result[1] = {'FPR': 0, 'DR': 1}
    # print('res', result)
    return result


def averages_internal(Data_):
    averages = {}  # 用于存储平均值的字典

    for i in range(10):
        alp_start = i / 10  # 计算起始值，如：0.0, 0.1, 0.2, ...
        alp_end = (i + 1) / 10  # 计算结束值，如：0.1, 0.2, 0.3, ...
        
        fpr_sum = 0
        dr_sum = 0
        count = 0
        
        for alp, values in Data_.items():
            if alp_start <= alp < alp_end:
                fpr_sum += values['FPR']
                dr_sum += values['DR']
                count += 1
        
        if count > 0:
            avg_fpr = fpr_sum / count
            avg_dr = dr_sum / count
            averages[alp_start] = {'FPR': avg_fpr, 'DR': avg_dr}
    # for alp, values in averages.items():
    #     print(f'Average FPR for alp={alp:.1f}: {values["FPR"]:.2f}')
    #     print(f'Average DR for alp={alp:.1f}: {values["DR"]:.2f}')
    averages[1] = {'FPR': 0, 'DR': 1}
    return averages

    
def draw_pic_fig5(sum_data,type='b'):
    
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)
    parsers = ['bD--', 'r^--', 'go--', 'c*--', 'm+--', 'y>--', 'k<--', 'w>--', 'b>--', 'r>--']
    if type == 'b':
        labels = [r'$T_{\alpha}(2,2)$', r'$T_{\alpha}(2,2,2)$', r'$T_{\alpha}(2,2,2,2)$', r'$T_{\alpha}(2,2,2,2,2)$']
    elif type == 'c':
        labels = [r'$T_{{\alpha}^{\frac{1}{2}}} (2,2)$', r'$T_{{\alpha}^{\frac{1}{3}}} (2,2,2)$', r'$T_{{\alpha}^{\frac{1}{4}}} (2,2,2,2)$', r'$T_{{\alpha}^{\frac{1}{5}}} (2,2,2,2,2)$']
    elif type == 'a':
        labels = [r'$T_{\alpha}(2,2)$', r'$T_{\alpha}(3,3)$', r'$T_{\alpha}(10,10)$']
    for i in range(len(sum_data)):
        averages = get_averages(sum_data[i])
        # ave = averages_internal(averages)
        
        # 模式一 （用采得的 所有结果 进行排序 画图）
        alp_values = list(averages.keys())
        dr_values = [data['DR'] for data in averages.values()]
        # 按 alp_values 从小到大的顺序重新排列 alp_values 和 dr_values
        sorted_data = sorted(zip(alp_values, dr_values))
        sorted_alp_values, sorted_dr_values = zip(*sorted_data)
        
        
        # print(sorted_alp_values, sorted_dr_values)
        ax.plot(sorted_alp_values, sorted_dr_values, parsers[i], label=labels[i]) 
        
        # # 模式二 （ 采用 0-1 分10 个间隔）
        # alp_values = list(ave.keys())
        # dr_values = [data['DR'] for data in ave.values()]
        # ax.plot(alp_values, dr_values, parsers[i], label=labels[i]) 
        
        
        
    
    ax.set_xlabel('alp')
    ax.set_ylabel('dr')
    ax.set_title(f'Fig.5({type})')
    
    ax.set_ylim(0, 1)
    
    ax.set_xticks([i / 10 for i in range(11)])
    ax.set_xlim(0, 1)
    
    ax.legend()
   
    plt.show()


# 模拟一般拓扑
def simulator_general():
    
    
    net = 网络基类()
    net.配置拓扑("./topology_zoo数据集/Uninett2011.gml")
    simulation_on_general_topology(net.图,num=10,max_node_num=5,Z=[],round_num=-1,graphs=[],src=[],des=[])    #num和max_node_num可以改
    #print(Data_)
    #draw_pic_fig5([Data_])
    
def fig5_b():
    sum_data = []

    for i in tqdm(range(3,7)):
        # print("--------------------")
        container = []
        generate_gml(i, 2)
        net = 网络基类()
        net.配置拓扑("./tree_topology.gml")
        root = '0'     #需设置根节点
        #对于每一个拓扑，进行模拟
        alpha=0
        while alpha <= 1.0:
            for j in range(400):
                Data_={}
                simulation(net.图, root, Data_, output=False, proportion=alpha)
                if Data_:
                    key, value = list(Data_.items())[0]
                    container.append({key: value})
                    
            alpha+=0.1
            
        sum_data.append(container)

    draw_pic_fig5(sum_data,type='b')
    
def fig5_a():
    sum_data = []

    new_list = [2, 3, 10]

    for i in range(3):
        # print("--------------------")
        container = []
        generate_gml(3, new_list[i])
        net = 网络基类()
        net.配置拓扑("./tree_topology.gml")
        root = '0'     #需设置根节点

        #对于每一个拓扑，进行模拟
        alpha=0
        
        for k in tqdm(range(10), desc='Processing'):
            for j in range(300):
                Data_={}
                simulation(net.图, root, Data_, output=False, proportion=alpha)
                if Data_:
                    key, value = list(Data_.items())[0]
                    container.append({key: value})
                    
            alpha+=0.1
            
        sum_data.append(container)
        
    draw_pic_fig5(sum_data,type='a')
    
def fig5_c():
    sum_data = []

    for i in tqdm(range(3,7)):
        # print("--------------------")
        container = []
        generate_gml(i, 2)
        net = 网络基类()
        net.配置拓扑("./tree_topology.gml")
        root = '0'     #需设置根节点

        #对于每一个拓扑，进行模拟
        alpha=0
        while alpha <= 1.0:
            for j in range(1000):
                Data_={}
                simulation(net.图, root, Data_, output=False, proportion= alpha ** (1 / (i - 1)))
                if Data_:
                    key, value = list(Data_.items())[0]
                    container.append({alpha: value})
                    
            alpha+=0.1
            
        sum_data.append(container)

    draw_pic_fig5(sum_data,type='c')

if __name__ == '__main__':
    net = 网络基类()

    #net.配置拓扑("./topology_zoo数据集/Chinanet.gml")
    #simulation_on_general_topology(net.图,num=10,max_node_num=5,Z=[],round_num=-1,graphs=[],src=[],des=[])
    fig5_b()