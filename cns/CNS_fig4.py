import mindspore
from matplotlib import pyplot as plt

print(mindspore.__version__)
from mindspore.common.tensor import Tensor
import mindspore.numpy as np
import mindspore.ops as ops
from utils import hessian_Function, jacobian_Function, Fitness_Function, cons1, cons2, cons3, cons4, std, mean, Equation
import time
iii = 0
save_value = {}
tesnum = [0]
resss = []
resss.append(100000)
time_interval = np.zeros((1,100))

equation = Equation.engineering
#'sixhump','branin','griewank','sphere','fir','engineering'

if equation== Equation.sixhump:        optimal = -1.0316
elif equation== Equation.branin:       optimal = 0.397887
elif equation== Equation.griewank:     optimal = 0
elif equation== Equation.sphere:       optimal = 0
elif equation==Equation.engineering:   optimal = 7000
N = 300 #群体例子个数
D = 2 #粒子维度
T = 30 #最大迭代次数
c1 = 2.05 #学习因子1
c2 = 2.05 #学习因子2
# c1 = 1.45; #学习因子1
# c2 = 1.45; #学习因子2
Wmax = 1.2 #惯性权重最大值
Wmin = 0.5 #惯性权重最小值
# Xmax = 6; #位置最大值
# Xmin = -6; #位置最小值
# Vmax = 1; #速度最大值
# Vmin = -1; #速度最小值

# equation = 'griewank'; #'sixhump', 'branin', 'griewank'
if equation==Equation.sixhump:
    Xmax = 3 #位置最大值
    Xmin = -3 #位置最小值
    Vmax = 2 #速度最大值
    Vmin = -2 #速度最小值
    exceptValue = -0.3 #避免陷入局部最优的限制值
    optimal = -1.0316
elif equation==Equation.branin:
    Xmax = 10 #位置最大值
    Xmin = -5 #位置最小值
    Vmax = 15 #速度最大值
    Vmin = 0 #速度最小值
    exceptValue = 2 #避免陷入局部最优的限制值
    optimal = 0.397887
elif equation==Equation.griewank:
    Xmax = 10 #位置最大值
    Xmin = -10 #位置最小值
    Vmax = 10 #速度最大值
    Vmin = -10 #速度最小值
    exceptValue = 1 #避免陷入局部最优的限制值
    optimal = 0
elif equation==Equation.sphere:
    Xmax = 10 #位置最大值
    Xmin = -10 #位置最小值
    Vmax = 10 #速度最大值
    Vmin = -10 #速度最小值
    exceptValue = 1 #避免陷入局部最优的限制值
    optimal = 0
elif equation==Equation.engineering:
    D = 4 #粒子维度
    # Xmax = 2; #位置最大值
    Xmax = 6.1875 #位置最大值
    Xmin = 0.0625 #位置最小值
    Vmax = 200 #速度最大值
    Vmin = 10 #速度最小值
    exceptValue = 8000 #避免陷入局部最优的限制值
while iii<1:
    best_index = []
    isplot = 1
    x =np.rand(( D, N)) * (Xmax - Xmin) + Xmin
    #初始化种群个体（限定位置和速度）

    v = np.rand((D, N)) * (Vmax - Vmin) + Vmin
    if equation==Equation.engineering:
        x1 = np.round((np.rand((1, N)) * (2 - 1.1) + 3.1) / 0.0625) * 0.0625
        x2 = np.round((np.rand((1, N)) * (2 - 0.6) + 2.6) / 0.0625) * 0.0625
        #% x1 = (rand(1, N) * (2 - 1.1) + 3.1);
        #% x2 = (rand(1, N) * (2 - 0.6) + 3.6);
        x3 = np.rand((1, N)) * 10 + 100
        x4 = np.rand((1, N) )* 40 + 150
        x = np.concatenate([x1,x2,x3,x4],axis=0)

    noisetype = 0  #  0: 'none', 1: 'constant', 2: 'random'
    if noisetype==2:
        randomnoiseRange = 2
        if equation==Equation.sixhump:    beta = 0.5;threshold = 20#10    for constant noise
        elif  equation==Equation.branin:  beta = 0.1;threshold = 20
        elif equation==Equation.engineering:beta = 0.05;threshold = 2000
    elif noisetype==1:
        noise = 3 * np.ones((D, N)) #常量噪声(D,T)
        if equation==Equation.sixhump:
                beta = 0.3 #beta = 0.3 for sixhump
                threshold = 30  #10 for constant noise
        elif equation== Equation.branin:
                beta = 0.1
                threshold = 20
        elif equation== Equation.engineering:
                beta = 0.05
                threshold = 2000
                noise = 3 * np.ones((D, T))
    elif noisetype==0:
        beta = 0.0001
        noise = np.zeros((D, N))#T->N
        threshold = 2000
        if equation==Equation.sixhump:
            beta = 0.0001 #branin_without
            threshold = 20
        elif equation==Equation.branin:
            beta = 0.0001
            threshold = 20
        elif equation==Equation.sphere:
            beta = 0.0001
            threshold = 10
    p = x
    pbest = np.ones((1, N))
    for i in range(0,N):
        pbest[0][i] = Fitness_Function(x[:, i], equation)
    #%%%%%%%%%%初始化全局最优位置和最优值 %%%%%%%%%
    g = np.ones((1, D))
    gbest = np.inf
    for i in  range(0,N):
        if pbest[0][i] < gbest:
            g = p[:, i]
            gbest = pbest[0][i]
    before = gbest
    before_g = g
    xcns = np.zeros((D, N))
    gbest_index = g
    gb = np.ones((1, T))
    jacobiRes = 0
    error_sum = np.zeros((D, N))
    hessianRes = np.zeros((D, D, N))
    start_time=time.time()
    #%%%%%%%%%%按照公式依次迭代直到满足精度或者迭代次数 %%%%%%%%%
    for i in range(0,1):
        for j in range(0,N):
            xx=x[:, j]
            jacobiRes = jacobian_Function(xx, equation)
            hessianRes = hessian_Function(xx, equation)
            error_sum[:, j] = jacobiRes
            #%%%%%%%%%%牛顿法更新位移 %%%%%%%%%
            #求逆
            xcns[:, j] = xx - (hessianRes.inverse()) @ (jacobiRes + beta * error_sum[:, j])
            if equation==Equation.engineering:
                xcns[0, j] = round(xcns[0, j] / 0.0625) * 0.0625
                xcns[1, j] = round(xcns[1, j] / 0.0625) * 0.0625
                if xcns[0, j]<0.6 or xcns[0, j] > 2:
                    xcns[0, j] = round((np.rand()[0] * (2 - 0.6) + 0.6) / 0.0625) * 0.0625
                if xcns[1, j] < 0.6 or xcns[1, j] > 2:
                    xcns[1, j] = round((np.rand()[0]* (2 - 0.6) + 0.6) / 0.0625) * 0.0625
                if xcns[2, j] < 10:
                    xcns[2, j] = 10
                if xcns[3, j] < 50 or xcns[3, j] > 240:
                    xcns[3, j] = np.rand()[0] * (240 - 50) + 50
            if equation==Equation.engineering:
                if cons1(xcns[:,j]) == 1:
                    if cons2(xcns[:,j]) == 1:
                        if cons3(xcns[:,j]) == 1:
                            if cons4(xcns[:,j]) == 1:
                                fitnessRes = Fitness_Function(xcns[:,j], equation)
                                pass
                            else:
                                fitnessRes = 10 ** 10
                        else:
                            fitnessRes = 10 ** 10
                    else:
                        fitnessRes = 10 ** 10
                else:
                    fitnessRes = 10 ** 10
            else:
                #%%%%%%%%%%更新个体最优位置和最优值 %%%%%%%%%
                fitnessRes = Fitness_Function(xcns[:, j], equation)

            if fitnessRes < pbest[0][j]:
                p[:, j] = xcns[:, j]
                pbest[0][j] = fitnessRes
                #print(fitnessRes,j)
#            %%%%%%%%%%更新全局最优位置和最优值 %%%%%%%%%
            if pbest[0][j] < gbest:
                g = p[:, j]
                gbest = pbest[0][j]
            #%%%%%%%%%%动态计算惯性权重值 %%%%%%%%%
            w = Wmax - (Wmax - Wmin) * (i+1) / T
            #%%%%%%%%%%更新位置和速度值 %%%%%%%%%
            s=time.time()
            x[:, j] = x[:, j] + w * (xcns[:, j] - x[:, j]) + c1 * np.rand()[0] * (p[:, j] - x[:, j]) + c2 * np.rand()[0] * (g - x[:, j])
            e=time.time()
            for ii in range(D):
                if (x[ii, j] > Xmax) or (x[ii, j] < Xmin):
                    x[ii, j] = np.rand()[0] * (Xmax - Xmin) + Xmin
        gb[0][i] = gbest
        #gbest_index[:, i] = g
        gbest_index = np.array([gbest_index, g])

    for i in range(1,T):
        start_time=time.time()
        for j in range(N):

            #%%%%%%%%%%牛顿法更新位移 %%%%%%%%%
            jacobiRes = jacobian_Function(x[:, j], equation)
            hessianRes = hessian_Function(x[:, j], equation)
            error_sum[:, j] = error_sum[:, j] + jacobiRes
            if noisetype == 2:
                xcns[:, j] = x[:, j] - (hessianRes.inverse()) @(jacobiRes + beta * error_sum[:, j] + (3 + randomnoiseRange * np.rand((D,))))
            else:
                xcns[:, j] = x[:, j] - (hessianRes.inverse()) @(jacobiRes + beta * error_sum[:, j] + noise[:, j])

            if equation==Equation.engineering:
                xcns[0, j] = round(xcns[0, j] / 0.0625) * 0.0625
                xcns[1, j] = round(xcns[1, j] / 0.0625) * 0.0625
                if xcns[0, j]<0.6 or xcns[0, j] > 2:
                    xcns[0, j] = round((np.rand()[0] * (2 - 0.6) + 0.6) / 0.0625) * 0.0625
                if xcns[1, j] < 0.6 or xcns[1, j] > 2:
                    xcns[1, j] = round((np.rand()[0] * (2 - 0.6) + 0.6) / 0.0625) * 0.0625
                if xcns[2, j] < 10:
                    xcns[2, j] = 10
                if xcns[3, j] < 50 or xcns[3, j] > 240:
                    xcns[3, j] = np.rand()[0] * (240 - 50) + 50
            if equation==Equation.engineering:
                if cons1(xcns[:,j]) == 1:
                    if cons2(xcns[:,j]) == 1:
                        if cons3(xcns[:,j]) == 1:
                            if cons4(xcns[:,j]) == 1:
                                fitnessRes = Fitness_Function(xcns[:,j], equation)
                            else:
                                fitnessRes = 10 ** 10
                        else:
                            fitnessRes = 10 ** 10
                    else:
                        fitnessRes = 10 ** 10
                else:
                    fitnessRes = 10 ** 10
            else:
                #%%%%%%%%%%更新个体最优位置和最优值 %%%%%%%%%
                fitnessRes = Fitness_Function(xcns[:, j], equation)
            #%%%%%%%%%%更新全局最优位置和最优值 %%%%%%%%%
            if fitnessRes < pbest[0][j]:
                p[:, j]=xcns[:,j]
                pbest[0][j]=fitnessRes
            if pbest[0][j] < gbest:
                g = p[:, j]
                gbest = pbest[0][j]
            #%%%%%%%%%%动态计算惯性权重值 %%%%%%%%%
            w = Wmax - (Wmax - Wmin) * (i+1) / T
            #%%%%%%%%%%更新位置和速度值 %%%%%%%%%
            #s=time.time()
            x_=x[:,j]
            xc=xcns[:, j]
            x[:, j] = x_ + w * (xc - x_) + c1 * np.rand()[0]* (p[:, j] - x_) + c2 * np.rand()[0] * (g - x_)
            #e=time.time()
            #print("{:.2f}".format(e - s))
            if equation== Equation.engineering:
                x[0, j] = round(x[0, j] / 0.0625) * 0.0625
                x[1, j] = round(x[1, j] / 0.0625) * 0.0625
                if x[0, j] < 0.6 or x[0, j] > 2:
                    #% x(1, j) = round((rand * (2 - 1.1) + 1.1) / 0.0625) * 0.0625;
                    x[0, j] = (np.rand()[0] * (2 - 0.6) + 0.6)

                if x[1,j] < 0.6 or x[1,j] > 2:
                    #% x[1,j] = round((rand * (2 - 0.6) + 0.6) / 0.0625) * 0.0625;
                    x[1,j] = (np.rand()[0] * (2 - 0.6) + 0.6)
                if (x[2,j] < 10):
                    x[2,j] = 10
                if (x[3,j] < 50 or x[3,j] > 240):
                    x[3,j] = np.rand()[0]* (240 - 50) + 50
            else:
                for ii in range(0,D):
                    if (v[ii,j] > Vmax) or (v[ii,j] < Vmin):
                        v[ii,j] = np.rand()[0] * (Vmax - Vmin) + Vmin
                    if (x[ii,j] > Xmax) or (x[ii,j] < Xmin):
                        x[ii,j] = np.rand()[0]* (Xmax - Xmin) + Xmin

        #%%%%%%%%%%记录历代全局最优值 %%%%%%%%%
        gb[0][i] = gbest
        gbest_index = [gbest_index, g]
        end_time=time.time()
        print("{:.2f}".format(end_time - start_time))
        print(gb[0].numpy())
    print(['最优个体：',g])
    print(['最优值：' ,gb[0][-1]])

    iii = iii + 1

    save_value[str(iii)] = gb[0][-1]
    resss.append(gb[0][-1])
    if abs(gb[0][-1] - optimal) < 1e-4:
        tesnum[0] = tesnum[0] + 1
        isplot = 1

    if (isplot == 1):
        isplot = 0
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值')
        plt.title('适应度进化曲线')
        plt.plot(gb[0].numpy())
        plt.show()
        print(gb[0].numpy())




print(['CNS最优值：', min(save_value.values())])
print(['CNS标准差：', std(save_value.values())])
print(['CNS均值：' ,mean(save_value.values())])
print(['beta,tesnum,iii：',beta,tesnum, iii])






