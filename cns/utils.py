from enum import Enum
from mindspore import jacrev, jacfwd, vmap, vjp, jvp, grad
import mindspore.numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops
ms.context.set_context(device_target='CPU')
class Equation(Enum):
    sixhump=1
    branin=2
    griewank=3
    sphere=4
    fir=5
    engineering=6
def Fitness_Function(x:Tensor, equation=Equation.sixhump):
    if equation==Equation.sixhump:
        F = x[0].pow(2) * (x[0].pow(4) / 3 - (21 * x[0].pow(2)) / 10 + 4) + x[0] * x[1] + x[1].pow(2) * (
                    4 * x[1].pow(2) - 4)
        return F
    elif equation == equation.branin:
        F = 9.602112642270262 * np.cos(x[0]) + (- 0.1291845091439807 * x[0].pow(2) + 1.591549430918954 * x[0] + x[1] - 6).pow(2) + 10
        return F
    elif equation == equation.griewank:
        F = x[0].pow(2) / 4000 - np.cos((2 ** (1 / 2) * x[1]) / 2) * np.cos(x[0]) + x[1].pow(2) / 4000 + 1
        return F
    elif equation == equation.engineering:
        F = 0.6224*x[0]*x[2]*x[3]+1.7781*x[1]*(x[2].pow(2))+3.1661*(x[0].pow(2))*x[3]+19.84*(x[0].pow(2))*x[2]
        #if np.isclose(F,0.):
            #print(F,x)
        return F
    elif equation == equation.sphere:
        F = sum(x.pow(2))
        return F
    elif equation == equation.fir:
        (row,col)=x.shape
        N=2*col-1#阶数,奇数阶
        t=(N-1)*0.5
        M=64 #划分区间
        y=0
        w=[0, np.pi, np.pi / 63]
        # Hd=[msnp.zeros((1,31)),0.3968,0.7937,msnp.ones((1,31))];
        Hd= np.zeros((1, 64))#高通
        for i in range(t+1):
            x[ t + i] = x[t - i]
        H=np.zeros(M)
        e=np.zeros(M)
        for i in range(0,M):
            sum1 = 0
            for j in range(0,t+1):
                Bb = 2 * x[t - j] * np.cos((j) * w[i])
                sum1 = sum1 + Bb
            sum1 = sum1 - x[1+t] * np.cos(0 * w[i]) #使用2 * h(n)
            H[i] = sum1
            e[i] = (abs(H[i]) - abs(Hd[i])) ** 2
            y = e[i] + y

        return y

def hessian_Function(x, equation):
    if equation == equation.sixhump:
        H = [
            [x[0].pow(2)*(4.*x[0].pow(2) - 21/5) - 4*x[0]*((21*x[0])/5 - (4*x[0].pow(3))/3) - (21*x[0].pow(2))/5 + (2*x[0].pow(4))/3 + 8,1],
            [1, 48*x[1].pow(2) - 8]
        ]
        return np.array(H)
    elif equation == equation.branin:
        H = [
            [2.0 * (0.2583690182879614 * x[0] - 1.591549430918954) ** 2 - 0.5167380365759228 * x[1] - 9.602112642270262 * np.cos(x[0]) - 0.8224141280465875 * x[0] + 0.06675454961108492 * x[0] ** 2 + 3.100428219455537, 3.183098861837908 - 0.5167380365759228 * x[0]],
            [3.183098861837908 - 0.5167380365759228 * x[0], 2.0]]
        return np.array(H)
    elif equation == equation.griewank:
        H = [[np.cos((2 ** (1 / 2) * x[1]) / 2) * np.cos(x[0]) + 1 / 2000, -(2 ** (1 / 2) * np.sin((2 ** (1 / 2) * x[1]) / 2) * np.sin(x[0])) / 2],
             [-(2 ** (1 / 2) * np.sin((2 ** (1 / 2) * x[1]) / 2) * np.sin(x[0])) / 2, (np.cos((2 ** (1 / 2) * x[1]) / 2) * np.cos(x[0])) / 2 + 1 / 2000]]
        return np.array(H)
    elif equation == equation.sphere:
        H = [[2,0],
            [0,2]]
        return np.array(H)
    elif equation == equation.engineering:
        H = np.array([[992*x[2]/25 + 31661*x[3]/5000, 0.0, 992*x[0]/25 + 389*x[3]/625, 31661*x[0]/5000 + 389*x[2]/625],
              [0.0, 0.0, 17781*x[2]/5000, 0.0],
              [992*x[0]/25 + 389*x[3]/625, 17781*x[2]/5000, 17781*x[1]/5000, 389*x[0]/625],
              [31661*x[0]/5000 + 389*x[2]/625, 0.0, 389*x[0]/625, 0.0]])
        #print(H)
        return H

def jacobian_Function(x,equation=Equation.sixhump):
    J=None
    
    if equation==equation.sixhump:
        J = np.array([x[1] - x[0].pow(2)*((21*x[0])/5 - (4*x[0].pow(3))/3) + 2*x[0]*(x[0].pow(4)/3 - (21*x[0].pow(2))/10 + 4),
              x[0] + 2*x[1]*(4*x[1].pow(2) - 4) + 8*x[1].pow(3)])
        return J
    elif equation==equation.branin:
        J =np.array( [
            - 9.602112642270262 * np.sin(x[0]) - 2.0 * (0.2583690182879614 * x[0] - 1.591549430918954) * (- 0.1291845091439807 * x[0] ** 2 + 1.591549430918954 * x[0] + x[1] - 6.0),
        - 0.2583690182879614*x[0]**2 + 3.183098861837908*x[0] + 2.0*x[1] - 12.0
        ])
        return J
    elif equation== equation.griewank:
        J = np.array([x[0] / 2000 + np.cos((2 ** (1 / 2) * x[1]) / 2) * np.sin(x[0]),
             x[1] / 2000 + (2 ** (1/2) * np.sin((2 ** (1 / 2) * x[1]) / 2) * np.cos(x[0])) / 2])
        return J
    elif equation== equation.sphere:
        J = np.array([2*x[0],
              2*x[1]])
        return J
    elif equation.engineering:
        J = np.array([(992*x[0]*x[2])/25 + (31661*x[0]*x[3])/5000 + (389*x[2]*x[3])/625,
              (17781*x[2].pow(2))/10000,
             (496*x[0].pow(2))/25 + (389*x[3]*x[0])/625 + (17781*x[1]*x[2])/5000,
              (31661*x[0].pow(2))/10000 + (389*x[2]*x[0])/625])
        return J
def cons1(x):
    res = 0
    y = x[0]-0.0193*x[2]
    if y >= 0:
        res = 1
    else:
        res = 0
    return res
def cons2(x):
    res = 0
    y = x[1]-0.00954*x[2]
    if y >= 0:
        res = 1
    else:
        res = 0
    return res
def cons3(x):
    res = 0
    y = 3.14*(x[2].pow(2))*x[3]+3.14*(4/3)*x[2].pow(3)-1296000
    if y >= 0:
        res = 1
    else:
        res = 0
    return res
def cons4(x):
    res = 0
    y = 240-x[3]
    if y >= 0:
        res = 1
    else:
        res = 0
    return res
def mean(x):
    sumx=sum(x)
    return sumx/len(x),sumx,[v/sumx for v in x ]
def std(x):
    import numpy
    return numpy.std(x)



def fitness(h:Tensor, Hd:Tensor)->Tensor:

    row = h.shape[0]
    N = 2 * row - 1 #%single or double
    L = int((N - 1) * 0.5)
    M = 64 #%Sampling
    w = np.linspace(0, np.pi, M)
    #y = Tensor(0.,dtype=ms.dtype.float32)
    n_h=np.concatenate([h,h[:L].reverse([0])],axis=0)
    n_h[L+1:]=n_h[0:L].reverse([0])
    #% Hd = [ones(1, 31), 0.7937, 0.3968, zeros(1, 31)]; %Low
    #% Hd = [zeros(1, 31), 0.3968, 0.7937, ones(1, 31)]; %High
    #对称

    #H=np.zeros((M,))
    #e=np.zeros((M,))
    #for i in range(L):
    #h=np.concatenate([h,h[0:15].reverse([0])],0)
    '''for i in range(M):
        sum1 = 0
        for j in range(L):
            Bb = 2 * h[L - j,:]*np.cos((j) * w[i])
            sum1 = sum1 + Bb
        H[i] = sum1 - h[L,:]*np.cos(0 * w[i])
        e[i] = (abs(H[i]) - abs(Hd[i])).pow(2)
        y = e[i] + y'''

    sum1=ops.sum(
        (2 * n_h[0:L + 1].reverse([0]).unsqueeze(1)* np.cos(np.linspace(0, 15, 16).unsqueeze(1) * w.unsqueeze(0)))
        ,dim=[0])
    H= sum1- n_h[L] * 1.
    e=(ops.abs(H) - ops.abs(Hd)).pow(2)


    return ops.sum(e)
def  Fir1(M,Hd,h:Tensor):
    #FIR1 1型数字滤波器
    row=h.shape[0]
    N = 2*row - 1
    L = int((N - 1) * 0.5)
    w = np.linspace(0,np.pi,M)
    H=np.zeros((M,))
    e=np.zeros((M,))
    def y_func(h):
        #Bb = 2 * h[:L + 1].reverse([0]) * np.cos(np.linspace(0, 15, 16) * np.repeat(w[i], 16))
        '''for i in range(M):
            Bb=2 * h[:L+1].reverse([0]) * np.cos(np.linspace(0,15,16) * np.repeat(w[i],16))
            sum1=ops.sum(Bb)
            #for j in range(L):
            #    Bb = 2 * h[L-j-1] * np.cos(j*w[i])
                #sum1=sum1+Bb
            H[i] = sum1 - h[L] *1. #np.cos(0*w[i])
            e[i]=(ops.abs(H[i])-ops.abs(Hd[i])).pow(2)'''

        #Bb=2 * h[:L+1].reverse([0]) * np.cos(np.linspace(0,15,16) * np.repeat(w[i],16))
        sum1 = ops.sum(
            (2 * h[0:L + 1].unsqueeze(1).reverse([0]) * np.cos(np.linspace(0, 15, 16).unsqueeze(1) * w.unsqueeze(0)))
            , dim=[0])
        H = sum1 - h[L]
        e=(ops.abs(H)-ops.abs(Hd)).pow(2)
        return ops.sum(e,dim=[0])
    #%符号式
    jacobianY = jacobian(y_func)#.permute(1,0)
    hessianY = hessian(y_func)
    return y_func,jacobianY,hessianY

def jacobian(forecast):
    return  jacrev(forecast,grad_position=0)


def hessian(y):
    hess=jacrev(jacrev(y,0),0)
    return hess

def NII(h, h_history,jacobianY,hessianY, beta, noise):

    row, col, page = h_history.shape
    sum_Ji = None
    for i in range(page):
        x=h_history.squeeze(2).squeeze(1)
        sum_Ji += jacobianY(x)

    resJ = jacobianY(h)
    resH = hessianY(h)

    a = resH.inverse() @ (resJ.unsqueeze(1) + beta * sum_Ji.unsqueeze(1) + noise)
    result = h - a

    return result


if __name__=="__main__":
    x=np.randn((16,))
    #print("x:",x.shape)
    def forecast(x):
        return ops.sum(x)
#
    '''jacob=jacobian(forecast)(x)
    hess=hessian(forecast)(x)
    print(jacob.shape)
    print(jacob)
    print(hess.shape)
    print(hess)'''
    h=Tensor([-0.797570923620687,
                -0.596831345509501,
                -0.730507978524020,
                -0.352421692004473,
                0.901090096192672,
                0.0642611212316686,
                -0.504628580130620,
                -0.125447493654226,
                0.338175566339471,
                0.0953914219057575,
                0.218108988633436,
                0.726269520504055,
                -0.238608100356329,
                0.497911230954302,
                -0.686606404884208,
                -0.883752364210057
                ])
    Hd=Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.396800000000000,0.793700000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    print(fitness(h,Hd))
    y,J,H=Fir1(64,Hd,h)
    print(J(h))
    print(H(h))



