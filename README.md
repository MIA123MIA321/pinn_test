## Module

$$
\begin{aligned}
\Delta u + k^2(1+q)u &= f  \quad \text{in}\quad  \Omega\\
u &= 0\quad \text{on} \quad \partial \Omega\\
\end{aligned}
$$

Given $q,f$ , to find a solution $u$ .

***

## Setting

$$
\begin{aligned}
\Omega &= (0,1)^2\\
q(x,y) &= \exp(-200(x-0.3)^2-100(y-0.6)^2)\\
u_{truth} &= \sin(m_1 x)\sin(m_2 y),m_1,m_2 \in \mathbb{N}^+\\
f(x,y) &= (k^2(1+q(x,y)) - m_1^2-m_2^2)\sin(m_1 x)\sin(m_2 y)
\end{aligned}
$$

We input the $q,f$ to the network, aftre some training, we get the output $u_{res}$ .<br/>

We test the relative error on the resolution = 256 $\times$ 256

***

## Network

Unsupervised Learning 

* nn.Linear(2,64)
* nn.Tanh()
* Nn.Linear(64,64)

* nn.Tanh()
* nn.Linear(64,64)
* nn.Tanh()
* nn.Linear(64,64)
* nn.Tanh()
* nn.Linear(64,1)

#### Sampling

Random Sampling

* $N$ points in $\Omega$ 

* $n$ 

#### Loss function

MSE LOSS

$$
\begin{aligned}
l_{\text {int }}(x, y) &=\left(\Delta u_{\text {res }}+k^2(1+q) u_{\text {res }}-f\right)(x, y) \\
l_{\text {left }}(0, y) &=u(0, y) \\
l_{\text {right }}(1, y) &=u(1, y) \\
l_{\text {up }}(x, 1) &=u(x, 1) \\
l_{\text {down }}(x, 0) &=u(x, 0) \\
l_{\text {total }} &=\operatorname{MSE}\left(l_{\text {int }}\right)+\operatorname{MSE}\left(l_{\text {left }}\right)+\operatorname{MSE}\left(l_{\text {right }}\right)+\operatorname{MSE}\left(l_{\text {up }}\right)+\operatorname{MSE}\left(l_{\text {down }}\right)
\end{aligned}
$$

#### Optimizer

Adam , learning rate = 1e-4

***

## Getting Started

1. Modifying the following arguments in `test.sh`:
* maxiter : total epochs of trainging
* N : number of points sampled in $\Omega$
* n : number of points sampled on each boundary of $\partial \Omega$  
* grids : resolution of testing
* k : frequency of the equation
* m : frequency of $u_{truth}$
* gpu : 'yes' or 'no' to compute with gpu or not
2.  `bash test.sh`
3.  `tail -f .tmp.log` to get the code progress
4.  check `heatmap/` for the training results and `relative/` for the process of training
    `'maxiter_cpu/gpu_N_n_k_m.jpg'`

## Examples of Results
N = 40000 n = 1000 k = 2 m = (3,4) with `GPU` after 10000 epochs
<img src="https://github.com/MIA123MIA321/pinn_test/blob/main/heatmap/gpu_10000_N_40000_n_5000_k_2.0_m_3%2C4.jpg" width="800"/>
<img src="https://github.com/MIA123MIA321/pinn_test/blob/main/relative/gpu_10000_N_40000_n_5000_k_2.0_m_3%2C4.jpg" width="800"/>
