## Module

$$
\begin{aligned}
\Delta u + k^2(1+q)u &= f \,\, \text{in} \, \,\Omega\\
u &= 0\,\, \text{on}\,\, \partial \Omega\\
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

* $n$ points on each boundary of $\partial \Omega$  

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

#### Modifying the following arguments in (test.sh):

