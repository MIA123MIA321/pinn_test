import torch
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def q_continuous(x, y, b1=0.3, b2=0.6, a1=200, a2=100, gamma=1):
    return gamma * torch.exp(-a1 * (x - b1) ** 2) * \
        torch.exp(-a2 * (y - b2) ** 2)


def f_continuous(x, y, k, q_, *args1):
    m1, m2 = args1
    return torch.sin(m1 * np.pi * x) * torch.sin(m2 * np.pi * y) * \
        ((1 + q_) * k * k - (m1 ** 2 + m2 ** 2) * np.pi ** 2)


def Error(x, x_t):
    return torch.norm(x - x_t) / torch.norm(x_t)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64,
                                                       64), torch.nn.Tanh(),
                                       torch.nn.Linear(64,
                                                       64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 64),
                                       torch.nn.Tanh(), torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u, device=device),
            create_graph=True,
            only_inputs=True,
        )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


loss = torch.nn.MSELoss().cuda()


class FUNCTION:

    def __init__(self, N, n, k, m=(1, 1), u=MLP()):
        self.N = N
        self.n = n
        self.k = k
        self.m = m
        self.u = u

    def interior(self):
        x = torch.rand(self.N, 1, device=device)
        y = torch.rand(self.N, 1, device=device)
        q_ = q_continuous(x, y)
        f_ = f_continuous(x, y, self.k, q_, *self.m)
        return x.requires_grad_(True), y.requires_grad_(True), q_, f_

    def left(self):
        y = torch.rand(self.n, 1, device=device)
        y = torch.cat([y, torch.ones((1, 1), device=device)], 0)
        y = torch.cat([y, torch.zeros((1, 1), device=device)], 0)
        x = torch.zeros_like(y, device=device)
        cond = torch.zeros_like(y, device=device)
        return x.requires_grad_(True), y.requires_grad_(True), cond

    def right(self):
        y = torch.rand(self.n, 1, device=device)
        y = torch.cat([y, torch.ones((1, 1), device=device)], 0)
        y = torch.cat([y, torch.zeros((1, 1), device=device)], 0)
        x = torch.ones_like(y, device=device)
        cond = torch.zeros_like(y, device=device)
        return x.requires_grad_(True), y.requires_grad_(True), cond

    def down(self):
        x = torch.rand(self.n, 1, device=device)
        x = torch.cat([x, torch.ones((1, 1), device=device)], 0)
        x = torch.cat([x, torch.zeros((1, 1), device=device)], 0)
        y = torch.zeros_like(x, device=device)
        cond = torch.zeros_like(x, device=device)
        return x.requires_grad_(True), y.requires_grad_(True), cond

    def up(self):
        x = torch.rand(self.n, 1, device=device)
        x = torch.cat([x, torch.ones((1, 1), device=device)], 0)
        x = torch.cat([x, torch.zeros((1, 1), device=device)], 0)
        y = torch.ones_like(x, device=device)
        cond = torch.zeros_like(x, device=device)
        return x.requires_grad_(True), y.requires_grad_(True), cond

    def l_interior(self):
        x, y, q_, f_ = self.interior()
        x, y, q_, f_ = x.cuda(), y.cuda(), q_.cuda(), f_.cuda()
        uxy = self.u(torch.cat([x, y], dim=1))
        return loss(gradients(uxy, x, 2) + gradients(uxy, y, 2) +
                    self.k * self.k * (1 + q_) * uxy, f_)

    def l_left(self):
        x, y, cond = self.left()
        x, y, cond = x.cuda(), y.cuda(), cond.cuda()
        uxy = self.u(torch.cat([x, y], dim=1))
        return loss(uxy, cond)

    def l_right(self):
        x, y, cond = self.right()
        x, y, cond = x.cuda(), y.cuda(), cond.cuda()
        uxy = self.u(torch.cat([x, y], dim=1))
        return loss(uxy, cond)

    def l_down(self):
        x, y, cond = self.down()
        x, y, cond = x.cuda(), y.cuda(), cond.cuda()
        uxy = self.u(torch.cat([x, y], dim=1))
        return loss(uxy, cond)

    def l_up(self):
        x, y, cond = self.up()
        x, y, cond = x.cuda(), y.cuda(), cond.cuda()
        uxy = self.u(torch.cat([x, y], dim=1))
        return loss(uxy, cond)


parser = argparse.ArgumentParser()
parser.add_argument('--maxiter', type=int, default=10000)
parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--m', type=str, default='3,2')
parser.add_argument('--grids', type=int, default=256)
parser.add_argument('--k', type=float, default=2.)
parser.add_argument('--heatmap_dir', type=str, default='heatmap/')
parser.add_argument('--relative_dir', type=str, default='relative/')
args = parser.parse_args()

maxiter, N, n, m, grids, k = args.maxiter, args.N, args.n, args.m, args.grids, args.k
heatmap_dir, relative_dir = args.heatmap_dir, args.relative_dir
m = m.split(',')
m = (eval(m[0]), eval(m[1]))
u = MLP()
u = u.cuda()
MODULE = FUNCTION(N, n, k, m, u)
opt = torch.optim.Adam(params=u.parameters())
xc = torch.linspace(0, 1, grids, device=device)
xx, yy = torch.meshgrid(xc, xc)
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
u_real = torch.sin(m[0] * np.pi * xx) * torch.sin(m[1] * np.pi * yy)
Error_list, Loss_list = [], []
iter_list = [i * (maxiter // 10) for i in range(1, 10)]
Iter = [_ for _ in range(maxiter)]


print('start time:', '          %s' % str(datetime.now())[:-7])


for i in range(1, maxiter + 1):
    opt.zero_grad()
    l = MODULE.l_interior() + MODULE.l_left() + MODULE.l_right() + \
        MODULE.l_down() + MODULE.l_up()
    Loss_list.append(float(l))
    l.backward()
    opt.step()
    u_pred = u(xy)
    Error_list.append(float(Error(u_pred, u_real)))
    if i in iter_list:
        per = int(i / (maxiter // 10))
        print(
            '{}0% completed'.format(per),
            '        %s' % str(
                datetime.now())[
                :-7])
    elif i == maxiter:
        print('100% completed', '       %s' % str(datetime.now())[:-7])

u_pred = u_pred.cpu()
u_real = u_real.cpu()

Loss_list = [i / Loss_list[0] for i in Loss_list]
gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])
fig = plt.figure(figsize=(8, 4))
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
fig.tight_layout()
ax0.set_title("Rel-err")
ax1.set_title("Rel-loss")
plt.subplot(121)
plt.plot(Iter, Error_list, color='r', label='Rel-err')
plt.ylim(0, 1)
plt.xlabel("Iter")
plt.subplot(122)
plt.plot(Iter, np.log10(Loss_list), color='b', label='Rel-err')
plt.xlabel("Iter")
plt.ylabel('10^-')
plt.savefig(
    relative_dir +
    'N_{}_n_{}_k_{}_m_{},{}.jpg'.format(
        N,
        n,
        k,
        m[0],
        m[1]))
plt.close()


def plot_heapmap(u_list, title_list, save_path):
    max_value = max([torch.abs(u).max() for u in u_list])
    img_list = []
    lln = len(u_list)
    width_list = [4] * (lln - 1) + [5]
    cbar_list = [False] * (lln - 1) + [True]

    for i in range(lln):
        plt.figure(figsize=(width_list[i], 4))
        plt.title(title_list[i])
        sns.heatmap(
            u_list[i].reshape(grids, grids).detach().numpy(),
            xticklabels=False,
            yticklabels=False,
            cmap="gist_rainbow",
            vmin=-max_value,
            vmax=max_value,
            cbar=cbar_list[i])
        plt.savefig('.tmp.jpg')
        img_list.append(cv2.imread('.tmp.jpg'))
        os.remove('.tmp.jpg')
        plt.close()
    img1 = img_list[0]
    for i in range(1, lln):
        img1 = np.hstack((img1, img_list[i]))
    cv2.imwrite(save_path, img1)


plot_heapmap([u_pred, u_real], ['Prediction', 'Ground truth'],
             heatmap_dir + 'N_{}_n_{}_k_{}_m_{},{}.jpg'.format(N, n, k, m[0], m[1]))
