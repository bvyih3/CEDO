import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import json
import os
import sys

sys.path.append(os.getcwd())
import utils.config as config
from main_arcface_ours import parse_args


class uniform_loss(nn.Module):
    def __init__(self, t=0.07):
        super(uniform_loss, self).__init__()
        self.t = t

    def forward(self, x):
        return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    dataset = args.dataset
    config.dataset = dataset
    config.update_paths(args.dataset)

    ans_list = json.load(
        open(os.path.join(config.cache_root, "traintest_label2ans.json"), "r")
    )

    N = len(ans_list)
    M = 768
    print("N =", N)
    print("M =", M)
    criterion = uniform_loss()
    x = Variable(torch.randn(N, M).float(), requires_grad=True)
    optimizer = optim.Adam([x], lr=1e-3)
    min_loss = 100
    optimal_target = None
    
    N_iter = 10000
    for i in range(N_iter):
        x_norm = F.normalize(x, dim=1)
        loss = criterion(x_norm)
        if i % 100 == 0:
            print(i, loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            optimal_target = x_norm
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(os.path.join(config.qa_path, 'optimal_{}_{}.npy'.format(N, M)))
    np.save(os.path.join(config.qa_path, 'optimal_{}_{}.npy'.format(N, M)), optimal_target.detach().numpy())

    target = np.load(os.path.join(config.qa_path, "optimal_{}_{}.npy".format(N, M)))
    print("optimal loss = ", criterion(torch.tensor(target)).item())
    # print(target)
    target = torch.tensor(target)

    # Compute dot product between target and its transpose
    dis = torch.mm(target, target.T)
    # Create mask for values greater than 1.0
    mask = dis > 1.0
    dis[mask] = 1.0
    # Compute the angles using arccos
    angles = torch.acos(dis)
    # Get indices of the closest 6 neighbors (excluding self)
    idxs = torch.argsort(angles, dim=1)[:, 1:7]
    # Sort angles to get the smallest 6
    angles = torch.sort(angles, dim=1).values[:, 1:7]
    # Compute the average angle and divide by 4
    average_angles = torch.mean(angles, dim=1) / 4
    sin_theta = torch.sin(average_angles)
    # 优化过程调整 v3 使其与 v2 满足目标距离条件，同时保持与 v1 的角度约束
    target_distance = torch.sqrt(torch.tensor(3.0)) * sin_theta  # 计算目标距离
    tolerance = 1e-5  # 设置允许的误差范围

    sub_pros = []
    maxi = 0
    maxbia = 0

    # Iterate over each class in target
    for i in range(target.size(0)):
        print("-------------")
        print(i)
        tmp = []
        v1 = target[i]
        theta = average_angles[i]

        # Get the dimension of the input vector
        dim = v1.size(0)

        # Generate a random vector
        tangent_vec = torch.randn(3, dim, device=v1.device)
        for j in range(tangent_vec.shape[0]):
            # Project onto the tangent space, making it orthogonal to the input vector
            tangent_vec[j] -= torch.dot(tangent_vec[j], v1) * v1
            tangent_vec[j] /= torch.norm(tangent_vec[j])  # Normalize

        # Rotate the vector by the given angle theta using the tangent vector
        v2 = torch.cos(theta) * v1 + torch.sin(theta) * tangent_vec[0]
        v3 = torch.cos(theta) * v1 + torch.sin(theta) * tangent_vec[1]
        v4 = torch.cos(theta) * v1 + torch.sin(theta) * tangent_vec[2]
        v2 /= torch.norm(v2)
        v3 /= torch.norm(v3)  # 单位化
        v4 /= torch.norm(v4)  # 单位化

        # Return the normalized new vector to ensure unit length
        sub_pro1 = v2

        # 迭代优化 v3
        for i in range(100):
            # 计算与 v2 方向的调整
            current_distance = torch.norm(v3 - v2)
            distance_error = target_distance[j] - current_distance

            # 根据误差调整 v3 的方向
            v3 -= distance_error * (v2 - v3)

            # 同时保持 v1 和 v3 的角度不变，通过角度修正
            cos_angle_v1_v3 = torch.dot(v1, v3) / (torch.norm(v1) * torch.norm(v3))
            angle_error = torch.acos(cos_angle_v1_v3) - theta
            v3 += 5 * angle_error * (v1 - v3)

            v3 /= v3.norm()  # 保持 v3 单位化

            # 检查误差是否在容差范围内
            if abs(distance_error) < tolerance and abs(angle_error) < tolerance:
                break
        sub_pro2 = v3

        tolerance = 5e-3  # 设置允许的误差范围
        # 迭代优化 v4
        for i in range(1000):
            # 计算与 v2 方向的调整
            current_distance1 = torch.norm(v4 - v2)
            distance_error1 = target_distance[j] - current_distance1

            # 根据误差调整 v4 的方向
            v4 -= 1 * distance_error1 * (v2 - v4)

            # 计算与 v3 方向的调整
            current_distance2 = torch.norm(v4 - v3)
            distance_error2 = target_distance[j] - current_distance2

            # 根据误差调整 v4 的方向
            v4 -= 1 * distance_error2 * (v3 - v4)

            # 同时保持 v1 和 v4 的角度不变，通过角度修正
            cos_angle_v1_v4 = torch.dot(v1, v4) / (torch.norm(v1) * torch.norm(v4))
            angle_error = torch.acos(cos_angle_v1_v4) - theta
            v4 += 2.5 * angle_error * (v1 - v4)

            v4 /= v4.norm()  # 保持 v3 单位化

            maxi = max(maxi, i)
            # 检查误差是否在容差范围内
            if (
                abs(torch.norm(v1 - v2, p=2) - torch.norm(v1 - v3, p=2)) < tolerance
                and abs(torch.norm(v1 - v2, p=2) - torch.norm(v1 - v4, p=2)) < tolerance
                and abs(torch.norm(v2 - v3, p=2) - torch.norm(v2 - v4, p=2)) < tolerance
                and abs(torch.norm(v2 - v3, p=2) - torch.norm(v3 - v4, p=2)) < tolerance
            ):
                break
        sub_pro3 = v4

        maxbia = max(maxbia, abs(torch.norm(v1 - v2, p=2) - torch.norm(v1 - v3, p=2)))
        maxbia = max(maxbia, abs(torch.norm(v1 - v2, p=2) - torch.norm(v1 - v4, p=2)))
        maxbia = max(maxbia, abs(torch.norm(v2 - v3, p=2) - torch.norm(v2 - v4, p=2)))
        maxbia = max(maxbia, abs(torch.norm(v2 - v3, p=2) - torch.norm(v3 - v4, p=2)))

        # 验证几何约束

        assert (
            abs(torch.norm(v1 - v2, p=2) - torch.norm(v1 - v3, p=2)) < tolerance
            and abs(torch.norm(v1 - v2, p=2) - torch.norm(v1 - v4, p=2)) < tolerance
        )
        assert (
            abs(torch.norm(v2 - v3, p=2) - torch.norm(v2 - v4, p=2)) < tolerance
            and abs(torch.norm(v2 - v3, p=2) - torch.norm(v3 - v4, p=2)) < tolerance
        )

        # Append the results to the list
        tmp.append(sub_pro1.tolist())
        tmp.append(sub_pro2.tolist())
        tmp.append(sub_pro3.tolist())
        sub_pros.append(tmp)
    print("maxi:", maxi)  # 504
    print("maxbia:", maxbia)  # 0.0026
    with open(os.path.join(config.qa_path, "sub_pros.json"), "w") as f:
        json.dump(sub_pros, f, indent=4)
