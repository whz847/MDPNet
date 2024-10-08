# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:03:06 2023

@author: Sana
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, output, target):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, self.alpha)
        p = torch.exp(-logp)
        focal_loss = (1-p)**self.gamma*logp
 
        return torch.mean(focal_loss)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance. 
        You can start with a small value and gradually increase it to observe the impact on the model's performance. 
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.
   
        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem. 
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies 
        the impact of the logits and can be useful when dealing with highly imbalanced datasets. 
        You can experiment with different values of s to find the one that works best for your dataset and model.

        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)

        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        index_bool=index.bool()
        output = torch.where(index_bool, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class LMFLoss(nn.Module):
        def __init__(self,cls_num_list,weight,alpha=1,beta=1, gamma=2, max_m=0.5, s=30):
            super().__init__()
            self.focal_loss = FocalLoss(weight, gamma)
            self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
            self.alpha= alpha
            self.beta = beta

        def forward(self, output, target):
            focal_loss_output = self.focal_loss(output, target)
            ldam_loss_output = self.ldam_loss(output, target)
            total_loss = self.alpha*focal_loss_output + self.beta*ldam_loss_output
            return total_loss

if __name__=='__main__':
    # 假设我们有如下数据
    num_classes = 2
    cls_num_list = [500] * num_classes  # 假设类别数量均衡
    alpha = torch.tensor([5.0] * num_classes)  # Focal Loss 的权重参数
    gamma = 2
    max_m = 0.5
    s = 30
    weight = torch.tensor([101.0,67.0])
    batch_size = 64
    input_size = 2  # 假设输出层的特征维度
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU如果可用，否则使用CPU

    # 将模型和损失函数移动到设备
    focal_loss = FocalLoss(alpha.to(device), gamma)
    ldam_loss = LDAMLoss(cls_num_list, max_m, weight.to(device) if weight is not None else None, s).to(device)
    lmf_loss = LMFLoss(cls_num_list, weight.to(device) if weight is not None else None, alpha=1, beta=1, gamma=gamma,
                       max_m=max_m, s=s).to(device)

    # 确保数据也在相同设备上
    output = torch.randn(batch_size, num_classes).to(device)
    target = torch.randint(0, num_classes, size=(batch_size,), dtype=torch.long).to(device)

    # 计算损失时就不会再报错
    fl_loss = focal_loss(output, target)
    print(f"Focal Loss: {fl_loss.item()}")
    ldam_loss_value = ldam_loss(output, target)
    print(f"LDAM Loss: {ldam_loss_value.item()}")
    lmf_loss_value = lmf_loss(output, target)
    print(f"LMF Loss: {lmf_loss_value.item()}")