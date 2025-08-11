import time
import torch
import torch.nn.functional as F
from attacks.attacker import Attacker
from utils.loss import compute_loss

class PGD(Attacker):
    def __init__(self, model, config=None, target=None, epsilon=0.2, lr = 0.01, epoch = 10):
        super(PGD, self).__init__(model, config, epsilon)
        self.target = target
        self.epsilon = epsilon # total update limit
        self.lr = lr # amount of update in each step
        self.epoch = epoch # time of attack steps

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        
        with torch.enable_grad():
            self.model.train()
            x_adv = x.clone().detach()
            for _ in range(self.epoch):
                self.model.zero_grad()
                x_adv.requires_grad = True
                logits = self.model(x_adv) #f(T((x))
                loss, loss_components = compute_loss(logits, y, self.model)
                loss.backward()   
                                   
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv + self.lr * grad

                # Projection
                x_adv = x + torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
                x_adv = x_adv.detach()
                x_adv = torch.clamp(x_adv, 0, 1)
                self.model.zero_grad()
            return x_adv