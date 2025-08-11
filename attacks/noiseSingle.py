import torch
import torch.nn.functional as F
from attacks.attacker import Attacker
from utils.loss import compute_loss

class Noise(Attacker):
    def __init__(self, model, config=None, target=None, epsilon=0.2):
        super(Noise, self).__init__(model, config, epsilon)
        self.target = target # target class
        self.epsilon = epsilon

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nosie = torch.normal(0, self.epsilon/2, size=x.shape).to(device)
        x_adv = x + torch.clamp(nosie, min=-self.epsilon, max=self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv