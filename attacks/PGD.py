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
            # Handle both DDP and non-DDP models
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.train()
            
            # Random initialization (helps escape local minima)
            x_adv = x.clone().detach()
            x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            for i in range(self.epoch):
                # Reset gradients once at the beginning of each iteration
                model.zero_grad()
                x_adv.requires_grad = True
                logits = model(x_adv) #f(T((x))
                loss, loss_components = compute_loss(logits, y, self.model)
                loss.backward()   
                
                # Get gradients                
                grad = x_adv.grad.detach()
                
                # Use sign method for efficiency, can be replaced with normalized gradient for stronger attacks
                # grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                # grad = grad / (grad_norm + 1e-8)
                grad = grad.sign()
                
                x_adv = x_adv + self.lr * grad

                # Projection
                x_adv = x + torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(x_adv, 0, 1).detach()
                # No need for second zero_grad call here
                
            # Final clipping to ensure the perturbation is within bounds
            perturbation = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
            x_adv = torch.clamp(x + perturbation, 0, 1)
            return x_adv