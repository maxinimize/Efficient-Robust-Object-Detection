import torch
import torch.nn.functional as F
from attacks.attacker import Attacker
from utils.loss import compute_loss

class CnW(Attacker):
    def __init__(self, model, config=None, target=None, epsilon=0.2, lr=0.01, epoch=10, confidence=0, c=1e-4):
        super(CnW, self).__init__(model, config, epsilon)
        self.target = target
        self.epsilon = epsilon  # total update limit
        self.lr = lr  # amount of update in each step
        self.epoch = epoch  # time of attack steps
        self.confidence = confidence  # Confidence parameter for C&W loss
        self.c = c  # Constant for box-constrained optimization, balance the two losses

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target: Target label
        :return adversarial image
        """

        self.model.train()
        x_adv = x.detach().clone()
        y[:, 1] = 1 # change target labels to class 1
        x_adv = x.detach().clone()
        x_adv.requires_grad = True

        for _ in range(self.epoch):
            self.model.zero_grad()
            logits = self.model(x_adv)
            target_loss, loss_components = compute_loss(logits, y, self.model) # loss
            l2_norm = torch.sum(torch.square(x_adv - x))  # Perturbation  L2 norm
            loss = target_loss + self.c * l2_norm # C&W loss function
            loss.backward()
            grad = x_adv.grad.detach().sign()
            x_adv = x_adv + self.lr * grad
            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
        return x_adv