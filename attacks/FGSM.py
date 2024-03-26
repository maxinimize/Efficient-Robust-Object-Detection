import torch
import torch.nn.functional as F
from attacks.attacker import Attacker
from utils.loss import compute_loss

class FGSM(Attacker):
    def __init__(self, model, config=None, target=None, epsilon=0.2):
        super(FGSM, self).__init__(model, config, epsilon)
        self.target = target # target class
        self.epsilon = epsilon

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        self.model.train() # has to be in train, or the model output dims change
        x.requires_grad=True
        logits = self.model(x)
        loss, loss_components = compute_loss(logits, y, self.model)
        if x.grad is not None:
            x.grad.data.fill_(0)

        self.model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            gradients_sign = x.grad.sign() # Collect the sign of the gradients
            x_adv = x + self.epsilon * gradients_sign # Generate perturbed image
            x_adv = torch.clamp(x_adv, 0, 1) # Clip the perturbed image to ensure it stays within valid pixel range

        return x_adv