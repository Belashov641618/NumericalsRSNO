import torch
from torch import Tensor
from typing import Literal



def material_transition(n0: Tensor, n1: Tensor):
    M = n0 / n1
    mask = (2 * torch.eye(2, device=n0.device) - 1).view(*[1]*len(M.shape), 2, 2)
    transition = 0.5*(1.0 + M.view(*M.shape,1,1)*mask)
    return transition.to(torch.complex128)

def material_propagation(wavelength:Tensor, n:Tensor, d:Tensor):
    device = wavelength.device
    k = 2*torch.pi*n/wavelength
    arg = (1j*k*d).to(torch.complex128)
    mask = (torch.eye(2,device=device)*(-torch.arange(-1,2,2,device=device)).view(-1,1)).view(*[1]*len(arg.shape),2,2)
    propagation = torch.exp(arg.view(*arg.shape,1,1)*mask)
    return torch.where(mask==0,0,propagation)

def matrix_power(matrix:Tensor, powers:Tensor):
    expanding = [max(dim0, dim1) for dim0, dim1 in zip(matrix.shape, powers.shape)]
    powers = powers.expand(*expanding)
    matrix = matrix.expand(*expanding, *matrix.shape[-2:])
    unique_power = torch.unique(powers, sorted=True)
    n_power_matrix = torch.zeros_like(matrix, device=matrix.device)
    for power in unique_power:
        indices = (powers == power.item()).view(*powers.shape, 1, 1).expand(*powers.shape, 2, 2)
        temp = torch.linalg.matrix_power(matrix[indices].view(-1, 2, 2), power.item())
        n_power_matrix[indices] = temp.view(-1)
    return n_power_matrix



class BasicElement(torch.nn.Module):
    _conversation:torch.Tensor
    def _register_conversation(self, matrix:Tensor):
        self.register_buffer("_conversation", matrix.to(torch.complex128))
    @property
    def conversation(self):
        return self._conversation

    _first_n:Tensor
    def _register_first_n(self, tensor:Tensor):
        self.register_buffer("_first_n", tensor)
    @property
    def first_n(self):
        return self._first_n

    _last_n:Tensor
    def _register_last_n(self, tensor:Tensor):
        self.register_buffer("_last_n", tensor)
    @property
    def last_n(self):
        return self._last_n

    @property
    def reflection(self):
        t11, t12, t21, t22 = self.conversation[...,0,0], self.conversation[...,0,1], self.conversation[...,1,0], self.conversation[...,1,1]
        return -t21/t22
    @property
    def transition(self):
        t11, t12, t21, t22 = self.conversation[...,0,0], self.conversation[...,0,1], self.conversation[...,1,0], self.conversation[...,1,1]
        return t11-t12*t21/t22

    @property
    def device(self):
        return self._conversation.device

    def __init__(self):
        super().__init__()

class Crystal(BasicElement):
    def __init__(self, *elements:BasicElement):
        super().__init__()
        conversation = elements[0].conversation
        for element1, element0 in zip(elements[1:], elements[:-1]):
            conversation = conversation @ material_transition(element0.first_n, element1.last_n) @ element1.conversation
        self._register_conversation(conversation)



class Layer(BasicElement):
    def __init__(self, wavelength:Tensor, n:Tensor, d:Tensor):
        super().__init__()
        self._register_conversation(material_propagation(wavelength, n, d))
        self._register_first_n(n)
        self._register_last_n(n)

class DoubleLayerMassive(BasicElement):
    def __init__(self, wavelength:Tensor, n1:Tensor, d1:Tensor, n2:Tensor, d2:Tensor, pairs:Tensor):
        super().__init__()
        propagation1 = material_propagation(wavelength, n1, d1)
        propagation2 = material_propagation(wavelength, n2, d2)
        transition12 = material_transition(n1,n2)
        transition21 = material_transition(n2,n1)
        self._register_first_n(n1)
        self._register_last_n(n2)
        self._register_conversation(propagation1 @ transition12 @ matrix_power(propagation2@transition21@propagation1@transition12, pairs) @ propagation2)

class DoubleLayerMassiveNonOrthogonal(BasicElement):
    def __init__(self, wavelength:Tensor, n1:Tensor, d1:Tensor, n2:Tensor, d2:Tensor, pairs:Tensor, polarization:Literal["S","P"], theta:Tensor):
        super().__init__()
        alpha1 = -n1*torch.cos(theta) if polarization == "S" else +n1/torch.cos(theta)
        alpha2 = -n2*torch.cos(theta) if polarization == "S" else +n2/torch.cos(theta)
        propagation1 = material_propagation(wavelength, n1, d1)
        propagation2 = material_propagation(wavelength, n2, d2)
        transition12 = material_transition(alpha1,alpha2)
        transition21 = material_transition(alpha2,alpha1)
        self._register_first_n(n1)
        self._register_last_n(n2)
        self._register_conversation(propagation1 @ transition12 @ matrix_power(propagation2@transition21@propagation1@transition12, pairs) @ propagation2)

class Air(BasicElement):
    def __init__(self, n0:Tensor):
        super().__init__()
        self._register_first_n(n0)
        self._register_last_n(n0)
        self._register_conversation(torch.eye(2,device=n0.device).view(*[1]*len(n0.shape),2,2))
