import torch
from torch import Tensor
from typing import Literal

def material_transition(n0: Tensor, n1: Tensor, theta:Tensor, polarization:Literal["S","P"]):
    alpha0 = -torch.sqrt(n0**2-torch.sin(theta)**2) if polarization == "S" else +n0**2/torch.sqrt(n0**2-torch.sin(theta)**2)
    alpha1 = -torch.sqrt(n1**2-torch.sin(theta)**2) if polarization == "S" else +n1**2/torch.sqrt(n1**2-torch.sin(theta)**2)
    M = alpha0 / alpha1
    mask = (2 * torch.eye(2, device=n0.device) - 1).view(*[1]*len(M.shape), 2, 2)
    transition = 0.5*(1.0 + M.view(*M.shape,1,1)*mask)
    return transition.to(torch.complex128)
def material_propagation(wavelength:Tensor, theta:Tensor, n:Tensor, d:Tensor):
    device = wavelength.device
    k0 = 2*torch.pi/wavelength
    kz = k0*torch.sqrt(n**2-torch.sin(theta)**2)
    k = kz
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
    def inverse_reflection(self):
        t11, t12, t21, t22 = self.conversation[...,0,0], self.conversation[...,0,1], self.conversation[...,1,0], self.conversation[...,1,1]
        return +t12/t22
    @property
    def inverse_transition(self):
        t11, t12, t21, t22 = self.conversation[...,0,0], self.conversation[...,0,1], self.conversation[...,1,0], self.conversation[...,1,1]
        return 1/t22

    @property
    def s_matrix(self):
        r_uu, r_dd, t_ud, t_du = self.reflection, self.inverse_reflection, self.inverse_transition, self.transition
        return torch.stack([
            torch.stack([r_uu, t_ud], dim=-1),
            torch.stack([t_du, r_dd], dim=-1),
        ], dim=-2)
    @property
    def r_matrix(self):
        return torch.linalg.inv(self.s_matrix)

    def wavelength_correction(self, wavelength_real:Tensor, wavelength_imag:Tensor):
        r_matrix = self.r_matrix
        r_grads_real, r_grads_imag = [], []
        for i in range(2):
            r_grads_real_, r_grads_imag_ = [], []
            for j in range(2):
                real_part = r_matrix[..., i, j].real
                imag_part = r_matrix[..., i, j].imag
                grad_real_real = torch.autograd.grad(outputs=real_part, inputs=wavelength_real, grad_outputs=torch.ones_like(real_part), create_graph=True, retain_graph=True, only_inputs=True)[0]
                grad_imag_real = torch.autograd.grad(outputs=imag_part, inputs=wavelength_real, grad_outputs=torch.ones_like(imag_part), create_graph=True, retain_graph=True, only_inputs=True)[0]
                grad_real_imag = torch.autograd.grad(outputs=real_part, inputs=wavelength_imag, grad_outputs=torch.ones_like(real_part), create_graph=True, retain_graph=True, only_inputs=True)[0]
                grad_imag_imag = torch.autograd.grad(outputs=imag_part, inputs=wavelength_imag, grad_outputs=torch.ones_like(imag_part), create_graph=True, retain_graph=True, only_inputs=True)[0]
                grad_ij_real = grad_real_real.to(torch.complex128) + 1j*grad_imag_real.to(torch.complex128)
                grad_ij_imag = grad_real_imag.to(torch.complex128) + 1j*grad_imag_imag.to(torch.complex128)
                r_grads_real_.append(grad_ij_real)
                r_grads_imag_.append(grad_ij_imag)
            r_grads_real.append(torch.stack(r_grads_real_,dim=-1))
            r_grads_imag.append(torch.stack(r_grads_imag_,dim=-1))
        r_grads_real = torch.stack(r_grads_real, dim=-1).swapdims(-1,-2)
        r_grads_imag = torch.stack(r_grads_imag, dim=-1).swapdims(-1,-2)
        r_grads = r_grads_real + 1j*r_grads_imag
        a_matrix = -torch.linalg.inv(r_grads) @ r_matrix
        D, V = torch.linalg.eig(a_matrix)
        deltas = torch.gather(D, dim=-1, index=torch.argmin(D.abs(), dim=-1, keepdim=True)).squeeze(-1)
        return wavelength_real + 1j*wavelength_imag - deltas

    @property
    def device(self):
        return self._conversation.device

    def __init__(self):
        super().__init__()
class Crystal(BasicElement):
    def __init__(self, theta:Tensor, polarization:Literal["S","P"], *elements:BasicElement):
        super().__init__()
        conversation = elements[0].conversation
        for element1, element0 in zip(elements[1:], elements[:-1]):
            conversation = conversation @ material_transition(element0.first_n, element1.last_n, theta=theta, polarization=polarization) @ element1.conversation
        self._register_conversation(conversation)



class DoubleLayerMassive(BasicElement):
    def __init__(self, wavelength:Tensor, theta:Tensor, polarization:Literal["S","P"], n1:Tensor, d1:Tensor, n2:Tensor, d2:Tensor, pairs:Tensor):
        super().__init__()
        propagation1 = material_propagation(wavelength, theta, n1, d1)
        propagation2 = material_propagation(wavelength, theta, n2, d2)
        transition12 = material_transition(n1,n2,theta,polarization)
        transition21 = material_transition(n2,n1,theta,polarization)
        self._register_first_n(n1)
        self._register_last_n(n2)
        self._register_conversation(propagation1 @ transition12 @ matrix_power(propagation2@transition21@propagation1@transition12, pairs) @ propagation2)
class Layer(BasicElement):
    def __init__(self, wavelength:Tensor, theta:Tensor, n:Tensor, d:Tensor):
        super().__init__()
        self._register_conversation(material_propagation(wavelength, theta, n, d))
        self._register_first_n(n)
        self._register_last_n(n)
class Air(BasicElement):
    def __init__(self, n0:Tensor):
        super().__init__()
        self._register_first_n(n0)
        self._register_last_n(n0)
        self._register_conversation(torch.eye(2,device=n0.device).view(*[1]*len(n0.shape),2,2))
