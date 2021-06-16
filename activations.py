import torch
import torch.nn.functional as F
import math


__sin_dict__ = {
  'name': 'sin',
  'fct': torch.sin,
  'm': lambda A, M: torch.exp(-A**2/2)*torch.sin(M),
  'Epr': lambda Q, M, l_nb: sin_E_prod(Q, M, l_nb)}


__ReLU_dict__ = {
  'name': 'ReLU',
  'fct': F.relu,
  'm': lambda A, M: .5 * (M + A*torch.exp(-M**2/(2*A**2))*math.sqrt(2/math.pi) + M*torch.erf(M/(A*math.sqrt(2)))),
  'Epr': lambda Q, M, l_nb: relu_E_prod(Q, M, l_nb)}



__act__ = {
  'sin': __sin_dict__,
  'relu': __ReLU_dict__,
}

def sin_E_prod(Q, M, l_nb):
  assert l_nb == 0, "Multilayer StochNet network not implemented"
  return (1 - torch.exp(-2*Q)*torch.cos(2*M)) / 2


def relu_E_prod(Q, M, l_nb):
  assert l_nb == 0, "Multilayer StochNet not implemented"
  return .5 * (Q + M**2 + torch.sqrt(2*Q/math.pi)*M*torch.exp(-M**2/(2*Q)) + (Q+M**2)*torch.erf(M/(torch.sqrt(2*Q))))
