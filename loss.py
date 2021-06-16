import torch

def Expected01Bin(SNN, input, target):
  assert SNN.out_features == 2
  M, Q = SNN.GetMQ(input)
  #MM = M0(x) - M1(x) batchwise
  MM = (M * torch.tensor([1, -1], device=SNN.device)).sum(-1)
  #QQ = Q00(x) + Q11(x)^2 - 2*Q01(x) batchwise
  QQ = (Q * torch.tensor([[1,-1],[-1,1]], device=SNN.device)).sum((-1,-2))
  Z = MM/torch.sqrt(QQ)
  return ((1-target)*gauss_ccdf(Z) + target*gauss_ccdf(-Z)).mean()


def Expected01(SNN, input, target, samples=10**4): #input: [batch, classes], target : [batch]
  classes = SNN.out_features
  M, Q = SNN.GetMQ(input) #Q : [batch, classes, classes], M : [batch, classes]
  perm = _mk_perm(classes, device=SNN.device) #perm : [classes, classes, classes]
  PERM = perm[target] #PERM : [batch, classes, classes]
  Q = PERM @ Q @ PERM #Q : [batch, classes, classes]
  M = (PERM @ M.unsqueeze(-1)).squeeze(-1) #M : [batch, classes]
  Ch = torch.cholesky(Q) #Ch : [batch, classes, classes]
  tCh = tilde_A(Ch).unsqueeze(-3) #A : [batch, 1, classes-1, classes-1]
  tM = tilde_M(M, Ch).unsqueeze(-2) #tM : [batch, 1, classes-1]
  X = torch.randn(samples, classes-1, 1, device=SNN.device, requires_grad=False) #X : [samples, classes-1, 1]
  tX = tM + torch.matmul(tCh,X).squeeze(-1) #tX : [batch, samples, classes-1]
  return 1 - gauss_ccdf(tX.max(-1)[0]).mean()

def _mk_perm(dim, device):
  OUT = torch.eye(dim, device=device, requires_grad=False).repeat(dim, 1, 1)
  for k, i in enumerate(OUT):
    i[k,k] = 0.
    i[-1,-1] = 0.
    i[k, -1] = 1.
    i[-1, k] = 1.
  return OUT

def gauss_ccdf(x):
  return (1-torch.erf(x/2**.5))/2

def tilde_A(A):
  return (A[..., :-1, :-1]-A[..., -1, :-1].unsqueeze(-2))/A[..., -1, -1].unsqueeze(-1).unsqueeze(-1)

def tilde_M(M, A):
  return (M[..., :-1]-M[..., -1].unsqueeze(-1))/A[..., -1,-1].unsqueeze(-1)
