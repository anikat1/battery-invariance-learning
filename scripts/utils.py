import torch
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def jac_calc(nr,dr): #cont enc
    jac=list()
    for i in range(nr.size(0)):
        _jac = torch.autograd.grad(nr[i],dr,retain_graph=True)
        jac.append(_jac[0].item())
    #pdb.set_trace()
    
    #print("Size of Latent Rep {}".format(nr.size()))
    #print("Size of Input {}".format(dr.size()))
    #print("Size of Jacobian {}".format(len(jac)))
    #Jacobian should be nr.size() \times dr.size() so each row indicates the sensitivity of latent feature `i' to all indices of encoding
    return jac

def jac_calc2(nr,dr): #material disc enc
    jac=list()
    for i in range(nr.size(0)):
        _jac = torch.autograd.grad(nr[i],dr,retain_graph=True)
        jac.append(_jac[0])
    #pdb.set_trace()
    
    stacked_jac = torch.stack(jac).squeeze()

    #print("Size of Latent Rep {}".format(nr.size()))
    #print("Size of Input {}".format(dr.size()))
    #print("Size of Jacobian {}".format(stacked_jac.size()))
    #Jacobian should be nr.size() \times dr.size() so each row indicates the sensitivity of latent feature `i' to all indices of encoding
    return stacked_jac.cpu().data.numpy()

def jac_calc3(nr,dr): #soc dis enc
    jac=list()
    for i in range(nr.size(0)):
        _jac = torch.autograd.grad(nr[i],dr,retain_graph=True)
        jac.append(_jac[0].cpu().data.numpy()[0,1]) #for 3 classes of material
    
    #print("Size of Latent Rep {}".format(nr.size()))
    #print("Size of Input {}".format(dr.size()))
    #print("Size of Jacobian {}".format(stacked_jac.size()))
    #Jacobian should be nr.size() \times dr.size() so each row indicates the sensitivity of latent feature `i' to all indices of encoding
    return jac

