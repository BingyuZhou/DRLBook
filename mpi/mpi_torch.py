import torch
from mpi4py import MPI
from mpi_tools import num_procs, proc_id, broadcast, mpi_avg


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's Pytorch using more than its fair share of CPU resources.
    """
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(torch.get_num_threads() // num_procs(), 1)
    torch.set_num_threads(fair_num_threads)


def mpi_avg_grads(module):
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_grad_np = p.grad.numpy()
        avg_p_grad = mpi_avg(p_grad_np)
        p_grad_np[:] = avg_p_grad[:]


def sync_params(module):
    if num_procs() == 1:
        return

    for p in module.parameters():
        p_np = p.data.numpy()
        broadcast(p_np)
