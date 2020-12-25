from mpi4py import MPI
import os, subprocess, sys
import numpy as np


def mpi_fork(n, bind_to_core=False):
    """
    :param n: Number of process to split into
    :param bind_to_core: Bind each MPI process to a core.
    """
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", IN_MPI="1")
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(msg):
    print("Message from %d: %s" % (MPI.COMM_WORLD.Get_rank(), msg))


def proc_id():
    """Get rank of process"""
    return MPI.COMM_WORLD.Get_rank()


def num_procs():
    """Get number of active MPI processes"""
    return MPI.COMM_WORLD.Get_size()


def broadcast(x, root=0):
    """Broadcast value x at specific process to all processes"""
    return MPI.COMM_WORLD.Bcast(x, root=root)


def mpi_op(x, op):
    x, is_scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buffer = np.zeros_like(x, dtype=np.float32)
    MPI.COMM_WORLD.Allreduce(x, buffer, op=op)
    return buffer[0] if is_scalar else buffer


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    """Average a scalar or vector over MPI processes"""
    return mpi_sum(x) / num_procs()


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    :param x: An array containing samples of the scalar
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean) ** 2))
    std = np.sqrt(global_sum_sq / global_n)

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
