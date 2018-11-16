import os
import torch

from .general import approximate_size

def get_device():
    """Return torch.device('cuda') if cuda is available else torch.device('cpu')
    """
    has_cuda = torch.cuda.is_available()
    return torch.device('cuda' if has_cuda else 'cpu')


def get_cpu_count():
    """Return the number of available cpus
    """
    if hasattr(os, 'sched_getaffinity'):
        return os.sched_getaffinity(0)
    else:
        return os.cpu_count()


def get_gpu_count():
    """Return the number of available cpus
    """
    return torch.cuda.device_count()


def get_gpu_info(return_info=False):
    """Return the properties of available GPUs
    """
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        print('No available GPUs.')
        return None
    else:
        gpu_info = {}
        for gpu_idx in range(gpu_count):
            gpu_properties = torch.cuda.get_device_properties(gpu_idx)
            gpu_info[gpu_idx] = {
                'name': gpu_properties.name,
                'major': gpu_properties.major,
                'minor': gpu_properties.minor,
                'total_mem': approximate_size(gpu_properties.total_memory),
                'multi_processor_count': gpu_properties.multi_processor_count
            }
            str_info = "GPU{}: {name} {major}.{minor} [{total_mem} total memory, {multi_processor_count} processors]"
            print(str_info.format(gpu_idx, **gpu_info[gpu_idx]))

        if return_info: return gpu_info
