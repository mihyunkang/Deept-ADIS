from torch import distributed



def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False
