import numpy
import torch
import math
tensor = [
    [[1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]],

    [[2., 25., 2.],
    [2., 26., 2.],
    [2., 2., 2.]],

    [[3., 3., 3.],
    [3., 3., 3.],
    [3., 33., 3.]]
    ]

data = torch.Tensor(tensor)

def lower_dimensionality(tensor: torch.Tensor):
    #2x2x2 filter
    tensor_size = tensor.size()
    kernel_size = 2
    #layers | top-down | sideways
    twodim_kernel = []
    for layer in tensor:
        layer_kernel = []
        for i in range(int(math.ceil(len(layer)/kernel_size))):
            print(i)
            layer_kernel.append(
                                [
                                [layer[i][i], layer[i][i+1]],
                                [layer[i+1][i], layer[i+1][i+1]]
                                ]
                                )
            layer_kernel.append(
                                [
                                [layer[i+1][i], layer[i+1][i+1]],
                                [layer[i+1+1][i], layer[i+1+1][i+1]]
                                ]
                                )
        twodim_kernel.append(layer_kernel)
            

    convoluted_tensor = tensor[:, :, :]
    return convoluted_tensor

lower_dimensionality(data)
