import numpy
import torch
import math
from rich import print
tensor = [
    [[1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]],

    [[10., 11., 12.],
    [13., 14., 15.],
    [16., 17., 18.]],

    [[19., 20., 21.],
    [22., 23., 24.],
    [25., 26., 27.]]
    ]
tensor = [
    [[1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]],

    [[1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]],

    [[1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]]
    ]

data = torch.Tensor(tensor)

def lower_dimensionality(tensor: torch.Tensor):
    #2x2x2 filter
    tensor_size = tensor.size()
    kernel_size = 2
    #layers | top-down | sideways
    slices = []
    for layer in tensor:
        iterability = int(math.ceil(len(layer)/kernel_size))
        layer_kernel = []
        for i in range(iterability):
            for j in range(iterability):
                layer_kernel.append(
                                    [
                                    [layer[i][j], layer[i][j+1]],
                                    [layer[i+1][j], layer[i+1][j+1]]
                                    ]
                                    )
        slices.append(layer_kernel)
    #reconstruct kernel blocks
    blocks = []
    for layer_idx in range(len(slices)-1): #sub1 as we are looking at next layer for next slice
        for slice_idx in range(len(slices[0])):
            blocks.append(
                [
                    slices[layer_idx][slice_idx],
                    slices[layer_idx+1][slice_idx]
                ]
            )   
    averages = []
    for block in blocks:
        sums = 0
        item_count = 0
        for slice in block:
            for subslice in slice:
                for tensor in subslice:
                    sums += tensor.item()
                    item_count += 1
        averages.append(sums/item_count)
    output_tensor = []
    average_array_index = 0
    for x in range(tensor_size[0]-1):
        x_temp = []
        for y in range(tensor_size[1]-1):
            y_temp = []
            for z in range(tensor_size[2]-1):
                y_temp.append(torch.tensor(averages[average_array_index]))
                average_array_index += 1
            x_temp.append(y_temp)
        output_tensor.append(x_temp)
    return output_tensor

print(lower_dimensionality(data))
