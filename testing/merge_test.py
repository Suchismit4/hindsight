from src.data_layer.tensor import Tensor
from src.data_layer.data_manager import DataLoader

dl = DataLoader()

t1 = Tensor()

t2 = Tensor()

t3 = dl.merge_tensors(t1, t2)

