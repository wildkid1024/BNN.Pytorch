import torch
import torch.nn as nn
import torch.nn.functional as F

def Binarize(tensor):
        E = tensor.abs().mean()        
        return tensor.sign() * E

class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        
        self.weight.data=Binarize(self.weight.org) 
        out = F.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out
    
class BWN(nn.Module): 
    def __init__(self, in_size, out_size, hidden_num=256):
        super(BWN, self).__init__()
        self.fc1 = BinarizeLinear(in_size, hidden_num)
        self.fc2 = BinarizeLinear(hidden_num, hidden_num)
        self.fc3 = BinarizeLinear(hidden_num, out_size)

    def forward(self, x):
        # x = x.view(1, -1);
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x 