import torch
import torch.nn as nn

from qd_metarl.utils.env_utils import RunningMeanStd
from qd_metarl.utils.torch_utils import DeviceConfig


class DictRunningMeanStd(object):
    def __init__(self, shapes, epsilon=1e-4):
        #print('DICT RUNNING MEAN INIT')
        self.rms = dict()
        for k, v in shapes.items():
            #print('init key', k, 'with shape', v)
            self.rms[k] = RunningMeanStd(epsilon=epsilon, shape=v)
    
    def __getitem__(self, key):
        #print('getting key', key, 'with shape (mean)', self.rms[key].mean.shape)
        return self.rms[key]
    
    def update(self, x, k):
        #print('key', k, 'updated with shape', x.shape)
        self.rms[k].update(x)


class MazeFeatureExtractor(nn.Module):
    """Used for extracting features for dict-based inputs"""

    def __init__(self, input_sizes, output_size, activation_function, relevant_keys=None):
        super(MazeFeatureExtractor, self).__init__()

        # Save relevant keys and sizes
        if relevant_keys is None:
            self.relevant_keys = list(input_sizes.keys())
        else:
            self.relevant_keys = relevant_keys
        self.relevant_keys_set = set(relevant_keys)
        self.nonimage_relevant_keys = [x for x in self.relevant_keys if x != 'image']
        self.input_sizes = input_sizes
        self.output_size = output_size
        self.activation_function = activation_function
        
        # Calculate input size
        input_size = 0
        for k in self.relevant_keys:
            if k == 'image':
                input_size += input_sizes[k][0] * input_sizes[k][1] * input_sizes[k][2]
            else:
                input_size += input_sizes[k][0]

        # Initialize fully connected layer
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        og_inputs = inputs

        # Get relevant inputs in order specified by self.relevant_keys and
        # flatten image dimensions (for now)
        relevant_inputs = []
        for k in self.relevant_keys:
            # TODO: This is very messy/hardcoded
            # print('k', k)
            # print('inputs', inputs)
            # print("inputs[k].shape", inputs[k].shape)
            if k == 'image':
                if isinstance(inputs, dict):
                    pass
                else:
                    print('inputs is not dict:', inputs)
                
                inputs_k = inputs[k]
                inputs_k_shape = inputs_k.shape
                num_dims = len(inputs_k_shape)
                # num_dims = len(inputs[k].shape)
                if num_dims > 3:
                    relevant_inputs.append(inputs[k].view(*inputs[k].shape[0:num_dims-3], -1))
                elif num_dims == 3:
                    relevant_inputs.append(inputs[k].view(-1))
                else:
                    raise NotImplementedError
            else:
                relevant_inputs.append(inputs[k])

        inputs = torch.cat(relevant_inputs, dim=-1)

        # Previously, inputs was of shape either (BS, N) or (N)

        # Pass through fully connected layer
        if self.output_size != 0:
            ret = self.activation_function(self.fc(inputs))
            return ret
        else:
            return torch.zeros(0, ).to(DeviceConfig.DEVICE)



