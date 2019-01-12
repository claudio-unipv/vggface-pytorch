# Script used to convert another pytorch model into the vggface.
# Assume that the original model worked with images in the 0-255
# range, while the new one works in the 0-1 range (as most pytorch
# models).

import sys
import torch
import vggface


orig_name = sys.argv[1]
result_name = sys.argv[2]

data = torch.load(orig_name)
net = vggface.VggFace()
first = True
with torch.no_grad():
    for x, p in zip(data.values(), net.parameters()):
        if first:
            x = x * 255.0
        first = False
        p.copy_(x)

torch.save(net.state_dict(), result_name)
