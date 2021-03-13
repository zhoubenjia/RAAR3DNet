# Training the RAAR3DNet.

## Step 1:
**In order to visualize the Loss/Acc curve or feature map more clearly, we must first run visdom:**
```bash
 python -m visdom.server
```
## Step 2:
**Training the network on the IsoGD dataset:**
```bash
 bash run.sh config/IsoGD.yml  4
```

**Some training strategies:**
- For training the RAAR3D network, it is recommended to train the NI3D network first, and then use the NI3D-trained model as the pre-training model for training the RAAR3D network.
