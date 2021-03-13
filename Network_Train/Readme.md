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
