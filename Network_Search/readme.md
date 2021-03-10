# Reconstruction of the I3D network based on NAS.

## Step 1:
**In order to visualize the Loss/Acc curve or feature map more clearly, we must first run visdom:**
```bash
 python -m visdom.server
```
## Step 2:
**Searching and reorganizing the network structure on the IsoGD dataset:**
```bash
 bash run_search.sh config/IsoGD.yml  4
```
