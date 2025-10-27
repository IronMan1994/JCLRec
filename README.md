# JCLRec

### YooChoose Data

```
python -u JCLRec.py --data yc --batch_size 256 --timesteps 500 --lr 0.001 --beta_sche exp --w 2 --optimizer adamw --diffuser_type mlp1 --random_seed 100 --CL_Loss MSE_Loss --lambda1 1.0 --lambda2 1.0 --cuda 0 --threshold 100 --item_crop 0.6 --item_mask 0.3 --item_reorder 0.6
```


### KuaiRec Data

```
python -u JCLRec.py --data ks --batch_size 256 --timesteps 2000 --lr 0.00005 --beta_sche cosine --w 2 --optimizer adamw --diffuser_type mlp1 --random_seed 100 --CL_Loss MSE_Loss --lambda1 1.0 --lambda2 1.0 --cuda 0 --threshold 20 --item_crop 0.6 --item_mask 0.3 --item_reorder 0.6 
```


### Zhihu Data

```
python -u JCLRec.py --data zhihu --batch_size 256 --timesteps 200 --lr 0.01 --beta_sche linear --w 4 --optimizer adamw --diffuser_type mlp1 --random_seed 100 --CL_Loss MSE_Loss --lambda1 1.0 --lambda2 1.0 --cuda 0 --threshold 20 --item_crop 0.6 --item_mask 0.3 --item_reorder 0.6
```
