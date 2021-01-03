<!-- 
## Compile & install mmdetection3d tools
```bash
sudo pip3 install -v -e .
```
 -->

<!-- 
## Create dataset for mmdet3d (this commamd is for v1.0-mini)
python3 tools/create_data.py nuscenes --root-path ~/nuscenes --out-dir ~/nuscenes --extra-tag nuscenes --max-sweeps 2 --version v1.0-mini -->
# Prerequisite

## 1. Download nusc_kitti

### 1-1 Day and night dataset split 
Dataset split | Daytime     | Night     | 
--------------|:-----:      |-----:     | 
Train         | [link](https://drive.google.com/file/d/1PQszzW-U7rh1w3W67z6SfYDscAJnmYlt/view?usp=sharing)    |  [link](https://drive.google.com/file/d/1Vs283PRRsGBtQkTN9UvmXqSwgNDcijSh/view?usp=sharing) |   
Validation    | [link](https://drive.google.com/file/d/1UUE-viXM_60bbHQZC2TiImp0argVlKbi/view?usp=sharing)    |  [link](https://drive.google.com/file/d/1-ipnV8bXw0ApDMV_YLlw2vo0sdSTtKFR/view?usp=sharing) |
(optional) All-in-one dataset | [link](https://drive.google.com/file/d/1LFHgWlDKIFSqOnDsEE5cJ3w2IQGsJQEH/view?usp=sharing) 


### 1-2 Rename the dataset folder by postfix removing
* ex: val_daytime/ &ensp;-->&ensp; val/
* ex: train_night/ &ensp;&ensp;-->&ensp; train/



## 2. Configure docker_run.sh
```bash
# At around line 45
nuscenes_root="[WHERE_YOU_PUT_NUSC_KITTI]"
```

## 3. Docker run
```bash
# arg: cuda10   --> launch a new container
# arg: same     --> enter the same container
source docker_run.sh [cuda10 | same]
```

## 4. Create nusc_kitti soft link for M3D-RPN
```bash
cd M3D-RPN/data
ln -s /home/developer/nuscenes/nusc_kitti kitti
```

## 5. Build M3D-RPN library
```bash
cd M3D-RPN/lib/nms
make

# Build kitti benchmark tool
cd M3D-RPN
source data/kitti_split1/devkit/cpp/build.sh
```

<!-- ## Export Nuscenes to KITTI format directory (28130 samples totally for training)
```bash
# In the project root
python3 export_kitti.py nuscenes_gt_to_kitti --nusc_version v1.0-trainval --nusc_kitti_dir /home/developer/nuscenes/nusc_kitti --split train --image_count 70000
``` -->

<!-- Export Nuscenes to KITTI format directory (6019 samples totally for validation)
```bash
# In the project root
python3 export_kitti.py nuscenes_gt_to_kitti --nusc_version v1.0-trainval --nusc_kitti_dir /home/developer/nuscenes/nusc_kitti --split val --image_count 15000
``` -->

    
---
# Train M3D-RPN
## 1. Start logging server
```bash
cd M3D-RPN
python3 -m visdom.server -port 8100 -readonly
```

## 2. Train with warmup mode (New terminal)
```bash
# New terminal in the M3D-RPN directory
python3 scripts/train_rpn_3d.py --config=kitti_3d_multi_warmup
```

## 3. Train with depth-aware mode after finishing warmup
```bash
python3 scripts/train_rpn_3d.py --config=kitti_3d_multi_main
```

---



# Visualization detection result
```bash
sudo apt-get install python3-tk
```

