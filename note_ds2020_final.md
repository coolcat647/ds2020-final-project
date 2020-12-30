
## Compile & install mmdetection3d tools
```bash
sudo pip3 install -v -e .
```

## Create dataset for mmdet3d (this commamd is for v1.0-mini)
python3 tools/create_data.py nuscenes --root-path ~/nuscenes --out-dir ~/nuscenes --extra-tag nuscenes --max-sweeps 2 --version v1.0-mini

