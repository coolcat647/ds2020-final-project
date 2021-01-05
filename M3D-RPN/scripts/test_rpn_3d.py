# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import os

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

conf_path = 'output/kitti_3d_multi_warmup/conf.pkl'
weights_path = 'output/kitti_3d_multi_main/weights/model_20000_pkl'

# load config
conf = edict(pickle_read(conf_path))
conf.pretrained = None

data_path = os.path.join(os.getcwd(), 'data')
results_path = os.path.join('output', 'tmp_results', 'data')

# make directory
mkdir_if_missing(results_path, delete_if_exist=True)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# defaults
init_torch(conf.rng_seed, conf.cuda_seed)

# -----------------------------------------
# setup network
# -----------------------------------------

# net
# net = import_module('models.' + conf.model).build(conf)
network1, network2 = import_module('models.' + conf.model).my_build(conf)

# load weights
# load_weights(net, weights_path, remove_module=True)
load_weights(network1, "/home/developer/samliu/ds2020-final-project/M3D-RPN/output/kitti_3d_multi_warmup/weights/model_20000_pkl_feature", remove_module=True)
load_weights(network2, "/home/developer/samliu/ds2020-final-project/M3D-RPN/output/kitti_3d_multi_warmup/weights/model_20000_pkl_detection", remove_module=True)

# switch modes for evaluation
# net.eval()
network1.eval()
network2.eval()

print(pretty_print('conf', conf))

# -----------------------------------------
# test kitti
# -----------------------------------------

# test_kitti_3d(conf.dataset_test, net, conf, results_path, data_path, use_log=False)
my_test_kitti_3d(conf.dataset_test, network1, network2, conf, results_path, data_path, use_log=False)