# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import numpy as np
import sys
import os

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *


class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=1024, num_classes=1):
        super(Discriminator, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(16)

        self.prop_feats = nn.Sequential(
            nn.Conv2d(input_size, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.layer = nn.Sequential(
            nn.Linear(64*16*16, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, h):
        h = self.prop_feats(h)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        y = self.layer(h)
        return y


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.


def main(argv):

    # -----------------------------------------
    # parse arguments
    # -----------------------------------------
    opts, args = getopt(argv, '', ['config=', 'restore='])

    # defaults
    conf_name = None
    restore = None

    # read opts
    for opt, arg in opts:

        if opt in ('--config'): conf_name = arg
        if opt in ('--restore'): restore = int(arg)

    # required opt
    if conf_name is None:
        raise ValueError('Please provide a configuration file name, e.g., --config=<config_name>')

    # -----------------------------------------
    # basic setup
    # -----------------------------------------

    conf = init_config(conf_name)
    paths = init_training_paths(conf_name)

    init_torch(conf.rng_seed, conf.cuda_seed)
    init_log_file(paths.logs)

    vis = init_visdom(conf_name, conf.visdom_port)

    # defaults
    start_iter = 0
    tracker = edict()
    iterator = None
    has_visdom = vis is not None

    dataset = Dataset(conf, paths.data, paths.output)

    generate_anchors(conf, dataset.imdb, paths.output)
    compute_bbox_stats(conf, dataset.imdb, paths.output)


    # -----------------------------------------
    # store config
    # -----------------------------------------

    # store configuration
    # pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)

    # show configuration
    pretty = pretty_print('conf', conf)
    logging.info(pretty)


    # -----------------------------------------
    # network and loss
    # -----------------------------------------

    # training network
    feature_net, f_optimizer, detection_net, d_optimizer = my_init_training_model(conf, paths.output)

    # setup loss
    criterion_det = RPN_3D_loss(conf)

    # custom pretrained network
    if 'pretrained' in conf:
        load_weights(feature_net, conf.pretrained1)
        load_weights(detection_net, conf.pretrained2)

    # resume training
    if restore:
        start_iter = (restore - 1)
        resume_checkpoint(f_optimizer, feature_net, paths.weights, restore)

    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist
    freeze_layers(feature_net, freeze_blacklist, freeze_whitelist)
    freeze_layers(detection_net, freeze_blacklist, freeze_whitelist)

    f_optimizer.zero_grad()
    d_optimizer.zero_grad()

    start_time = time()

    # fake_images = torch.randn(1, 3, 512, 1760).cuda()
    # cls, prob, bbox_2d, bbox_3d, feat_size = feature_net(fake_images)

    discriminator = Discriminator().cuda()
    # discriminator(base_out)

    # iterator, images, imobjs = next_iteration(dataset.loader, iterator)
    # print(images.shape)

    # -----------------------------------------
    # train
    # -----------------------------------------

    for iteration in range(start_iter, conf.max_iter):

        # next iteration
        iterator, images, imobjs = next_iteration(dataset.loader, iterator)

        #  learning rate
        adjust_lr(conf, f_optimizer, iteration)
        adjust_lr(conf, d_optimizer, iteration)

        # forward
        base_output = feature_net(images)
        cls, prob, bbox_2d, bbox_3d, feat_size = detection_net(base_output)

        # loss
        det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)

        total_loss = det_loss
        stats = det_stats

        # backprop
        if total_loss > 0:

            total_loss.backward()

            # batch skip, simulates larger batches by skipping gradient step
            if (not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0:
                f_optimizer.step()
                f_optimizer.zero_grad()
                d_optimizer.step()
                d_optimizer.zero_grad()

        # keep track of stats
        compute_stats(tracker, stats)

        # -----------------------------------------
        # display
        # -----------------------------------------
        if (iteration + 1) % conf.display == 0 and iteration > start_iter:

            # log results
            log_stats(tracker, iteration, start_time, start_iter, conf.max_iter)

            # display results
            if has_visdom:
                display_stats(vis, tracker, iteration, start_time, start_iter, conf.max_iter, conf_name, pretty)

            # reset tracker
            tracker = edict()

        # -----------------------------------------
        # test network
        # -----------------------------------------
        if (iteration + 1) % conf.snapshot_iter == 0 and iteration > start_iter:

            # store checkpoint
            my_save_checkpoint(f_optimizer, feature_net, d_optimizer, detection_net, paths.weights, (iteration + 1))

            if conf.do_test:

                # eval mode
                feature_net.eval()
                detection_net.eval()

                # necessary paths
                results_path = os.path.join(paths.results, 'results_{}'.format((iteration + 1)))

                # -----------------------------------------
                # test kitti
                # -----------------------------------------
                if conf.test_protocol.lower() == 'kitti':

                    # delete and re-make
                    results_path = os.path.join(results_path, 'data')
                    mkdir_if_missing(results_path, delete_if_exist=True)

                    my_test_kitti_3d(conf.dataset_test, feature_net, detection_net, conf, results_path, paths.data)

                else:
                    logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))

                # train mode
                feature_net.train()

                freeze_layers(feature_net, freeze_blacklist, freeze_whitelist)
                freeze_layers(detection_net, freeze_blacklist, freeze_whitelist)


if __name__ == "__main__":
    main(sys.argv[1:])