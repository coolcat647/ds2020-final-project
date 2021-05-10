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
from importlib import import_module


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0



def sample_night_image(night_image_loader, night_image_set, step, n_batches):
    if step % n_batches == 0:
        night_image_set = iter(night_image_loader)
    return night_image_set, night_image_set.next()


def main(argv):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Night scene dataset and iterator
    night_dataset = SimpleImageDataset("/home/developer/nuscenes/nusc_kitti/train_night/image_2/", conf)
    night_image_loader = torch.utils.data.DataLoader(dataset=night_dataset,
                                                    batch_size=conf["batch_size"],
                                                    shuffle=True,
                                                    drop_last=True)
    n_batches = len(night_dataset) // conf["batch_size"]
    night_image_set = iter(night_image_loader)

    # for step in range(10):
    #     n_critic = 1 # for training more k steps about domain_net    
    #     night_image_set, tgt_images = sample_night_image(night_image_loader, night_image_set, step, n_batches)
    #     print(tgt_images.shape)
    #     grid = torchvision.utils.make_grid(tgt_images)
    #     npgrid = grid.cpu().numpy()
    #     plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    #     plt.pause(0.5)


    generate_anchors(conf, dataset.imdb, paths.output)
    compute_bbox_stats(conf, dataset.imdb, paths.output)


    # -----------------------------------------
    # store config
    # -----------------------------------------

    # store configuration
    pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)

    # show configuration
    pretty = pretty_print('conf', conf)
    logging.info(pretty)


    # -----------------------------------------
    # network and loss
    # -----------------------------------------

    # training network
    feature_net, f_opt, detection_net, det_opt, domain_net, domain_opt = my_init_training_model(conf, paths.output)

    # setup loss
    criterion_det = RPN_3D_loss(conf)
    # bce = nn.MSELoss()
    bce = nn.BCELoss()  

    # custom pretrained network
    if 'pretrained' in conf:
        load_weights(feature_net, conf.pretrained1)
        load_weights(detection_net, conf.pretrained2)
        load_weights(domain_net, conf.pretrained3)

    # resume training
    if restore:
        start_iter = (restore - 1)
        my_resume_checkpoint(f_opt, feature_net, det_opt, detection_net, domain_opt, domain_net, paths.weights, restore)

    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist
    freeze_layers(feature_net, freeze_blacklist, freeze_whitelist)
    freeze_layers(detection_net, freeze_blacklist, freeze_whitelist)

    f_opt.zero_grad()
    det_opt.zero_grad()

    start_time = time()

    # fake_images = torch.randn(1, 3, 512, 1760).cuda()
    # cls, prob, bbox_2d, bbox_3d, feat_size = feature_net(fake_images)

    # domain_net = domain_net().to(DEVICE)
    # domain_opt = torch.optim.Adam(domain_net.parameters(), lr=0.01)
    D_src = torch.ones(conf["batch_size"], 1).to(DEVICE) # domain_net Label to real
    D_tgt = torch.zeros(conf["batch_size"], 1).to(DEVICE) # domain_net Label to fake
    D_labels = torch.cat([D_src, D_tgt], dim=0)
    # bce = nn.BCELoss()
    # bce = nn.MSELoss()
    
    # domain_net.train()

    # -----------------------------------------
    # train
    # -----------------------------------------

    for iteration in range(start_iter, conf.max_iter):

        # next iteration
        iterator, src_images, imobjs = next_iteration(dataset.loader, iterator)

        ###
        night_image_set, tgt_images = sample_night_image(night_image_loader, night_image_set, iteration, n_batches)
        tgt_images = tgt_images.to(DEVICE)

        #  learning rate
        adjust_lr(conf, f_opt, iteration)
        adjust_lr(conf, det_opt, iteration)

        # forward
        # base_output = feature_net(src_images)
        # cls, prob, bbox_2d, bbox_3d, feat_size = detection_net(base_output)

        # # loss
        # det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)
        # total_loss = det_loss
        # stats = det_stats
        x = torch.cat([src_images, tgt_images], dim=0)
        base_output = feature_net(x)
        pred_discrimination = domain_net(base_output.detach())
        Ld = bce(pred_discrimination, D_labels)
        domain_net.zero_grad()
        Ld.backward()
        domain_opt.step()

        cls, prob, bbox_2d, bbox_3d, feat_size = detection_net(base_output[:conf["batch_size"]])
        pred_discrimination = domain_net(base_output)

        # loss
        Lc, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)
        Ld = bce(pred_discrimination, D_labels)
        lamda = 0.1 * get_lambda(iteration, conf.max_iter)

        
        total_loss = Lc - lamda * Ld
        stats = det_stats
        print("iter: {}/{}, lambda: {:.3f}, Lc: {:.2f}, Ld: {:.2f}, Ltotal:{:.2f}".format(iteration, conf.max_iter, lamda, Lc, Ld, total_loss))
        if math.isinf(total_loss) or math.isnan(total_loss):
            exit(-1)
        
        # backprop
        if total_loss > 0:
            

            # domain_net.zero_grad()
            total_loss.backward()

            # batch skip, simulates larger batches by skipping gradient step
            if (not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0:
                f_opt.step()
                f_opt.zero_grad()
                det_opt.step()
                det_opt.zero_grad()
        else:
            total_loss.backward()
        
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
            my_save_checkpoint(f_opt, feature_net, det_opt, detection_net, domain_opt, domain_net, paths.weights, (iteration + 1))

            if conf.do_test:

                # eval mode
                feature_net.eval()
                detection_net.eval()
                domain_net.eval()

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
                detection_net.train()
                domain_net.train()

                freeze_layers(feature_net, freeze_blacklist, freeze_whitelist)
                freeze_layers(detection_net, freeze_blacklist, freeze_whitelist)


if __name__ == "__main__":
    main(sys.argv[1:])