from ml_collections import ConfigDict


def get_config():

    config = ConfigDict()

    config.seed = 1234

    config.training = training = ConfigDict()
    training.batch_size = 16
    training.n_iters = 10000
    training.log_freq = 25
    training.snapshot_freq_for_preemption = 500
    training.eval_freq = 1000
    training.snapshot_freq = 1000
    training.num_eval = 50

    config.sampling = sampling = ConfigDict()
    sampling.mask = 'uniform_mask'
    sampling.n_signals = 32
    
    config.data = data = ConfigDict()
    data.image_size = 128
    data.num_channels = 1
    data.location = '/root/data/mice/npy'
    data.len_signal = 1000
    data.num_sensors = 128

    config.eval = evaluate = ConfigDict()
    evaluate.ckpt_id = 10

    config.model = model = ConfigDict()
    model.dim = 64

    config.optim = optim = ConfigDict()
    optim.optimizer = 'AdamW'
    optim.initial_lr = 2.5e-4

    return config