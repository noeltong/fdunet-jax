import jax
import jax.numpy as jnp
import os
from scipy.io import loadmat
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import logging
from flax.metrics import tensorboard
from fd_unet import FDUNet
import optax
from flax.training.train_state import TrainState
from flax.training import checkpoints
import functools
import flax
import numpy as np
from absl import app, flags
from ml_collections.config_flags import config_flags
import warnings
import sys

sys.path.append("/root/PA-SGM")
from physics_model.PAT_sparse import load_measurement, get_transforms_fn
from utils.solvers import get_masks
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
    mean_squared_error,
)
from PIL import Image, ImageDraw
import time
from typing import Any


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory to store files.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.mark_flags_as_required(["mode", "config", "workdir"])


class BNTrainState(TrainState):
    batch_stats: Any


def train(config, workdir):
    """Runs the training of UNet

    Args:
        config: a ml_collections.ConfigDict containing configs
        workdir: dirrectory containing useful files
    """

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s: %(message)s"
    )

    fh = logging.FileHandler(os.path.join(workdir, "train_log.log"), encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Num of local devices: {jax.local_device_count()}")
    logger.info(
        f"Training with mask {config.sampling.mask} with {config.sampling.n_signals} known signals. Work files restored in {workdir}."
    )

    # ------------------------------------------
    # Logging with tensorboard
    # ------------------------------------------

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)

    if jax.process_index() == 0:
        writer = tensorboard.SummaryWriter(tb_dir)

    # ------------------------------------------
    # Initialize the model
    # ------------------------------------------

    logger.info("Initializing UNet model...")

    model = FDUNet()

    rng = jax.random.PRNGKey(config.seed)
    rng, step_rng = jax.random.split(rng)

    # input_shape = (jax.local_device_count(), config.data.len_signal,
    #                config.data.num_sensors, config.data.num_channels)
    input_shape = (
        jax.local_device_count(),
        config.data.image_size,
        config.data.image_size,
        config.data.num_channels,
    )
    fake_input = jax.random.normal(rng, input_shape)
    params = model.init(step_rng, fake_input, train=False)

    rng, step_rng = jax.random.split(rng)

    logger.info("Initialization complete.")

    # ------------------------------------------
    # Create directory for checkpoints
    # ------------------------------------------

    ckpt_dir = os.path.join(workdir, "ckpt")
    ckpt_meta_dir = os.path.join(workdir, "ckpt-meta")
    tf.io.gfile.makedirs(ckpt_dir)
    tf.io.gfile.makedirs(ckpt_meta_dir)

    # ------------------------------------------
    # Defining the optimizer and training state
    # ------------------------------------------

    lr = optax.cosine_decay_schedule(
        init_value=config.optim.initial_lr,
        decay_steps=config.training.n_iters,
        alpha=1e-3 * config.optim.initial_lr,
    )

    if config.optim.optimizer.lower() == "adamw":
        optimizer = optax.adamw(learning_rate=lr)

    optimizer.init(params)

    state = BNTrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        batch_stats=params["batch_stats"],
        tx=optimizer,
    )

    # ------------------------------------------
    # Resume training
    # ------------------------------------------
    state = checkpoints.restore_checkpoint(ckpt_meta_dir, state)
    initial_step = int(state.step)

    # rng = state.rng
    rng, step_rng = jax.random.split(rng)

    # ----------------
    # Loading data
    # ----------------

    logger.info("Loading training data...")

    train_loc = glob(os.path.join(config.data.location, "train", "*.npz"))
    test_loc = glob(os.path.join(config.data.location, "test", "*.npz"))

    def get_dataset(path_list):
        features, gts = [], []
        for path in tqdm(path_list):
            idx = np.load(path)
            features.append(idx["sinogram"])
            gts.append(idx["gt"][..., np.newaxis])

        dataset = tf.data.Dataset.from_tensor_slices((features, gts))
        batch_size = config.training.batch_size

        if batch_size % jax.device_count() != 0:
            raise ValueError(
                f"Batch sizes ({batch_size} must be divided by"
                f"the number of devices ({jax.device_count()})"
            )

        per_device_batch_size = batch_size // jax.device_count()

        # batch_dims = [jax.local_device_count(), per_device_batch_size]
        batch_dims = [jax.local_device_count() * per_device_batch_size]

        dataset = dataset.repeat(count=None)
        dataset = dataset.shuffle(1000)

        for bs in reversed(batch_dims):
            dataset = dataset.batch(bs, drop_remainder=True)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.as_numpy_iterator()

        return dataset

    train_iter = get_dataset(train_loc)
    test_iter = get_dataset(test_loc)

    logger.info("Complete.")

    # ------------------
    # Define train step
    # ------------------

    @jax.jit
    def L2Loss(x, y):
        diff = x - y
        return jnp.mean(diff**2)

    @jax.jit
    def train_step_fn(state, x, y):
        """Execute one training step"""

        def loss_fn(params):
            logits, updates = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                x,
                train=True,
                mutable=["batch_stats"],
            )
            loss = L2Loss(logits, y)

            return loss, updates

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, updates), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates["batch_stats"])

        return state, loss

    @jax.jit
    def eval_step_fn(state, x, y):
        logits = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats}, x, train=False
        )
        loss = L2Loss(logits, y)

        return logits, loss

    # ------------------
    # Load transforms
    # ------------------

    logger.info("Loading transformation matrix...")

    transform_mat = load_measurement()
    from_space, to_space = get_transforms_fn(config, transform_mat)

    def mask_image(sinogram):
        mask = get_masks(config)
        known = sinogram * mask
        img_masked = from_space(known)

        return img_masked

    logger.info("Complete.")

    # -------------------------------------
    # preparation for multidevice parallel
    # -------------------------------------

    p_train_step = jax.pmap(train_step_fn, axis_name="device", donate_argnums=1)
    p_eval_step = jax.pmap(eval_step_fn, axis_name="device", donate_argnums=1)

    pstate = flax.jax_utils.replicate(state)
    num_train_steps = config.training.n_iters
    logger.info(f"Starting training loop at step {initial_step}.")
    rng = jax.random.fold_in(step_rng, jax.process_index())

    @jax.jit
    def normalize(img):
        """Scale intensities to [0, 1]"""
        img = img - img.min()
        img = img / img.max()

        return img

    # ------------------
    # start training
    # ------------------

    for step in range(initial_step, num_train_steps + 1):
        x, y = next(train_iter)
        x, y = jax.device_put(x), jax.device_put(y)
        x = mask_image(x)

        x = x.reshape(
            [
                jax.local_device_count(),
                -1,
                config.data.image_size,
                config.data.image_size,
                1,
            ]
        )
        y = y.reshape(
            [
                jax.local_device_count(),
                -1,
                config.data.image_size,
                config.data.image_size,
                1,
            ]
        )

        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)

        pstate, ploss = p_train_step(pstate, x, y)
        # ploss = jax.lax.pmean(ploss, axis_name='device')
        loss = flax.jax_utils.unreplicate(ploss).item()

        if step % config.training.log_freq == 0:
            logger.info(f"Step: {step}, Training loss: {loss:.4f}")
            writer.scalar("Training loss", loss, step)

        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            logger.info("Saving snapshots...")
            saved_state = flax.jax_utils.unreplicate(pstate)
            # saved_state = saved_state.replace(rng=rng)
            checkpoints.save_checkpoint(
                ckpt_meta_dir,
                saved_state,
                step=step // config.training.snapshot_freq_for_preemption,
                keep=1,
            )

        # ----------------------------------------------
        # Report the loss on eval dataset periodically
        # ----------------------------------------------

        if step != 0 and step % config.training.eval_freq == 0:
            logger.info("Starting evaluation on test dataset...")
            eval_losses = []
            eval_ssims = []
            eval_psnrs = []

            for i_batch in range(config.training.num_eval):
                x, y = next(test_iter)
                x, y = jax.device_put(x), jax.device_put(y)
                x = mask_image(x)

                x = x.reshape(
                    [
                        jax.local_device_count(),
                        -1,
                        config.data.image_size,
                        config.data.image_size,
                        1,
                    ]
                )
                y = y.reshape(
                    [
                        jax.local_device_count(),
                        -1,
                        config.data.image_size,
                        config.data.image_size,
                        1,
                    ]
                )

                rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
                next_rng = jnp.asarray(next_rng)
                plogits, peval_loss = p_eval_step(pstate, x, y)
                # peval_loss = jax.lax.pmean(peval_loss, axis_name='device')
                eval_loss = flax.jax_utils.unreplicate(peval_loss).item()
                eval_losses.append(eval_loss)

                s = tf.image.ssim(normalize(plogits), normalize(y), max_val=1.0)
                p = tf.image.psnr(normalize(plogits), normalize(y), max_val=1.0)

                eval_ssims.extend([*s])
                eval_psnrs.extend([*p])

                if jax.process_index() == 0:
                    logger.info(
                        f"Eval step: {i_batch}, Eval loss: {eval_loss:.4f}, SSIM: {np.mean(s):.4f}, PSNR: {np.mean(p):.4f}"
                    )

            if jax.process_index() == 0:
                avg_eval_loss = np.mean(eval_losses)
                logger.info(
                    f"Test average on {config.training.num_eval} samples: {avg_eval_loss:.4f}"
                )
                writer.scalar("Eval loss", avg_eval_loss, step)
                writer.scalar("Eval SSIM", np.mean(eval_ssims), step)
                writer.scalar("Eval PSNR", np.mean(eval_psnrs), step)

            # ------------------------------------------
            # Saving checkpoints
            # ------------------------------------------

            if (
                step != 0
                and step % config.training.snapshot_freq == 0
                or step == num_train_steps
            ):
                logger.info("Saving checkpoints...")

                if jax.process_index() == 0:
                    saved_state = flax.jax_utils.unreplicate(pstate)
                    # saved_state = saved_state.replace(rng=rng)
                    checkpoints.save_checkpoint(
                        ckpt_dir,
                        saved_state,
                        step=step // config.training.snapshot_freq,
                        keep=np.inf,
                    )


def evaluate(config, workdir):
    """Evaluate trained UNet.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """

    # jax.config.update('jax_platform_name', 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # ---------------------------
    # Generate the logger
    # ---------------------------

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s: %(message)s"
    )

    # fh = logging.FileHandler(os.path.join(workdir, "eval_log.log"), encoding="utf-8")
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    logger.info(f"Num of local devices: {jax.local_device_count()}")
    logger.info(
        f"Evaluating name: {workdir}, mask: {config.sampling.mask}, number of signals: {config.sampling.n_signals}."
    )

    # ---------------------------
    # Creating work dir
    # ---------------------------

    eval_dir = os.path.join(
        "workspace/fdunet",
        "eval",
        str(config.sampling.n_signals),
        "fdunet",
        config.sampling.mask,
    )
    tf.io.gfile.makedirs(eval_dir)

    rng = jax.random.PRNGKey(config.seed + 1)
    rng = jax.random.fold_in(rng, jax.process_index())

    # ---------------------------
    # Build data pipeline
    # ---------------------------

    test_loc = glob(os.path.join(config.data.location, "test", "*.npz"))

    def load_test_data(test_loc):
        test_data = []

        def data_scaler(img):
            img = img - img.min()
            img = img / img.max()

            return img

        for path in tqdm(test_loc):
            idx = np.load(path)
            test_data.append([idx["sinogram"][np.newaxis, ...], idx["gt"]])
            # [input, gt]

        return test_data

    logger.info("Loading test data...")
    test_data = load_test_data(test_loc)
    logger.info("Complete.")

    # ------------------------------------------
    # Initialize the model
    # ------------------------------------------

    logger.info("Initializing UNet model...")

    model = FDUNet()

    rng = jax.random.PRNGKey(config.seed)
    rng, step_rng = jax.random.split(rng)

    input_shape = (
        jax.local_device_count(),
        config.data.image_size,
        config.data.image_size,
        config.data.num_channels,
    )
    fake_input = jax.random.normal(rng, input_shape)
    params = model.init(step_rng, fake_input)

    rng, step_rng = jax.random.split(rng)

    logger.info("Initialization complete.")

    # ------------------------------------------
    # Create directory for checkpoints
    # ------------------------------------------

    ckpt_dir = os.path.join(workdir, "ckpt")
    ckpt_meta_dir = os.path.join(workdir, "ckpt-meta")
    tf.io.gfile.makedirs(ckpt_dir)
    tf.io.gfile.makedirs(ckpt_meta_dir)

    # ------------------------------------------
    # Defining the optimizer and training state
    # ------------------------------------------

    lr = optax.cosine_decay_schedule(
        init_value=config.optim.initial_lr,
        decay_steps=config.training.n_iters,
        alpha=1e-3 * config.optim.initial_lr,
    )

    if config.optim.optimizer.lower() == "adamw":
        optimizer = optax.adamw(learning_rate=lr)

    optimizer.init(params)

    state = BNTrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        batch_stats=params["batch_stats"],
        tx=optimizer,
    )

    # ckpt_dir = os.path.join(workdir, 'ckpt')
    ckpt_dir = os.path.join("workspace/fdunet/32/uniform_mask", "ckpt")

    logger.info("Completed.")

    # --------------------------
    # Loading the transformation
    # --------------------------

    logger.info("Loading transformation matrix...")

    transform_mat = load_measurement()
    from_space, to_space = get_transforms_fn(config, transform_mat)

    logger.info("Completed.")

    # --------------------------
    # Restoring checkpoints
    # --------------------------

    logger.info("Restoring checkpoints...")

    state = checkpoints.restore_checkpoint(ckpt_dir, state, step=config.eval.ckpt_id)

    logger.info("Complete.")

    # -----------------------------
    # defining eval metrics
    # -----------------------------

    def mask_image(sinogram):
        mask = get_masks(config)
        known = sinogram * mask
        img_masked = from_space(known)

        return img_masked

    def get_central_mask():
        img_size = config.data.image_size
        mask = Image.new("L", (img_size, img_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.pieslice([0, 0, img_size, img_size], 0, 360, fill=255)
        mask = np.array(mask)[..., np.newaxis]

        return mask / 255.0

    def get_metrics(prediction, target, mask_roi=False, hist_norm=False):
        if hist_norm:
            pred_hist = jnp.histogram(prediction, bins=255)[0]
            targ_hist = jnp.histogram(target, bins=255)[0]

            peak_pred1 = jnp.argmax(pred_hist[:75]) / 255
            peak_pred2 = (jnp.argmax(pred_hist[75:]) + 75) / 255
            peak_targ1 = jnp.argmax(targ_hist[:75]) / 255
            peak_targ2 = (jnp.argmax(targ_hist[75:]) + 75) / 255

            prediction = jnp.clip(
                (prediction - peak_pred1) / (peak_pred2 - peak_pred1), a_min=0
            )
            target = jnp.clip(
                (target - peak_targ1) / (peak_targ2 - peak_targ1), a_min=0
            )

            prediction = jnp.clip(prediction, a_max=target.max(), a_min=0)
            prediction /= target.max()
            target /= target.max()

        if mask_roi:
            mask = get_central_mask()
            prediction = prediction * mask
            target = target * mask

        target = jax.device_get(target)
        prediction = jax.device_get(prediction)

        psnr = peak_signal_noise_ratio(target, prediction, data_range=1.0)
        ssim = structural_similarity(target, prediction, data_range=1.0)
        rmse = np.sqrt(mean_squared_error(target, prediction))

        return psnr, ssim, rmse

    # -----------------------------
    # Start calculation
    # -----------------------------

    logger.info("Start evaluation...")

    all_samples = []
    all_gts = []
    all_ssims = []
    all_psnrs = []
    all_rmses = []
    all_ssims_mask = []
    all_psnrs_mask = []
    all_rmses_mask = []
    all_ssims_hist = []
    all_psnrs_hist = []
    all_rmses_hist = []
    all_ssims_mask_hist = []
    all_psnrs_mask_hist = []
    all_rmses_mask_hist = []

    times = []
    for sg, gt in tqdm(test_data):
        sg = jax.device_put(sg)
        gt = jax.device_put(gt)
        img = mask_image(sg)
        tic = time.time()
        pred = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats}, img, train=False
        ).squeeze()
        toc = time.time()
        times.append(toc - tic)

        # pred = jnp.clip(pred, a_min=0., a_max=1.)
        # gt = jnp.clip(gt, a_min=0., a_max=1.)

        pred = (pred - pred.min()) / (pred.max() - pred.min())
        gt = (gt - gt.min()) / (gt.max() - gt.min())

        all_samples.append(pred)
        all_gts.append(gt)

        p, s, r = get_metrics(pred, gt)
        all_psnrs += [p]
        all_ssims += [s]
        all_rmses += [r]

        # logger.info(f'PSNR: {p.mean():.4f}, SSIM: {s.mean():.4f}, RMSE: {r.mean():.4f}')

        p, s, r = get_metrics(pred, gt, mask_roi=True)
        all_psnrs_mask += [p]
        all_ssims_mask += [s]
        all_rmses_mask += [r]

        # logger.info(
        #     f'With mask: PSNR: {p.mean():.4f}, SSIM: {s.mean():.4f}, RMSE: {s.mean():.4f}')

        p, s, r = get_metrics(pred, gt, hist_norm=True)
        all_psnrs_hist += [p]
        all_ssims_hist += [s]
        all_rmses_hist += [r]

        # logger.info(
        #     f'With hist: PSNR: {p.mean():.4f}, SSIM: {s.mean():.4f}, RMSE: {s.mean():.4f}')

        p, s, r = get_metrics(pred, gt, mask_roi=True, hist_norm=True)
        all_psnrs_mask_hist += [p]
        all_ssims_mask_hist += [s]
        all_rmses_mask_hist += [r]

        # logger.info(
        #     f'With mask & hist: PSNR: {p.mean():.4f}, SSIM: {s.mean():.4f}, RMSE: {s.mean():.4f}')

    all_samples = np.stack(all_samples, axis=0).astype(np.float32)
    all_gts = np.stack(all_gts, axis=0).astype(np.float32)
    np.savez_compressed(
        os.path.join(eval_dir, "recon.npz"), prediction=all_samples, groundtruth=all_gts
    )

    all_psnrs = np.asarray(all_psnrs)
    all_ssims = np.asarray(all_ssims)
    all_rmses = np.asarray(all_rmses)
    all_psnrs_mask = np.asarray(all_psnrs_mask)
    all_ssims_mask = np.asarray(all_ssims_mask)
    all_rmses_mask = np.asarray(all_rmses_mask)
    all_psnrs_hist = np.asarray(all_psnrs_hist)
    all_ssims_hist = np.asarray(all_ssims_hist)
    all_rmses_hist = np.asarray(all_rmses_hist)
    all_psnrs_mask_hist = np.asarray(all_psnrs_mask_hist)
    all_ssims_mask_hist = np.asarray(all_ssims_mask_hist)
    all_rmses_mask_hist = np.asarray(all_rmses_mask_hist)

    logger.info("All samples evaluated.")
    logger.info(
        f"PSNR: {all_psnrs.mean():.4f}, SSIM: {all_ssims.mean():.4f}, RMSE: {all_rmses.mean():.4f}"
    )
    logger.info(
        f"With mask: PSNR: {all_psnrs_mask.mean():.4f}, SSIM: {all_ssims_mask.mean():.4f}, RMSE: {all_rmses_mask.mean():.4f}"
    )
    logger.info(
        f"With hist: PSNR: {all_psnrs_hist.mean():.4f}, SSIM: {all_ssims_hist.mean():.4f}, RMSE: {all_rmses_hist.mean():.4f}"
    )
    logger.info(
        f"With mask & hist: PSNR: {all_psnrs_mask_hist.mean():.4f}, SSIM: {all_ssims_mask_hist.mean():.4f}, RMSE: {all_rmses_mask_hist.mean():.4f}"
    )
    logger.info(f"Average process time: {np.asarray(times).mean():.6f} per image.")

    logger.info(f"Saving results to {eval_dir}.")

    np.savez_compressed(
        os.path.join(eval_dir, "metrics.npz"),
        psnrs=all_psnrs,
        ssims=all_ssims,
        rmses=all_rmses,
        psnrs_mask=all_psnrs_mask,
        ssims_mask=all_ssims_mask,
        rmses_mask=all_rmses_mask,
        psnrs_hist=all_psnrs_hist,
        ssims_hist=all_ssims_hist,
        rmses_hist=all_rmses_hist,
        psnrs_mask_hist=all_psnrs_mask_hist,
        ssims_mask_hist=all_ssims_mask_hist,
        rmses_mask_hist=all_rmses_mask_hist,
    )

    logger.info("Completed.")


def main(argv):
    warnings.filterwarnings("ignore")

    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    config = FLAGS.config

    work_dir = os.path.join(
        "workspace",
        FLAGS.workdir,
        str(config.sampling.n_signals),
        config.sampling.mask,
    )

    os.makedirs(work_dir, exist_ok=True)

    if FLAGS.mode == "train":
        train(config=config, workdir=work_dir)
    elif FLAGS.mode == "eval":
        evaluate(config=config, workdir=work_dir)


def autorun(argv):
    warnings.filterwarnings("ignore")

    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    config = FLAGS.config

    for i_signals in [16, 24, 32, 40, 48, 56, 64]:
        for i_task in ["uniform_mask", "random_mask", "limited_view"]:
            config.sampling.n_signals = i_signals
            config.sampling.mask = i_task

            work_dir = os.path.join(
                "workspace",
                FLAGS.workdir,
                str(config.sampling.n_signals),
                config.sampling.mask,
            )

            if FLAGS.mode == "train":
                os.makedirs(work_dir, exist_ok=True)
                train(config=config, workdir=work_dir)
            elif FLAGS.mode == "eval":
                evaluate(config=config, workdir=work_dir)


if __name__ == "__main__":
    # app.run(main)
    app.run(autorun)
