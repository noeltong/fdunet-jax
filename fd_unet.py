import jax
from flax import linen as nn
import jax.numpy as jnp


class conv1x1(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, train=True):
        out = nn.Conv(self.out_channels, kernel_size=(1, 1), strides=1, padding=0)(x)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)

        return out


class conv3x3(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, train=True):
        out = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=1, padding=1)(x)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)

        return out


class BasicBlock(nn.Module):
    k: int
    F: int

    @nn.compact
    def __call__(self, x, train=True):
        out = conv3x3(out_channels=self.F)(x, train)
        out = conv1x1(out_channels=self.k)(out, train)

        return out


class DenseBlock(nn.Module):
    k: int
    F: int

    @nn.compact
    def __call__(self, x, train=True):
        k = self.k
        F = self.F

        h1 = x
        b1 = BasicBlock(k=k, F=F)(h1, train)

        h2 = jnp.concatenate([h1, b1], axis=-1)
        b2 = BasicBlock(k=k, F=F)(h2, train)

        h3 = jnp.concatenate([h1, b1, b2], axis=-1)
        b3 = BasicBlock(k=k, F=F)(h3, train)

        h4 = jnp.concatenate([h1, b1, b2, b3], axis=-1)
        b4 = BasicBlock(k=k, F=F)(h4, train)

        out = jnp.concatenate([h1, b1, b2, b3, b4], axis=-1)

        return out


class Downsample(nn.Module):
    @nn.compact
    def __call__(self, x):
        out = nn.max_pool(
            x, window_shape=(4, 4), strides=(2, 2), padding=((1, 1), (1, 1))
        )

        return out


class Upsample(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, s, train=True):
        out = nn.ConvTranspose(x.shape[-1] // 2, kernel_size=(2, 2), strides=(2, 2))(x)
        out = nn.BatchNorm(use_running_average=not train)(out)
        out = nn.relu(out)
        out = conv1x1(out_channels=self.out_channels)(
            jnp.concatenate([out, s], axis=-1), train
        )

        return out


class FDUNet(nn.Module):

    @nn.compact
    def __call__(self, x, train=True):
        x_in = x

        # Downsample
        h1 = conv3x3(32)(x_in, train)
        h1 = DenseBlock(F=64, k=8)(h1, train)

        h2 = Downsample()(h1)
        h2 = DenseBlock(F=128, k=16)(h2, train)

        h3 = Downsample()(h2)
        h3 = DenseBlock(F=256, k=32)(h3, train)

        h4 = Downsample()(h3)
        h4 = DenseBlock(F=512, k=64)(h4, train)

        # Bottleneck
        m = Downsample()(h4)
        m = DenseBlock(F=1024, k=128)(m, train)

        # Upsample
        u1 = Upsample(out_channels=512)(m, h4, train)
        u1 = DenseBlock(F=512, k=64)(u1, train)

        u2 = Upsample(out_channels=256)(u1, h3, train)
        u2 = DenseBlock(F=256, k=32)(u2, train)

        u3 = Upsample(out_channels=128)(u2, h2, train)
        u3 = DenseBlock(F=128, k=16)(u3, train)

        u4 = Upsample(out_channels=64)(u3, h1, train)
        u4 = DenseBlock(F=64, k=8)(u4, train)

        u5 = nn.Conv(1, kernel_size=(1, 1), strides=(1, 1), padding=0)(u4)

        out = x_in + u5

        return out
