from time import sleep
import numpy as np
from pipeline import run_pipeline
import neptune
import os


ctx = neptune.Context()


def neptune_init():
    if ctx.params['cuda_visible_devices'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(ctx.params['cuda_visible_devices'])
    ctx.create_channel(name="Inner loss", channel_type = neptune.ChannelType.NUMERIC)
    ctx.create_channel(name="Outer loss", channel_type=neptune.ChannelType.NUMERIC)


def main():
    x = 0
    for step in range(1000):
        ctx.channel_send("X", x)
        ctx.channel_send("sin X", np.sin(x))
        x += 0.01

        sleep(0.01)

if __name__ == "__main__":
    neptune_init()
    run_pipeline(ctx)