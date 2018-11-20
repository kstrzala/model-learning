import numpy as np
import tensorflow as tf
from maze import Maze
from models import FullModel
import os
import random
from PIL import Image
import neptune

tf.enable_eager_execution()

def generate_maze(height=5, width=5):
    """
    Generate valid maze (i.e. nontrivial maze with existing solution) of a given size.
    :return: valid maze
    """
    while True:
        maze = Maze(height, width)
        if maze.valid():
            trajectory = maze.run_optimal_trajectory(form='coordinates', probs=False)
            if len(trajectory)>1:
                maze.reset()
                return maze


def run_pipeline(ctx):
    model = FullModel()
    outer_optimizer = tf.train.AdamOptimizer()

    for outer_loop_count in range(ctx.params['outer_model_runs']):
        maze = generate_maze()
        ctx.channel_send('Current Maze', neptune.Image(
            name=str(outer_loop_count),
            description='whatever',
            data=Image.fromarray(maze.get_maze_image(), "RGB")
        )
                         )

        #get ideal trajectory and solutions for a given maze
        trajectory, solutions = maze.run_optimal_trajectory(form='coordinates')
        ctx.channel_send('Ideal trajectory length', len(trajectory))

        #Inner model initialization
        model.reset_inner_model()
        inner_optimizer = tf.train.AdamOptimizer()

        current_loss = 100
        loss_counter = 0
        inner_loop_count = 0
        while (loss_counter < 10):
            #Get shuffled trajectory & solutions
            traj_sol_zip = list(zip(trajectory, solutions))
            random.shuffle(traj_sol_zip)
            random_trajectory, random_solutions = zip(*traj_sol_zip)
            random_solutions = np.stack(random_solutions).astype(np.float32)
            imgs = tf.convert_to_tensor(maze.trajectory_to_numpy(random_trajectory))

            #Update inner model params
            with tf.GradientTape() as tape:
                result = model(imgs, inner_train=True)
                loss = tf.keras.losses.kld(tf.convert_to_tensor(random_solutions), result)
            grad = tape.gradient(loss, model.inner_model.variables)
            inner_optimizer.apply_gradients(zip(grad, model.inner_model.variables))

            # Convergence condition: 10 steps without noticeably desceasing loss
            last_loss = current_loss
            current_loss = loss.numpy().mean()
            if ((last_loss - current_loss) < ctx.params['inner_eps']):
                loss_counter += 1
            else:
                loss_counter = 0
            inner_loop_count += 1
            ctx.channel_send("Inner loss", current_loss)

        #Test current algorithm
        maze.reset()
        test_trajectory = maze.run_policy(model.policy)
        ctx.channel_send('Trajectory length', len(test_trajectory))
        ctx.channel_send("Inner loop count", inner_loop_count)

        traj_sol_zip = list(zip(trajectory, solutions))
        random.shuffle(traj_sol_zip)
        random_trajectory, random_solutions = zip(*traj_sol_zip)
        random_solutions = np.stack(random_solutions).astype(np.float32)
        imgs = tf.convert_to_tensor(maze.trajectory_to_numpy(random_trajectory))

        #Update outer model params
        with tf.GradientTape() as tape:
            result = model(imgs, outer_train=True)
            loss = tf.keras.losses.kld(tf.convert_to_tensor(random_solutions), result)
        grad = tape.gradient(loss, model.outer_model.variables)
        outer_optimizer.apply_gradients(zip(grad, model.outer_model.variables))

        ctx.channel_send("Outer loss", loss.numpy().mean())

    #root = tf.train.Checkpoint(model=model)
    #root.save(os.path.join(ctx.params['model_filepath'], 'model_params'))
