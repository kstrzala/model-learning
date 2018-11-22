import numpy as np
import tensorflow as tf
from maze import generate_maze
from models import FullModel
import random
import os
from PIL import Image
import neptune

tf.enable_eager_execution()


def shuffle_trajectory(trajectory, solutions):
    traj_sol_zip = list(zip(trajectory, solutions))
    random.shuffle(traj_sol_zip)
    random_trajectory, random_solutions = zip(*traj_sol_zip)
    return random_trajectory, random_solutions


def trajectory_to_tensor(maze, trajectory, solutions):
    tf_solutions = tf.convert_to_tensor(np.stack(solutions).astype(np.float32))
    tf_trajectory = tf.convert_to_tensor(maze.trajectory_to_numpy(trajectory))
    return tf_trajectory, tf_solutions


def run_pipeline(ctx):
    maze_size = ctx.params['maze_size']
    model = FullModel(maze_size)
    outer_optimizer = tf.train.AdamOptimizer()

    for outer_loop_count in range(ctx.params['outer_model_runs']):
        maze = generate_maze(maze_size, maze_size, ctx.params['maze_prob'])
        ctx.channel_send('Current Maze', neptune.Image(
            name=str(outer_loop_count),
            description='whatever',
            data=Image.fromarray(maze.get_maze_image(), "RGB")
            )
        )

        trajectory, solutions = maze.run_optimal_trajectory(form='coordinates')
        ctx.channel_send('Ideal trajectory length', len(trajectory))

        model.reset_inner_model()
        inner_optimizer = tf.train.AdamOptimizer()

        current_loss = 100
        loss_counter = 0
        inner_loop_count = 0
        while (loss_counter < ctx.params['loss_stabilization_steps']):
            random_trajectory, random_solutions = shuffle_trajectory(trajectory, solutions)
            tf_imgs, tf_solutions = trajectory_to_tensor(maze, random_trajectory, random_solutions)

            with tf.GradientTape() as tape:
                result = model(tf_imgs, inner_train=True)
                loss = tf.keras.losses.kld(tf_solutions, result)
            grad = tape.gradient(loss, model.inner_model.variables)
            inner_optimizer.apply_gradients(zip(grad, model.inner_model.variables))

            # Convergence condition: some steps without noticeably desceasing loss
            last_loss = current_loss
            current_loss = loss.numpy().mean()
            if ((last_loss - current_loss) < ctx.params['inner_eps']):
                loss_counter += 1
            else:
                loss_counter = 0
            inner_loop_count += 1
            ctx.channel_send("Inner loss", current_loss)

        ctx.channel_send("Inner loop count", x=outer_loop_count, y=inner_loop_count)

        if ctx.params['compare_with_random']:
            random_model = FullModel(maze_size)
            current_loss = 100
            loss_counter = 0
            inner_loop_count = 0
            random_inner_optimizer = tf.train.AdamOptimizer()
            while (loss_counter < ctx.params['loss_stabilization_steps']):
                random_trajectory, random_solutions = shuffle_trajectory(trajectory, solutions)
                tf_imgs, tf_solutions = trajectory_to_tensor(maze, random_trajectory, random_solutions)

                with tf.GradientTape() as tape:
                    result = random_model(tf_imgs, inner_train=True)
                    loss = tf.keras.losses.kld(tf_solutions, result)
                grad = tape.gradient(loss, random_model.inner_model.variables)
                random_inner_optimizer.apply_gradients(zip(grad, random_model.inner_model.variables))

                last_loss = current_loss
                current_loss = loss.numpy().mean()
                if ((last_loss - current_loss) < ctx.params['inner_eps']):
                    loss_counter += 1
                else:
                    loss_counter = 0
                inner_loop_count += 1

            ctx.channel_send("Inner loop count random", x=outer_loop_count, y=inner_loop_count)

        maze.reset()
        test_trajectory = maze.run_policy(model.policy)
        ctx.channel_send('Trajectory length', len(test_trajectory))

        random_trajectory, random_solutions = shuffle_trajectory(trajectory, solutions)
        tf_imgs, tf_solutions = trajectory_to_tensor(maze, random_trajectory, random_solutions)

        with tf.GradientTape() as tape:
            result = model(tf_imgs, outer_train=True)
            loss = tf.keras.losses.kld(tf_solutions, result)
        grad = tape.gradient(loss, model.outer_model.variables)
        outer_optimizer.apply_gradients(zip(grad, model.outer_model.variables))

        ctx.channel_send("Outer loss", loss.numpy().mean())

    root = tf.train.Checkpoint(model=model)
    root.save(os.path.join(ctx.params['model_filepath'], ctx.params['model_name']))
