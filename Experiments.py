import os
import time
import shutil
import pathlib
import numpy as np

from Preprocess_Footage import create_test_and_train_data
from Train_Evaluate_NN import train_and_save_model, load_and_evaluate_model

# Delete everything in the environment except for
# <env>\test\raw_pics
# <env>\test\videos
def reset(env_path):
    model_path = pathlib.Path(env_path + '\\model')
    train_path = pathlib.Path(env_path + '\\train')
    paths = [model_path, train_path]
    for p in paths:
        if p.exists() and p.is_dir():
            shutil.rmtree(p)

    mp_path = env_path + '\\test\\masked_pics'
    if os.path.isdir(mp_path):
        for subdir, dirs, files in os.walk(mp_path):
            for file in files:
               os.remove(os.path.join(subdir, file))

def window_sizes_32_vs_64(env_path):
    win_sizes = np.array([32, 64]*10)
    np.random.shuffle(win_sizes)
    experiment_count = 0
    experiment_results = []
    write_results = []
    accuracy_32 = 'accuracy_32 = ['
    accuracy_64 = 'accuracy_64 = ['
    time_32 = 'time_32 = ['
    time_64 = 'time_64 = ['
    for win_size in win_sizes:
        reset(env_path)
        experiment_time = time.time()
        img_size = 512
        num_rotations = 3
        create_test_and_train_data(env_path, win_size=win_size, num_transforms=16, num_rotations=num_rotations, img_size=img_size)
        train_and_save_model(env_path, img_size)
        results = load_and_evaluate_model(env_path, img_size)

        elapsed_time = '{:.9f}'.format(time.time() - experiment_time)
        header = 'Window Size = {} completed in {} seconds\n'.format(win_size, elapsed_time)
        write_results.append(header)
        toks = results[0].split(' ')
        accuracy = toks[-1]
        write_results.append(accuracy + '\n')
        write_results.append('\n')

        # Format accuracy and time nicely for use in Statistic_Analysis.py
        if win_size==32:
            accuracy_32 = accuracy_32 + accuracy + ', '
            time_32 = time_32 + elapsed_time + ', '
        elif win_size==64:
            accuracy_64 = accuracy_64 + accuracy + ', '
            time_64 = time_64 + elapsed_time + ', '
        else:
            print("ENCOUNTERED UNEXPECTED WINDOW SIZE {}".format(win_size))
            exit()

        experiment_count += 1
        print('COMPLETED EXPERIMENT #{}'.format(experiment_count))
        print(header)

    write_results.append(accuracy_32[:-2] + ']\n')
    write_results.append(accuracy_64[:-2] + ']\n')
    write_results.append(time_32[:-2] + ']\n')
    write_results.append(time_64[:-2] + ']\n')

    for line in experiment_results:
        print(line)

    with open('results_winsize_32_vs_64.txt', 'w') as fd:
        fd.writelines(write_results)

def rotation_num_3_vs_9(env_path):
    rot_nums = np.array([3, 9]*10)
    np.random.shuffle(rot_nums)

    experiment_count = 0
    experiment_results = []
    write_results = []
    accuracy_3 = 'accuracy_3 = ['
    accuracy_9 = 'accuracy_9 = ['
    time_3 = 'time_3 = ['
    time_9 = 'time_9 = ['
    for rot_num in rot_nums:
        reset(env_path)
        experiment_time = time.time()
        img_size = 512
        win_size = 64
        create_test_and_train_data(env_path, win_size=win_size, num_transforms=16, num_rotations=rot_num, img_size=img_size)
        train_and_save_model(env_path, img_size)
        results = load_and_evaluate_model(env_path, img_size)

        elapsed_time = '{:.9f}'.format(time.time() - experiment_time)
        header = 'Rotation Number = {} completed in {} seconds\n'.format(rot_num, elapsed_time)
        write_results.append(header)
        toks = results[0].split(' ')
        accuracy = toks[-1]
        write_results.append(accuracy + '\n')
        write_results.append('\n')

        # Format accuracy and time nicely for use in Statistic_Analysis.py
        if rot_num==3:
            accuracy_3 = accuracy_3 + accuracy + ', '
            time_3 = time_3 + elapsed_time + ', '
        elif rot_num==9:
            accuracy_9 = accuracy_9 + accuracy + ', '
            time_9 = time_9 + elapsed_time + ', '
        else:
            print("ENCOUNTERED UNEXPECTED ROTATION NUMBER {}".format(win_size))
            exit()

        experiment_count += 1
        print('COMPLETED EXPERIMENT #{}'.format(experiment_count))
        print(header)

    write_results.append(accuracy_3[:-2] + ']\n')
    write_results.append(accuracy_9[:-2] + ']\n')
    write_results.append(time_3[:-2] + ']\n')
    write_results.append(time_9[:-2] + ']\n')

    for line in experiment_results:
        print(line)

    with open('results_rotnum_3_vs_9.txt', 'w') as fd:
        fd.writelines(write_results)

def dark_vs_light(env_path, light_env_path):
    light_levels = np.array([env_path, light_env_path] * 1)
    np.random.shuffle(light_levels)

    experiment_count = 0
    experiment_results = []
    write_results = []
    accuracy_dark = 'accuracy_dark = ['
    accuracy_light = 'accuracy_light = ['
    time_dark = 'time_dark = ['
    time_light = 'time_light = ['
    for light_level in light_levels:
        reset(light_level)
        experiment_time = time.time()
        img_size = 512
        win_size = 64
        rot_num = 3
        create_test_and_train_data(light_level, win_size=win_size, num_transforms=16, num_rotations=rot_num, img_size=img_size)
        train_and_save_model(light_level, img_size)
        results = load_and_evaluate_model(light_level, img_size)

        elapsed_time = '{:.9f}'.format(time.time() - experiment_time)
        header = 'Light Level Path = {} completed in {} seconds\n'.format(light_level, elapsed_time)
        write_results.append(header)
        toks = results[0].split(' ')
        accuracy = toks[-1]
        write_results.append(accuracy + '\n')
        write_results.append('\n')

        # Format accuracy and time nicely for use in Statistic_Analysis.py
        if 'light' in light_level:
            accuracy_light = accuracy_light + accuracy + ', '
            time_light = time_light + elapsed_time + ', '
        else:
            accuracy_dark = accuracy_dark + accuracy + ', '
            time_dark = time_dark + elapsed_time + ', '

        experiment_count += 1
        print('COMPLETED EXPERIMENT #{}'.format(experiment_count))
        print(header)

    write_results.append(accuracy_dark[:-2] + ']\n')
    write_results.append(accuracy_light[:-2] + ']\n')
    write_results.append(time_dark[:-2] + ']\n')
    write_results.append(time_light[:-2] + ']\n')

    for line in experiment_results:
        print(line)

    with open('results_lightlevel_dark_vs_light.txt', 'w') as fd:
        fd.writelines(write_results)

if __name__ == '__main__':
    start_time = time.time()
    window_sizes_32_vs_64('path')
    rotation_num_3_vs_9('path')
    dark_vs_light('path', 'path_light')
    print("All experiments completed in {:.9f} seconds".format(time.time() - start_time))