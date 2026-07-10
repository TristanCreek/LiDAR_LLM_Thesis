import os
import cv2
import time
import math
import random
import imutils
import shutil
import numpy as np
import tensorflow as tf
from skimage.util import random_noise

from Layers import GetPatches

def mask_image(img):
    gray = img
    if (len(gray.shape) == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.dilate(gray, kernel, iterations=1)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def sliding_window_mask(img, win_size=64):
    move_speed = win_size
    x1 = 0
    y1 = 0
    x2 = win_size
    y2 = win_size
    vert_steps = math.ceil((len(img)-win_size)/move_speed)
    hor_steps = math.ceil((len(img[0])-win_size)/move_speed)
    for y in range(vert_steps + 1):
        for x in range(hor_steps + 1):
            sub_img = img[y1:y2, x1:x2]
            masked = mask_image(sub_img)
            img[y1:y2, x1:x2] = masked
            x1 = x1 + move_speed
            x2 = x2 + move_speed
            if x2 > len(img[0]) - 1:
                x2 = len(img[0]) - 1
                x1 = x2 - win_size
        x1 = 0
        x2 = win_size
        y1 = y1 + move_speed
        y2 = y2 + move_speed
        if y2 > len(img) - 1:
            y2 = len(img) - 1
            y1 = y2 - win_size
    return img

def sliding_window(img, win_size):
    move_speed = 100
    y1 = 0
    y2 = win_size
    x1 = 0
    x2 = win_size
    images = []
    for y in range(math.ceil((len(img)-win_size)/move_speed)):
        for x in range(math.ceil((len(img[0])-win_size)/move_speed)):
            sub_img = img[y1:y2, x1:x2]
            images.append(sub_img)
            x1 = x1 + move_speed
            x2 = x2 + move_speed
            if x2 > len(img[0]) - 1:
                x2 = len(img[0]) - 1
                x1 = x2 - win_size
        x1 = 0
        x2 = win_size
        y1 = y1 + move_speed
        y2 = y2 + move_speed
        if y2 > len(img) - 1:
            y2 = len(img) - 1
            y1 = y2 - win_size
    return np.array(images)

def get_bbox(img):
    gray = img
    if (len(img.shape) == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    join_cnts = np.concatenate(contours)
    return cv2.boundingRect(join_cnts)

def get_corners(img):
    og_x, og_y, og_w, og_h = get_bbox(img)
    cropped = img[og_y:(og_y + og_h), og_x:(og_x + og_w)]

    split_multiplier = 4
    split_height = int(len(cropped) / split_multiplier)
    split_width = int(len(cropped[0]) / split_multiplier)
    bot_split = split_height * (split_multiplier - 1)
    right_split = split_width * (split_multiplier - 1)

    top_left = cropped[:split_height, :split_width]
    top_right = cropped[:split_height, right_split:]
    bot_left = cropped[bot_split:, :split_width]
    bot_right = cropped[bot_split:, right_split:]

    x, y, w, h = get_bbox(top_left)
    tl = [x, y]
    x, y, w, h = get_bbox(top_right)
    tr = [x + w + right_split, y]
    x, y, w, h = get_bbox(bot_left)
    bl = [x, y + h + bot_split]
    x, y, w, h = get_bbox(bot_right)
    br = [x + w + right_split, y + h + bot_split]
    cropped_coords = [tl, tr, br, bl]
    true_coords = np.array([[i + og_x, j + og_y] for [i, j] in cropped_coords], dtype=np.float32)
    return true_coords

def normalize_image(img):
    gray = img
    if (len(img.shape) == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    true_coords = get_corners(mask)
    desired_coords = np.array([[680, 252], [1256, 252], [1256, 828], [680, 828]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(true_coords, desired_coords)
    transformed_gray = cv2.warpPerspective(gray, M, (len(gray[0]), len(gray)))
    transformed_mask = cv2.warpPerspective(mask, M, (len(gray[0]), len(gray)))
    masked = cv2.bitwise_and(transformed_gray, transformed_mask)
    return masked[252:828, 680:1256]

def normalize_image_light(img):
    gray = img
    if (len(img.shape) == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    true_coords = np.array([[507, 0], [1340, 0], [1411, 1005], [520, 1080]], dtype=np.float32) # DELETE
    desired_coords = np.array([[680, 252], [1256, 252], [1256, 828], [680, 828]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(true_coords, desired_coords)
    transformed_gray = cv2.warpPerspective(gray, M, (len(gray[0]), len(gray)))

    cropped = transformed_gray[252:828, 680:1256] # DELETE
    cropcopy = np.copy(cropped)
    cropcopy = sliding_window_mask(cropcopy, 64) # DELETE
    masked = cv2.bitwise_and(cropped, cropcopy)
    return masked

def affine_transform(img, i, j, split):
    true_coords = np.array([[0, 0], [len(img[0]) - 1, 0], [len(img[0]) - 1, len(img) - 1], [0, len(img) - 1]], dtype=np.float32)
    tl = [i * split, j * split]
    tr = true_coords[1]
    br = true_coords[2]
    bl = [i * split, (j + 1) * split]
    desired_coords = np.array([tl, tr, br, bl], dtype=np.float32)
    M = cv2.getPerspectiveTransform(true_coords, desired_coords)
    left_dst = cv2.warpPerspective(img, M, (len(img[0]), len(img)))
    #left_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    tl = [j * split, i * split]
    tr = [(j + 1) * split, i * split]
    br = true_coords[2]
    bl = true_coords[3]
    desired_coords = np.array([tl, tr, br, bl], dtype=np.float32)
    M = cv2.getPerspectiveTransform(true_coords, desired_coords)
    top_dst = cv2.warpPerspective(img, M, (len(img[0]), len(img)))
    #top_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    return left_dst, top_dst

def affine_transforms(img, num_transforms):
    # LiDAR pattern is only symmetrically diagonally
    # Make 90 degree rotate copy to cover all permutations
    transformed_imgs = []
    rot_img = imutils.rotate(img, 90)
    transformed_imgs.append(img)
    transformed_imgs.append(rot_img)

    # Transform num_transforms different ways
    if num_transforms > 1:
        split = int(len(img) / np.sqrt(num_transforms))
        for i in range(int(np.sqrt(num_transforms))):
            for j in range(int(np.sqrt(num_transforms))):
                left_dst, top_dst = affine_transform(img, i, j, split)
                transformed_imgs.append(left_dst)
                transformed_imgs.append(top_dst)
    return transformed_imgs

def get_patches(imgs, img_size):
    imgs = np.expand_dims(imgs, -1)
    patches4 = GetPatches(num_patches=4)(imgs)
    patches4 = tf.reshape(patches4, [-1, patches4.shape[-3], patches4.shape[-2], 1])
    patches16 = GetPatches(num_patches=4)(patches4)
    patches16 = tf.reshape(patches16, [-1, patches16.shape[-3], patches16.shape[-2], 1])
    imgs = np.squeeze(imgs, axis=-1)

    patches = []
    for patch in patches4.numpy():
        res = cv2.resize(patch, (img_size, img_size))
        patches.append(res)
    for patch in patches16.numpy():
        res = cv2.resize(patch, (img_size, img_size))
        patches.append(res)
    patches = np.array(patches)
    patches = np.concatenate((imgs, patches), axis=0)
    return patches

def validate_patches(patches):
    valid_patches = []
    for patch in patches:
        ret, mask = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_dots = len(contours)
        if num_dots > 16:
            join_cnts = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(join_cnts)
            bbox_area = w * h
            total_area = len(patch) * len(patch[0])
            coverage = bbox_area / total_area
            if coverage > 0.50:
                valid_patches.append(patch)

    return np.array(valid_patches)

def permute(img, path, num_transforms, num_rotations, win_size, img_size):
    transformed_imgs = affine_transforms(img, num_transforms)
    patches = get_patches(transformed_imgs, img_size)
    valid_patches = validate_patches(patches)

    write_count = 0
    for patch in valid_patches:
        rotations = []
        for r in range(num_rotations):
            rotated = imutils.rotate(patch, random.randint(0, 360))
            rotations.append(rotated)
        for pic in rotations:
            noisy = random_noise(pic, mode='gaussian', var=0.05 ** 2)
            noisy = (255 * noisy).astype(np.uint8)
            mask = sliding_window_mask(noisy, win_size)
            cv2.imwrite(path + 'lidar\\' + str(write_count) + '.png', mask)
            write_count += 1

def create_train_dataset(path, num_transforms=9, num_rotations=3, win_size=64, img_size=512):
    train_data_path = path + "\\train\\"
    # Creates folders recursively if they do not exist:
    # path\train\
    # path\train\lidar\
    # path\train\neg\
    os.makedirs(train_data_path + 'lidar', exist_ok=True)
    #os.makedirs(train_data_path + 'neg', exist_ok=True)
    dst = path+'\\train\\neg'
    try:
        shutil.rmtree(dst)
    except OSError as e:
        pass
    if 'light' in path:
        shutil.copytree(path + '\\test\\masked_pics\\light\\neg', dst)
    else:
        shutil.copytree(path + '\\test\\masked_pics\\dark\\neg', dst)

    pic = path + '\\light_pattern.png'
    img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    if 'light' in path:
        normalized = normalize_image_light(img)
    else:
        normalized = normalize_image(img)
    normalized = cv2.resize(normalized, (img_size, img_size))
    permute(normalized, train_data_path, num_transforms, num_rotations, win_size, img_size)

def create_pic_directories(video_path):
    # Split up video path by slashes
    split_path = video_path.split('\\')
    # Get video name from end of split list and remove .avi extension
    vid_name = split_path[-1].split('.')[0]
    # Remove top folder and join to get path to parent directory
    parent_path = ('\\').join(split_path[:-2])
    # Remove '_neg' from file name so we direct to the same pic folders as the lidar video
    if '_neg' in vid_name:
        vid_name = vid_name.replace('_neg', '')
    # Create path of folder in raw_pics that will contain pics for this video
    raw_pics_path = parent_path + '\\raw_pics\\' + vid_name + '\\'
    masked_pics_path = parent_path + '\\masked_pics\\' + vid_name + '\\'

    # Creates folders recursively if they do not exist:
    # path\test\raw_pics
    # path\test\raw_pics\<vid_name>
    # path\test\raw_pics\<vid_name>\lidar
    # path\test\raw_pics\<vid_name>\neg
    os.makedirs(raw_pics_path + 'lidar', exist_ok=True)
    os.makedirs(raw_pics_path + 'neg', exist_ok=True)

    # We do not use these immediately in the caller function
    # but it is most convenient to create the folders now
    # Creates folders recursively if they do not exist:
    # path\test\masked_pics
    # path\test\masked_pics\<vid_name>
    # path\test\masked_pics\<vid_name>\lidar
    # path\test\masked_pics\<vid_name>\neg
    os.makedirs(masked_pics_path + 'lidar', exist_ok=True)
    os.makedirs(masked_pics_path + 'neg', exist_ok=True)

    return raw_pics_path

def vid_to_frames(video_path, img_size):
    # Create necessary folders
    # Returns path to created lidar folder
    raw_pics_path = create_pic_directories(video_path)
    lidar_path = raw_pics_path + 'lidar'
    neg_path = raw_pics_path + 'neg'


    # Get video name from video_path
    split_path = video_path.split('\\')
    # Remove .avi extension from video name
    vid_name = split_path[-1].split('.')[0]

    img_count = 0
    cap = cv2.VideoCapture(video_path)
    ret, og_img = cap.read()

    root = video_path.split('\\')[0]
    pattern_path = root + '\\light_pattern.png'
    if not os.path.isfile(pattern_path) and 'neg' not in vid_name:
        cv2.imwrite(pattern_path, og_img)

    while og_img is not None:
        # Get full path to store a single frame it is respective \raw_pics\<video>\lidar folder
        frame_path = os.path.join(lidar_path, vid_name + "_" + str(img_count) + ".png")

        # Store videos with 'neg' in the title to the respective neg folder instead of the lidar folder
        #\raw_pics\<video>\neg
        if 'neg' in vid_name:
            frame_path = os.path.join(neg_path, vid_name + "_" + str(img_count) + ".png")

        # Extract middle square that contains majority of lidar light points
        crop = og_img[65:1015, 500:1450]
        res = cv2.resize(crop, (img_size, img_size))
        cv2.imwrite(frame_path, res)
        img_count += 1
        ret, og_img = cap.read()

def all_vids_to_frames(path, img_size):
    vid_path = path + '\\videos'
    videos = []
    for file in os.listdir(vid_path):
        if '.avi' in file:
            videos.append(os.path.join(vid_path, file))
    for video in videos:
        vid_to_frames(video, img_size)

def convert_raw_pics_to_masked_pics(dataset_path, win_size):
    raw_pic_paths = []
    for subdir, dirs, files in os.walk(dataset_path + '\\raw_pics'):
        for file in files:
            raw_pic_paths.append(os.path.join(subdir, file))
    for raw_pic_path in raw_pic_paths:
        raw_pic = cv2.imread(raw_pic_path, cv2.IMREAD_GRAYSCALE)
        masked_img = sliding_window_mask(raw_pic, win_size)
        masked_pic_path = raw_pic_path.replace('raw_pics', 'masked_pics')
        cv2.imwrite(masked_pic_path, masked_img)

def create_test_dataset(dataset_path, win_size=64, img_size=512):
    test_data_path = dataset_path + "\\test"
    if not os.path.isdir(test_data_path + '\\raw_pics'):
        # Create folder for each video
        # Save all frames as images to respective folder
        all_vids_to_frames(test_data_path, img_size)
    # Read all the images above in raw_pics
    # Mask them and save them to <dataset_path>\masked_pics\
    convert_raw_pics_to_masked_pics(test_data_path, win_size)

def create_test_and_train_data(path, win_size, num_transforms, num_rotations, img_size):
    # In same directory as script, create file structure
    # <path>\
    # <path>\light_pattern.png (LiDAR light pattern image to generate training images)
    # <path>\test\
    # <path>\test\videos (containing test videos)
    create_test_dataset(path, win_size=win_size, img_size=img_size)
    create_train_dataset(path, num_transforms=num_transforms, num_rotations=num_rotations, win_size=win_size, img_size=img_size)

if __name__ == '__main__':
    start_time = time.time()
    create_test_and_train_data('path', win_size=64, num_transforms=16, num_rotations=3, img_size=512)
    print("%f seconds" % (time.time() - start_time))


