import cv2
import numpy as np
import time
import os
import tqdm
import psutil
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


start_memory = psutil.virtual_memory()
start_time = time.time()

def StegerPlus(image, threshold=200, filter_size=(3, 3)):
    gray_origin = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray_origin, filter_size, 0, 0)
    Ix = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    Iy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    Ixx = cv2.Scharr(Ix, cv2.CV_32F, 1, 0)
    Ixy = cv2.Scharr(Ix, cv2.CV_32F, 0, 1)
    Iyy = cv2.Scharr(Iy, cv2.CV_32F, 0, 1)
    Iyx = cv2.Scharr(Iy, cv2.CV_32F, 1, 0)

    row = gray_origin.shape[0]
    col = gray_origin.shape[1]
    CenterPoint = []
    newimage = np.zeros((row, col), np.uint8)

    for i in range(col):
        for j in range(row):
            if gray_origin[j, i] > threshold:
                hessian = np.zeros((2, 2), np.float32)
                hessian[0, 0] = Ixx[j, i]
                hessian[0, 1] = Ixy[j, i]
                hessian[1, 0] = Iyx[j, i]
                hessian[1, 1] = Iyy[j, i]
                ret, eigenVal, eigenVec = cv2.eigen(hessian)

                lambda1 = 0.
                lambda2 = 0.
                nx, ny, fmaxD = 0.0, 0.0, 0.0
                if ret:
                    if np.abs(eigenVal[0, 0]) >= np.abs(eigenVal[1, 0]):
                        lambda1 = eigenVal[1, 0]
                        lambda2 = eigenVal[0, 0]
                        nx = eigenVec[0, 0]
                        ny = eigenVec[0, 1]
                        famxD = eigenVal[0, 0]
                    else:
                        lambda1 = eigenVal[0, 0]
                        lambda2 = eigenVal[1, 0]
                        nx = eigenVec[1, 0]
                        ny = eigenVec[1, 1]
                        famxD = eigenVal[1, 0]
                    if lambda1 < 15 and lambda2 < -50:
                        t = -(nx * Ix[j, i] + ny * Iy[j, i]) / (
                                nx * nx * Ixx[j, i] + 2 * nx * ny * Ixy[j, i] + ny * ny * Iyy[j, i])
                        if np.abs(t * nx) <= 0.5 and np.abs(t * ny) <= 0.5:
                            CenterPoint.append([i, j])

    for point in CenterPoint:
        newimage[point[1], point[0]] = 255
        image[point[1], point[0], :] = (0, 0, 255)

    return image, newimage

if __name__ == '__main__':
    image_folder_path = "path"
    save_folder_path = "path"
    
    if not os.path.isdir(save_folder_path):
        os.makedirs(save_folder_path)

    sum_time = 0
    response_times = []  
    memory_usage = []  

    for img_name in tqdm.tqdm(os.listdir(image_folder_path)):
        image_path = os.path.join(image_folder_path, img_name)
        
        image = cv2.imread(image_path)
        start_time = time.time()
        image_c, line = StegerPlus(image, threshold=200, filter_size=(3, 3))
        end_time = time.time()
        response_time = end_time - start_time
        response_times.append(response_time)
        sum_time += end_time - start_time
        
        save_image_path = os.path.join(save_folder_path, img_name)
        cv2.imwrite(save_image_path, image_c)
        
        line_image_path = os.path.join(save_folder_path, img_name.split('.')[0] + "_line.png")
        cv2.imwrite(line_image_path, line)


        current_memory = psutil.virtual_memory()
        memory_usage.append(current_memory.used)


    end_memory = psutil.virtual_memory()
    end_time = time.time()


    memory_used = end_memory.used - start_memory.used
    memory_used_mb = memory_used / 1024 / 1024  # 转换为MB


    avg_memory_usage = memory_used_mb / (end_time - start_time)


    smoothed_memory_usage = gaussian_filter1d(memory_usage, sigma=3)

    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB/s")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(response_times) + 1), response_times, marker='o', linestyle='-')
    plt.title('Image Processing Time')
    plt.xlabel('Image Number')
    plt.ylabel('Processing Time (seconds)')
    plt.tight_layout()
    plt.show()  

    average_time = sum_time / len(os.listdir(image_folder_path))
    print("Average one image time: ", average_time)


image_files = os.listdir(image_folder_path)