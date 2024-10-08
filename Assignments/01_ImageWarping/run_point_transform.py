import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    #warped_image = np.array(image)
    warped_image = np.zeros(image.shape, dtype=np.uint8)
    ### FILL: 基于MLS or RBF 实现 image warping

    DELTA = 1000.0

    # rows = source_pts.shape[0]
    # coef_mat = np.zeros((rows,rows), dtype=np.float32)
    # for i in range(rows):
    #     for j in range(rows):
    #         coef_mat[i, j] = 1.0 / (np.sum((source_pts[i]-source_pts[j]) ** 2) + DELTA)
    
    distances_sq = np.sum((source_pts[:, np.newaxis] - source_pts[np.newaxis, :]) ** 2, axis=-1)
    coef_mat = 1.0 / (distances_sq + DELTA)
    
    RBF_coef = np.linalg.solve(coef_mat, target_pts - source_pts)

    j_coords, i_coords = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    points = np.flip(np.stack((j_coords.ravel(), i_coords.ravel()), axis=-1))
    distances_sq = np.sum((points[:, np.newaxis, :] - source_pts) ** 2, axis=-1)
    points = np.flip(points)
    point_coef = 1.0 / (distances_sq + DELTA)
    offset = np.flip(np.round(point_coef @ RBF_coef).astype(int))
    tar_pts = offset + points
    for pt1, pt2 in zip(points, tar_pts):
        if pt2[0] < image.shape[0] and pt2[0] >=0 and pt2[1] < image.shape[1] and pt2[1] >= 0:
            warped_image[pt2[0], pt2[1], :] = image[pt1[0], pt1[1], :]

    # print(warped_image.shape)
    # print(source_pts)
    # print(target_pts)

    # for j in range(image.shape[1]):
    #     for i in range(image.shape[0]):
    #         point_coef = 1.0 / (np.sum((np.array([j, i]) - source_pts) ** 2, axis = 1) + DELTA)
    #         offset = np.round(point_coef @ RBF_coef).astype(int)
    #         target_pt = np.array([offset[1], offset[0]]) + np.array([i, j])
    #         if target_pt[0] < image.shape[0] and target_pt[1] < image.shape[1] and target_pt[0] >= 0 and target_pt[1] >= 0:
    #             warped_image[target_pt[0], target_pt[1], :] = image[i, j, :]

    # 填空
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if warped_image[i, j].all() == 0:
                sum = np.zeros((1,3), dtype=np.uint16)
                cnt = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        pr = i + dr
                        pc = j + dc
                        if pr >= 0 and pr < image.shape[0] and pc >= 0 and pc < image.shape[1] and warped_image[pr, pc].any() != 0:
                            sum += warped_image[pr, pc]
                            cnt += 1
                if cnt > 0:
                    warped_image[i, j] = sum / cnt 


    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
