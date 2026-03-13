import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# # Point-guided image deformation
# def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
#     """
#     Return
#     ------
#         A deformed image.
#     """

#     warped_image = np.array(image)
#     ### FILL: Implement MLS or RBF based image warping

#     return warped_image

def point_guided_deformation(image,target_pts, source_pts,  alpha=1.0, eps=1e-8):
    
    h, w = image.shape[:2]
    warped_image = np.zeros_like(image)

    # Convert points to float arrays
    source_pts = np.array(source_pts, dtype=np.float32)
    target_pts = np.array(target_pts, dtype=np.float32)

    # Step 1: compute control point displacements
    displacements = target_pts - source_pts

    # Step 2: iterate over all pixels
    for y in range(h):
        for x in range(w):
            offset = np.zeros(2)
            weight_sum = 0.0

            for i in range(len(source_pts)):
                # Distance from current pixel to control point
                r = np.linalg.norm(np.array([x, y]) - source_pts[i])

                # Thin-plate spline / RBF weight
                w_i = 1 / (r**(2*alpha) + eps)

                # Weighted displacement
                offset += w_i * displacements[i]
                weight_sum += w_i

            # Normalize by sum of weights
            if weight_sum > 0:
                offset /= weight_sum

            # Step 3: backward mapping
            src_x = int(np.clip(x + offset[0], 0, w - 1))
            src_y = int(np.clip(y + offset[1], 0, h - 1))

            warped_image[y, x] = image[src_y, src_x]

    return warped_image

# def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
#     h, w = image.shape[:2]
#     warped = np.zeros_like(image)
#     # 转换为浮点坐标
#     p = source_pts.astype(np.float32)  # 源点
#     q = target_pts.astype(np.float32)  # 目标点

#     # 对于每个输出像素 v
#     for v_y in range(h):
#         for v_x in range(w):
#             v = np.array([v_x, v_y], dtype=np.float32)

#             # 计算加权质心 p* 和 q*
#             weights = 1.0 / (np.linalg.norm(p - v, axis=1) ** (2*alpha) + eps)
#             if np.sum(weights) == 0:
#                 warped[v_y, v_x] = image[v_y, v_x]
#                 continue
#             p_star = np.average(p, axis=0, weights=weights)
#             q_star = np.average(q, axis=0, weights=weights)

#             # 去质心坐标
#             p_hat = p - p_star
#             q_hat = q - q_star

#             # 计算 mu 和变换矩阵 (刚性 MLS)
#             # 参考公式 (6) 和 (7)
#             # 此处需根据论文实现详细的矩阵计算
#             # 得到一个旋转向量 f 后，按公式 (8) 计算 u

#             # 简化：用相似性变形近似（省略 rigid 归一化）
#             # 实际应使用论文中的 rigid 公式

#             # 伪代码：假设我们已经算出了源坐标 u
#             u = v  # 待替换

#             # 双线性插值
#             if 0 <= u[0] < w-1 and 0 <= u[1] < h-1:
#                 x0, y0 = int(u[0]), int(u[1])
#                 x1, y1 = x0+1, y0+1
#                 dx = u[0] - x0
#                 dy = u[1] - y0
#                 # 插值四个角点
#                 top = (1-dx)*image[y0, x0] + dx*image[y0, x1]
#                 bottom = (1-dx)*image[y1, x0] + dx*image[y1, x1]
#                 warped[v_y, v_x] = (1-dy)*top + dy*bottom
#             else:
#                 warped[v_y, v_x] = 0
#     return warped


def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
