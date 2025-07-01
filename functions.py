import numpy as np
import cv2
import pyransac3d as pyrsc
import ArducamDepthCamera as ac


def init_camera(max_distance):
    cam = ac.ArducamCamera()
    cam.open(ac.Connection.CSI, 0)
    cam.start(ac.FrameType.DEPTH)
    cam.setControl(ac.Control.RANGE, max_distance)
    
    r = cam.getControl(ac.Control.RANGE)
    # info = cam.getCameraInfo()
    
    fx = cam.getControl(ac.Control.INTRINSIC_FX)
    fy = cam.getControl(ac.Control.INTRINSIC_FY)
    cx = cam.getControl(ac.Control.INTRINSIC_CX)
    cy = cam.getControl(ac.Control.INTRINSIC_CY)
    
    K = np.array([
        (fx, 0,  cx, 0),
        (0,  fy, cy, 0),
        (0,  0,  1,  0)
    ])

    return cam, r, K

def normalize_depth(depth_buf, r):
     return depth_buf * (255.0 / r)

def filter_by_confidence(depth_buf, confidence_buf, confindence_threshold, replace=np.nan):
    depth_buf_copy = depth_buf.copy()      
    depth_buf_copy[confidence_buf < confindence_threshold] = replace

    return depth_buf_copy

def filter_by_gradients(depth_buf, tgxmean, tgymean, stds):
    depth_buf_copy = depth_buf.copy()
    dgx = cv2.Sobel(depth_buf_copy, cv2.CV_32F, 1, 0, ksize=1)
    dgy = cv2.Sobel(depth_buf_copy, cv2.CV_32F, 0, 1, ksize=1)
    
    filtgx = (dgx > (tgxmean[0] - tgxmean[1] * stds)) & (dgx < (tgxmean[0] + tgxmean[1] * stds)).astype(np.uint8)
    filtgy = (dgy > (tgymean[0] - tgymean[1] * stds)) & (dgy < (tgymean[0] + tgymean[1] * stds)).astype(np.uint8)
    
    return filtgx & filtgy

def aim_metrics(depth_buf, aim_size=20):
    half = aim_size // 2
    tcy, tcx = depth_buf.shape[0] // 2, depth_buf.shape[1] // 2
    aim = depth_buf[tcy - half:tcy + half, tcx - half:tcx + half]
    aim = cv2.blur(aim, (half // 2, half // 2))
    aim_mean = np.nanmean(aim)
    aim_std = np.nanstd(aim)
    
    gx = cv2.Sobel(aim, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(aim, cv2.CV_32F, 0, 1, ksize=1)
    
    gx = np.delete(gx, 0, 1)
    gx = np.delete(gx, gx.shape[1] - 1, 1)
    
    gy = np.delete(gy, 0, 0)
    gy = np.delete(gy, gy.shape[0] - 1, 0)

    aim_gx_mean, aim_gx_std = np.nanmean(gx), np.nanstd(gx)
    aim_gy_mean, aim_gy_std = np.nanmean(gy), np.nanstd(gy)

    return (aim_mean, aim_std), (aim_gx_mean, aim_gx_std), (aim_gy_mean, aim_gy_std)

def compute_momentums(mask):
    # https://www.coursera.org/learn/cameraandimaging/supplement/0VOKM/week-4-lecture-handout
    # https://www.coursera.org/api/rest/v1/asset/download/pdf/rRqihEpDQjG1Rs3GE5niBQ?pageStart=&pageEnd=
    area = mask.sum()
    centroid_x = (mask * np.array(range(mask.shape[1]))).sum() / area
    centroid_y = (mask.T * np.array(range(mask.shape[0]))).sum() / area
    a = (mask * ((np.array(range(mask.shape[1])) - centroid_x) ** 2)).sum()
    c = (mask.T * ((np.array(range(mask.shape[0])) - centroid_y) ** 2)).sum()
    xx = np.repeat([np.array(range(mask.shape[1])) - centroid_x], mask.shape[0], axis=0)
    yy = np.repeat([np.array(range(mask.shape[0])) - centroid_y], mask.shape[1], axis=0).T
    b = (mask * np.multiply(xx, yy)).sum() * 2
    tau1 = np.arctan2(b, a - c) / 2
    tau2 = tau1 + np.pi / 2
    E1 = a * np.sin(tau1) ** 2 - b * np.sin(tau1) * np.cos(tau1) + c * np.cos(tau1) ** 2
    E2 = a * np.sin(tau2) ** 2 - b * np.sin(tau2) * np.cos(tau2) + c * np.cos(tau2) ** 2
    E_max, E_min = (E1, E2) if E1 > E2 else (E2, E1)
    roundness = E_min / E_max

    return area, centroid_x, centroid_y, a, b, c, tau1, tau2, roundness

# def compute_momentums(mask):
#     area = mask.sum()
#     centroid_x = (mask * np.array(range(mask.shape[1]))).sum() / area
#     centroid_y = (mask.T * np.array(range(mask.shape[0]))).sum() / area

#     i_bij = (mask * np.array(range(mask.shape[1]))).sum()
#     j_bij = (mask.T * np.array(range(mask.shape[0]))).sum()
          
#     a_ = (mask * (np.array(range(mask.shape[1])) ** 2)).sum()
#     a_2 = 2 * centroid_x * i_bij
#     a_3 = centroid_x ** 2 * area
#     a = a_ - a_2 + a_3
    
#     c_ = (mask.T * ((np.array(range(mask.shape[0]))) ** 2)).sum()
#     c_2 = 2 * centroid_y * j_bij
#     c_3 = centroid_y ** 2 * area
#     c = c_ - c_2 + c_3

#     xx = np.repeat([np.array(range(mask.shape[1]))], mask.shape[0], axis=0)
#     yy = np.repeat([np.array(range(mask.shape[0]))], mask.shape[1], axis=0).T

#     b_ = (mask * np.multiply(xx, yy)).sum()
#     b_2 = centroid_y * i_bij
#     b_3 = centroid_x * j_bij
#     b_4 = centroid_x * centroid_y * area
#     b = 2 * (b_ - b_2 - b_3 + b_4)

#     tau1 = np.arctan2(b, a - c) / 2
#     tau2 = tau1 + np.pi / 2
#     E1 = a * np.sin(tau1) ** 2 - b * np.sin(tau1) * np.cos(tau1) + c * np.cos(tau1) ** 2
#     E2 = a * np.sin(tau2) ** 2 - b * np.sin(tau2) * np.cos(tau2) + c * np.cos(tau2) ** 2
#     E_max, E_min = (E1, E2) if E1 > E2 else (E2, E1)
#     roundness = E_min / E_max

#     return area, centroid_x, centroid_y, a, b, c, tau1, tau2, roundness

def uv_to_world(K, uv, z):
    u, v = uv
    x = (u * 100 - K[0, 2]) * z / K[0, 0]
    y = (v * 100 - K[1, 2]) * z / K[1, 1]
    
    return x, y, z    
    
def points_to_coords(buf):
    # https://stackoverflow.com/questions/53453811/numpy-get-coordinates-where-condition-is-satisfied-together-with-coordinates
    i, j = np.ogrid[(*map(slice, buf.shape),)]
    return np.argwhere(buf > 0 & ((i|2==3) | (j|2==3)))

def combine_2d_and_depth(point_2d_coords, depth_buf):   
    point_depths = depth_buf[point_2d_coords[:, 0], point_2d_coords[:, 1]]
    return np.hstack((point_2d_coords, point_depths.reshape(-1, 1)))

def points_3d_fit_plane(points_3d, thresh):
    plane = pyrsc.Plane()
    best_eq, best_inliers = plane.fit(points_3d, thresh)
    
    plane_points_3d = np.vstack((
        points_3d[best_inliers, 0],
        points_3d[best_inliers, 1],
        points_3d[best_inliers, 2]
    )).T
    
    n = np.array(best_eq[:3])
    x = np.array([1,0,0])    
    x = x - np.dot(x, n) * n
    x /= np.sqrt((x**2).sum())   # make x a unit vector
    y = np.cross(n, x)
    
    projected_x = np.dot(plane_points_3d, x)
    projected_y = np.dot(plane_points_3d, y)    

    return projected_x, projected_y