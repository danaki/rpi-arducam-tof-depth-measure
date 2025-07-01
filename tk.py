from PIL import Image, ImageTk
import tkinter as tk
import cv2
import numpy as np
import ArducamDepthCamera as ac
from sklearn.cluster import HDBSCAN
import pyransac3d as pyrsc
from collections import Counter
from scipy.spatial import ConvexHull
from functions import *


CAMERA_RANGE = 2000
READ_TIMEOUT = 1000


class Application:
    def __init__(self):
        cam, r, K = init_camera(CAMERA_RANGE)
        self.cam = cam
        self.r = r
        self.K = K

        self.depth_buf = None
        self.depth_mask = None
        
        self.root = tk.Tk()
        self.root.title("Arducam ToF camera")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        frame = tk.Frame(self.root)
        frame.grid(row=0, column=0)

        # 1st row
        tk.Label(frame, text="Depth colormap").grid(row=0, column=0)
        tk.Label(frame, text="Confindence").grid(row=0, column=1)
        tk.Label(frame, text="Filtered").grid(row=0, column=2)

        # 2nd row
        self.depth_display = tk.Label(frame)
        self.depth_display.grid(row=1, column=0)
        self.confidence_display = tk.Label(frame)
        self.confidence_display.grid(row=1, column=1)
        self.filtered_display = tk.Label(frame)
        self.filtered_display.grid(row=1, column=2)

        # 3rd row
        tk.Label(frame, text="Aim size").grid(row=2, column=0)
        tk.Label(frame, text="Confidence threshold").grid(row=2, column=1)
        tk.Label(frame, text="Gradient threshold").grid(row=2, column=2)
        
        # 4th row
        self.aim_size = tk.IntVar(value=20)
        aim_size_slider = tk.Scale(
            frame,
            variable=self.aim_size,
            from_=10,
            to=100, 
            orient=tk.HORIZONTAL
        )
        aim_size_slider.grid(row=3, column=0, sticky=tk.EW)

        self.confidence_threshold = tk.IntVar(value=0)
        confidence_slider = tk.Scale(
            frame,
            variable=self.confidence_threshold,
            from_=0,
            to=255, 
            orient=tk.HORIZONTAL
        )
        confidence_slider.grid(row=3, column=1, sticky=tk.EW)

        self.gradient_stds = tk.DoubleVar(value=1.0)
        gradient_stds_slider = tk.Scale(
            frame,
            variable=self.gradient_stds,
            from_=0.1,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL
        )
        gradient_stds_slider.grid(row=3, column=2, sticky=tk.EW)

        # 5th row
        tk.Label(frame, text="Distance, mm (std)").grid(row=4, column=0)
        self.distance = tk.StringVar()
        tk.Label(frame, textvariable=self.distance).grid(row=4, column=1)

        # 6th row
        btn = tk.Button(frame, text="Measure!", highlightbackground='#3E4149', bg='yellow', command=self.take_measure)
        btn.grid(row=5, column=1)

        # 7th row
        self.result = tk.Message(frame, text="\n\n\n\n")
        self.result.grid(row=6, columnspan=3)

        self.video_loop()

    def video_loop(self):
        frame = self.cam.requestFrame(READ_TIMEOUT)
        if frame:
            self.depth_buf = frame.depth_data            
            confidence_buf = frame.confidence_data
            self.cam.releaseFrame(frame)
            
            normalized_depth = normalize_depth(self.depth_buf, self.r)
            confidence_buf[confidence_buf < self.confidence_threshold.get()] = 0
            
            colormap = np.nan_to_num(normalized_depth)
            colormap = colormap.astype(np.uint8)
            colormap = cv2.applyColorMap(255 - colormap, cv2.COLORMAP_DEEPGREEN)
            cv2.rectangle(
                colormap,
                ((colormap.shape[1] - self.aim_size.get()) // 2, (colormap.shape[0] - self.aim_size.get()) // 2),
                ((colormap.shape[1] + self.aim_size.get()) // 2, (colormap.shape[0] + self.aim_size.get()) // 2),
                (255, 0, 0),
                1
            )
            
            depth_tk = ImageTk.PhotoImage(image=Image.fromarray(colormap))
            self.depth_display.imgtk = depth_tk  # anchor imgtk so it does not be deleted by garbage-collector
            self.depth_display.config(image=depth_tk)

            confidence_tk = ImageTk.PhotoImage(image=Image.fromarray(confidence_buf.astype(np.uint8)))
            self.confidence_display.imgtk = confidence_tk
            self.confidence_display.config(image=confidence_tk)

            self.depth_buf = filter_by_confidence(self.depth_buf, confidence_buf, self.confidence_threshold.get())
            aim, gx, gy = aim_metrics(self.depth_buf, self.aim_size.get())

            self.distance.set(f"{aim[0]:.2f} ({aim[1]:.2f})")
            self.depth_mask = filter_by_gradients(self.depth_buf, gx, gy, self.gradient_stds.get())
            filtered_tk = ImageTk.PhotoImage(image=Image.fromarray(self.depth_mask.astype(np.uint8) * 255))
            self.filtered_display.imgtk = filtered_tk
            self.filtered_display.config(image=filtered_tk)
            
        self.root.after(100, self.video_loop)

    def take_measure(self):
        point_2d_coords = points_to_coords(self.depth_mask)
        points_3d_coords = combine_2d_and_depth(point_2d_coords, self.depth_buf)
        points_3d_coords_real = np.apply_along_axis(lambda row: uv_to_world(self.K, row[:2], row[2]), 1, points_3d_coords)
        
        hdb = HDBSCAN()
        hdb.fit(points_3d_coords_real)
        
        densiest_cluster = sorted(Counter(hdb.labels_).items(), key=lambda x: x[1], reverse=True)[0][0]
        points_3d_coords_real_clustered = points_3d_coords_real[(hdb.labels_ == densiest_cluster) & (hdb.probabilities_ > 0.8)]

        projected_x, projected_y = points_3d_fit_plane(points_3d_coords_real_clustered, 5)
        projected_points = np.column_stack((projected_y, projected_x))

        hull = ConvexHull(projected_points, qhull_options="FS")
        hull_points_x, hull_points_y = projected_points[hull.vertices, 0], projected_points[hull.vertices, 1]
        min_hull_points_x = min(hull_points_x)
        min_hull_points_y = min(hull_points_y)
        
        hull_points_x = (hull_points_x - min_hull_points_x + 10).astype(np.int32)
        hull_points_y = (hull_points_y - min_hull_points_y + 10).astype(np.int32)
        
        img = np.zeros((max(hull_points_y) + 20, max(hull_points_x) + 20))
        img = cv2.fillPoly(img, [np.column_stack((hull_points_x, hull_points_y)).reshape((-1,1,2))], 1)

        area, centroid_x, centroid_y, a, b, c, tau1, tau2, roundness = compute_momentums(img)
        side_small = np.sqrt(area * roundness / np.pi) * 2
        side_big = side_small / roundness
        
        message = f"area={area:.0f} mm2\nroundness={roundness:.2f}\na={side_big:.0f} mm\nb={side_small:.2f} mm"
        self.result.configure(text=message)

    def destructor(self):
        self.root.destroy()            
        cv2.destroyAllWindows()


pba = Application()
pba.root.mainloop()
