import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import random
from PIL import Image, ImageTk

def custom_kmeans(data, K=3, max_iter=100, epsilon=1.0):
    N = data.shape[0] # data has dimensions (N, 3)

    # randomly choose K pixels
    random_indices = random.sample(range(N), K)
    centers = data[random_indices].copy()  # (K,3) - 3 color channels

    labels = np.zeros(N, dtype=np.int32)

    for iteration in range(max_iter):
        # calculating distances of pixels to each centroid
        distances = []
        for c in range(K):
            dist = np.linalg.norm(data - centers[c], axis=1)
            distances.append(dist)
        distances = np.array(distances)  # (K, N)

        # assigning pixels to the nearest centroid
        new_labels = np.argmin(distances, axis=0)

        if np.array_equal(new_labels, labels):
            break  # nothing has changed, so we can stop
        labels = new_labels

        # calculating new centroids as the mean of the clusters
        new_centers = np.zeros((K, 3), dtype=np.float32)
        count = np.zeros(K, dtype=np.int32)
        for i in range(N):
            new_centers[labels[i]] += data[i]
            count[labels[i]] += 1

        for c in range(K):
            if count[c] > 0:
                new_centers[c] /= count[c]

        # calculating centroid shift as the stopping criterion
        shift = np.sum(np.linalg.norm(centers - new_centers, axis=1))
        centers = new_centers
        if shift < epsilon:
            break

    centers = np.uint8(centers)
    return labels, centers

def morphological_postprocess(labels_2d, K=3, morph_op=cv2.MORPH_OPEN, kernel_size=3):
    h, w = labels_2d.shape
    new_labels = np.full((h, w), -1, dtype=np.int32)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for c in range(K):
        # Binary mask: convert pixels of cluster c to 255
        mask_c = np.where(labels_2d == c, 255, 0).astype(np.uint8)
        # Morphological operation
        mask_c_clean = cv2.morphologyEx(mask_c, morph_op, kernel)
        # Overwrite new_labels where the mask is 255
        new_labels[mask_c_clean == 255] = c

    # For pixels that remain -1, restore them to the original cluster labels
    remain_mask = (new_labels == -1)
    new_labels[remain_mask] = labels_2d[remain_mask]
    return new_labels


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Artistic Image Segmentation")

        # stored images
        self.original_image_cv2 = None
        self.segmented_image_cv2 = None

        self.num_clusters = 3 # number of clusters - default is 3

        # flag to check whether to use custom K-means implementation
        self.use_custom_kmeans = tk.BooleanVar(value=False)

        # color space
        self.color_space_var = tk.StringVar(value='BGR')  # BGR, HSV, Lab

        # downsampling factor - 1 means no change
        self.downsample_factor = tk.IntVar(value=1)

        # control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        btn_load = tk.Button(control_frame, text="Load image", command=self.load_image)
        btn_load.pack(pady=5)

        cb_custom = tk.Checkbutton(
            control_frame,
            text="Use custom K-means implementation",
            variable=self.use_custom_kmeans)
        cb_custom.pack(pady=5)

        tk.Label(control_frame, text="Number of clusters (K):").pack()
        self.entry_k = tk.Entry(control_frame) # field for entering the number of clusters
        self.entry_k.insert(0, str(self.num_clusters))
        self.entry_k.pack(pady=5)

        # Color space selection
        tk.Label(control_frame, text="Color Space:").pack(pady=(10,0))
        rb_bgr = tk.Radiobutton(control_frame, text="BGR", value='BGR', variable=self.color_space_var)
        rb_bgr.pack(anchor='w')
        rb_hsv = tk.Radiobutton(control_frame, text="HSV", value='HSV', variable=self.color_space_var)
        rb_hsv.pack(anchor='w')
        rb_lab = tk.Radiobutton(control_frame, text="Lab", value='Lab', variable=self.color_space_var)
        rb_lab.pack(anchor='w')

        # downsampling factor selection
        tk.Label(control_frame, text="Downsample factor:").pack(pady=(10,0))

        down_factors = [1, 2, 4, 8, 16]
        for f in down_factors:
            rb = tk.Radiobutton(
                control_frame,
                text=str(f),
                value=f,
                variable=self.downsample_factor)
            rb.pack(anchor='w')

        # button to run the algorithm
        btn_cluster = tk.Button(control_frame, text="Run K-means", command=self.run_kmeans)
        btn_cluster.pack(pady=5)

        # save segmentation result to file
        btn_save_segmented = tk.Button(control_frame, text="Save Segmentation Result", command=self.save_segmented_image)
        btn_save_segmented.pack(pady=5)

        # image display panel
        display_frame = tk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas_original = tk.Canvas(display_frame, bg="gray")
        self.canvas_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_segmented = tk.Canvas(display_frame, bg="gray")
        self.canvas_segmented.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas_original.bind("<Configure>", self.on_resize_original)
        self.canvas_segmented.bind("<Configure>", self.on_resize_segmented)

        # references to currently displayed images
        self.tk_img_original = None
        self.tk_img_segmented = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        self.original_image_cv2 = img
        self.segmented_image_cv2 = None

        self.update_displays()

    def run_kmeans(self):
        if self.original_image_cv2 is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return

        try:
            k = int(self.entry_k.get()) # reading K
            if k <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Provide a valid (positive) number of clusters.")
            return
        self.num_clusters = k

        # reading the selected color space
        color_space = self.color_space_var.get()

        if color_space == 'BGR':
            working_img = self.original_image_cv2.copy()
            convert_back = None
        elif color_space == 'HSV':
            working_img = cv2.cvtColor(self.original_image_cv2, cv2.COLOR_BGR2HSV)
            convert_back = cv2.COLOR_HSV2BGR
        elif color_space == 'Lab':
            working_img = cv2.cvtColor(self.original_image_cv2, cv2.COLOR_BGR2Lab)
            convert_back = cv2.COLOR_Lab2BGR
        else:
            working_img = self.original_image_cv2.copy()
            convert_back = None

        h, w, c = working_img.shape

        # getting the downsampling factor
        ds_factor = self.downsample_factor.get()

        # if factor > 1 then downscale the image
        if ds_factor > 1:
            new_w = w // ds_factor
            new_h = h // ds_factor
            if new_w < 1 or new_h < 1:
                messagebox.showwarning("Warning", f"Factor {ds_factor} is too high relative to the image dimensions.")
                return
            down_img = cv2.resize(working_img, (new_w, new_h), interpolation=cv2.INTER_AREA) 
            reshaped = down_img.reshape((-1, 3)).astype(np.float32)
        else:
            new_h, new_w = h, w
            down_img = working_img
            reshaped = down_img.reshape((-1, 3)).astype(np.float32)

        if self.use_custom_kmeans.get():
            labels, centers = custom_kmeans(reshaped, K=k, max_iter=20, epsilon=1.0)
        else:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            attempts = 5
            _, labels, centers = cv2.kmeans(reshaped, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        labels_2d_down = labels.reshape((new_h, new_w))

        # adjusting labels to the original dimensions
        if ds_factor > 1:
            labels_2d_small_u8 = labels_2d_down.astype(np.uint8)
            labels_2d_up = cv2.resize(labels_2d_small_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            labels_2d = labels_2d_up.astype(np.int32)
        else:
            labels_2d = labels_2d_down

        # postprocessing
        labels_2d_clean = morphological_postprocess(labels_2d, K=k, morph_op=cv2.MORPH_OPEN, kernel_size=3)

        # reconstructing the image from pixels assigned to clusters
        clustered_pixels = centers[labels_2d_clean.flatten()]
        clustered_img = clustered_pixels.reshape((h, w, 3))

        if convert_back is not None:
            self.segmented_image_cv2 = cv2.cvtColor(clustered_img, convert_back)
        else:
            self.segmented_image_cv2 = clustered_img

        self.update_displays()

    def save_segmented_image(self):
        if self.segmented_image_cv2 is None:
            messagebox.showwarning("Warning", "There is no segmented image to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*.*")])
        if not file_path:
            return

        success = cv2.imwrite(file_path, self.segmented_image_cv2)
        if success:
            messagebox.showinfo("Success", f"File saved: {file_path}")
        else:
            messagebox.showerror("Error", "Failed to save the file.")

    # helper functions for resizing images
    def on_resize_original(self, event):
        self.update_displays()

    def on_resize_segmented(self, event):
        self.update_displays()

    def update_displays(self): # display update
        if self.original_image_cv2 is not None:
            self.show_img(self.original_image_cv2, self.canvas_original, is_original=True)
        else:
            self.canvas_original.delete("all")

        if self.segmented_image_cv2 is not None:
            self.show_img(self.segmented_image_cv2, self.canvas_segmented, is_original=False)
        else:
            self.canvas_segmented.delete("all")

    def show_img(self, cv2_img, canvas, is_original=False):
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 1 or canvas_height < 1:
            return

        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        pil_img = pil_img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

        if is_original:
            self.tk_img_original = tk_img
        else:
            self.tk_img_segmented = tk_img


def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
