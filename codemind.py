
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from urllib.request import urlretrieve

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk, ImageOps

APP_NAME = "CodeMind Vision Studio"
VERSION = "2.1"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR, "face_detection_yunet_2023mar.onnx"
)
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

C = {
    "bg": "#090E1A",
    "panel": "#111827",
    "panel2": "#1B2940",
    "border": "#293750",
    "text": "#E5EDF8",
    "muted": "#94A3B8",
    "blue": "#2563EB",
    "cyan": "#22D3EE",
    "green": "#059669",
    "red": "#991B1B",
    "orange": "#92400E",
}


class VisionStudio(ctk.CTk):
    MAX_HISTORY = 30

    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} {VERSION}")
        self.geometry("1400x900")
        self.minsize(1050, 700)
        self.configure(fg_color=C["bg"])

        self.original_image = None
        self.current_image = None
        self.source_path = None
        self.preview_photo = None
        self.preview_bounds = None
        self.crop_start = None
        self.crop_rect = None
        self.zoom = 1.0

        self.face_detector = None
        self.face_count = 0

        self.settings = {
            "brightness": 0,
            "contrast": 1.0,
            "saturation": 1.0,
            "grayscale": False,
            "face_detection": False,
            "filter": "none",
        }

        self.undo_stack = []
        self.redo_stack = []
        self.slider_before = None
        self.slider_job = None

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._status("Ready — open an image to begin")
        self._draw_histogram()

    # =====================================================
    # USER INTERFACE
    # =====================================================

    def _build_ui(self):
        header = ctk.CTkFrame(
            self,
            fg_color=C["panel"],
            corner_radius=14,
            border_width=1,
            border_color=C["border"],
        )
        header.pack(fill="x", padx=16, pady=(16, 10))

        ctk.CTkLabel(
            header,
            text="CM",
            width=46,
            height=46,
            corner_radius=12,
            fg_color=C["blue"],
            font=("Segoe UI", 18, "bold"),
        ).pack(side="left", padx=(14, 10), pady=10)

        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.pack(side="left", pady=10)

        ctk.CTkLabel(
            brand,
            text="CodeMind Vision Studio",
            font=("Segoe UI", 20, "bold"),
            text_color=C["text"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            brand,
            text="IMAGE EDITING  •  COMPUTER VISION  •  2.1",
            font=("Segoe UI", 10, "bold"),
            text_color=C["muted"],
        ).pack(anchor="w")

        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.pack(side="right", padx=10, pady=10)

        self.undo_button = self._button(
            actions, "↶ Undo", self.undo, width=75, state="disabled"
        )
        self.undo_button.pack(side="left", padx=3)

        self.redo_button = self._button(
            actions, "↷ Redo", self.redo, width=75, state="disabled"
        )
        self.redo_button.pack(side="left", padx=3)

        self._button(
            actions, "Open Image", self.open_image, width=100
        ).pack(side="left", padx=3)

        self._button(
            actions,
            "Export",
            self.export_image,
            width=80,
            fg_color=C["green"],
            hover_color="#047857",
        ).pack(side="left", padx=3)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # Image workspace
        workspace = ctk.CTkFrame(
            body,
            fg_color=C["panel"],
            corner_radius=14,
            border_width=1,
            border_color=C["border"],
        )
        workspace.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        workspace.grid_columnconfigure(0, weight=1)
        workspace.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(workspace, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=12)

        ctk.CTkLabel(
            top,
            text="IMAGE CANVAS",
            font=("Segoe UI", 12, "bold"),
            text_color=C["muted"],
        ).pack(side="left")

        self.zoom_label = ctk.CTkLabel(
            top, text="Zoom 100%", text_color=C["cyan"]
        )
        self.zoom_label.pack(side="right")

        self.canvas = tk.Canvas(
            workspace,
            bg="#080D17",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.grid(
            row=1, column=0, sticky="nsew", padx=12, pady=(0, 8)
        )

        self.canvas.bind("<Configure>", self._canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._crop_down)
        self.canvas.bind("<B1-Motion>", self._crop_move)
        self.canvas.bind("<ButtonRelease-1>", self._crop_up)
        self.canvas.bind("<MouseWheel>", self._wheel_zoom)

        self.info_label = ctk.CTkLabel(
            workspace,
            text="No image loaded",
            anchor="w",
            text_color=C["muted"],
            font=("Segoe UI", 11),
        )
        self.info_label.grid(
            row=2, column=0, sticky="ew", padx=16, pady=(0, 12)
        )

        # Tools panel
        tools = ctk.CTkScrollableFrame(
            body,
            width=315,
            fg_color=C["panel"],
            corner_radius=14,
            border_width=1,
            border_color=C["border"],
        )
        tools.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(
            tools,
            text="EDITING TOOLS",
            font=("Segoe UI", 14, "bold"),
            text_color=C["text"],
        ).pack(anchor="w", padx=14, pady=(15, 6))

        self._section(tools, "TONE & COLOR")
        self._slider(tools, "Brightness", -100, 100, 0, "brightness", True)
        self._slider(tools, "Contrast", 0.2, 2.5, 1.0, "contrast")
        self._slider(tools, "Saturation", 0.0, 2.0, 1.0, "saturation")

        self._section(tools, "FILTERS")
        filters = ctk.CTkFrame(tools, fg_color="transparent")
        filters.pack(fill="x", padx=8)

        filter_options = [
            ("Original", "none"),
            ("Sepia", "sepia"),
            ("Sharpen", "sharpen"),
            ("Blur", "blur"),
            ("Edges", "edges"),
            ("Invert", "invert"),
        ]

        for i, (label, mode) in enumerate(filter_options):
            self._button(
                filters,
                label,
                lambda m=mode: self._set_filter(m),
                fg_color=C["panel2"],
                height=32,
            ).grid(
                row=i // 2,
                column=i % 2,
                sticky="ew",
                padx=3,
                pady=3,
            )

        filters.grid_columnconfigure(0, weight=1)
        filters.grid_columnconfigure(1, weight=1)

        self._section(tools, "COMPUTER VISION")

        self.gray_button = self._button(
            tools, "Grayscale: OFF", self._toggle_gray, anchor="w"
        )
        self.gray_button.pack(fill="x", padx=10, pady=3)

        self.face_button = self._button(
            tools, "YuNet Detection: OFF", self._toggle_faces, anchor="w"
        )
        self.face_button.pack(fill="x", padx=10, pady=3)

        self._button(
            tools,
            "Load / Download YuNet",
            self._load_yunet,
            fg_color=C["panel2"],
        ).pack(fill="x", padx=10, pady=3)

        self._section(tools, "IMAGE ANALYTICS")

        self.histogram_canvas = tk.Canvas(
            tools,
            height=130,
            bg="#080D17",
            highlightthickness=1,
            highlightbackground=C["border"],
        )
        self.histogram_canvas.pack(fill="x", padx=10, pady=(3, 6))

        self.analytics_label = ctk.CTkLabel(
            tools,
            text="Open an image to analyze its colors.",
            text_color=C["muted"],
            font=("Consolas", 10),
            justify="left",
            anchor="w",
            wraplength=275,
        )
        self.analytics_label.pack(fill="x", padx=12, pady=4)

        self._button(
            tools,
            "Analyze Colors",
            self._show_color_analysis,
            fg_color=C["panel2"],
        ).pack(fill="x", padx=10, pady=4)

        self._section(tools, "TRANSFORM")

        row = ctk.CTkFrame(tools, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)

        self._button(
            row, "Rotate Left", lambda: self._transform("left")
        ).pack(side="left", expand=True, fill="x", padx=2)

        self._button(
            row, "Rotate Right", lambda: self._transform("right")
        ).pack(side="left", expand=True, fill="x", padx=2)

        row = ctk.CTkFrame(tools, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)

        self._button(
            row, "Flip H", lambda: self._transform("flip_h")
        ).pack(side="left", expand=True, fill="x", padx=2)

        self._button(
            row, "Flip V", lambda: self._transform("flip_v")
        ).pack(side="left", expand=True, fill="x", padx=2)

        dims = ctk.CTkFrame(tools, fg_color="transparent")
        dims.pack(fill="x", padx=10, pady=(6, 2))

        self.width_entry = ctk.CTkEntry(
            dims, placeholder_text="Width (px)"
        )
        self.width_entry.pack(
            side="left", fill="x", expand=True, padx=(0, 3)
        )

        self.height_entry = ctk.CTkEntry(
            dims, placeholder_text="Height (px)"
        )
        self.height_entry.pack(
            side="left", fill="x", expand=True, padx=(3, 0)
        )

        self._button(tools, "Apply Resize", self._resize).pack(
            fill="x", padx=10, pady=3
        )

        self._section(tools, "CROP")

        ctk.CTkLabel(
            tools,
            text="Drag over the image to select a crop area.",
            text_color=C["muted"],
            font=("Segoe UI", 10),
            wraplength=260,
        ).pack(anchor="w", padx=12, pady=(0, 4))

        self._button(
            tools, "Apply Crop", self._crop_apply
        ).pack(fill="x", padx=10, pady=3)

        self._button(
            tools, "Cancel Crop", self._crop_cancel, fg_color=C["panel2"]
        ).pack(fill="x", padx=10, pady=3)

        self._section(tools, "WORKSPACE")

        self._button(tools, "Fit to Window", self._fit).pack(
            fill="x", padx=10, pady=3
        )

        self._button(
            tools,
            "Reset Adjustments",
            self._reset_adjustments,
            fg_color=C["orange"],
            hover_color="#78350F",
        ).pack(fill="x", padx=10, pady=3)

        self._button(
            tools,
            "Reset to Original File",
            self._reset_original,
            fg_color=C["red"],
            hover_color="#7F1D1D",
        ).pack(fill="x", padx=10, pady=(3, 15))

        # Status bar
        footer = ctk.CTkFrame(
            self, fg_color=C["panel"], corner_radius=8, height=34
        )
        footer.pack(fill="x", padx=16, pady=(0, 13))
        footer.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            footer, text="Ready", anchor="w", text_color=C["muted"]
        )
        self.status_label.pack(
            side="left", fill="both", expand=True, padx=10
        )

        self.faces_label = ctk.CTkLabel(
            footer, text="Faces: —", text_color=C["cyan"]
        )
        self.faces_label.pack(side="right", padx=10)

        self._placeholder()

    def _button(self, parent, text, command, **kwargs):
        defaults = {
            "height": 33,
            "corner_radius": 8,
            "font": ("Segoe UI", 11, "bold"),
        }
        defaults.update(kwargs)
        return ctk.CTkButton(
            parent, text=text, command=command, **defaults
        )

    def _section(self, parent, text):
        ctk.CTkLabel(
            parent,
            text=text,
            text_color=C["cyan"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=12, pady=(13, 5))

    def _slider(self, parent, title, low, high, initial, key, integer=False):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=12, pady=4)

        line = ctk.CTkFrame(frame, fg_color="transparent")
        line.pack(fill="x")

        ctk.CTkLabel(
            line, text=title, text_color=C["text"]
        ).pack(side="left")

        value_label = ctk.CTkLabel(
            line, text=str(initial), text_color=C["cyan"]
        )
        value_label.pack(side="right")

        def changed(value):
            if self.original_image is None:
                return

            if self.slider_before is None:
                self.slider_before = self._snapshot()

            if integer:
                value = int(round(value))
                value_label.configure(text=str(value))
            else:
                value = float(value)
                value_label.configure(text=f"{value:.2f}")

            self.settings[key] = value
            self._render()

            if self.slider_job is not None:
                self.after_cancel(self.slider_job)

            self.slider_job = self.after(600, self._finish_slider)

        slider = ctk.CTkSlider(
            frame,
            from_=low,
            to=high,
            number_of_steps=100,
            command=changed,
        )
        slider.set(initial)
        slider.pack(fill="x", pady=(5, 1))

        setattr(self, key + "_slider", slider)
        setattr(self, key + "_label", value_label)

    # =====================================================
    # UNDO / REDO
    # =====================================================

    def _snapshot(self):
        return {
            "image": (
                self.original_image.copy()
                if self.original_image is not None else None
            ),
            "settings": self.settings.copy(),
            "source_path": self.source_path,
            "zoom": self.zoom,
        }

    def _finish_slider(self):
        self.slider_job = None
        if self.slider_before is not None:
            before = self.slider_before
            self.slider_before = None
            self._push_undo(before)

    def _finish_slider_if_needed(self):
        if self.slider_job is not None:
            try:
                self.after_cancel(self.slider_job)
            except tk.TclError:
                pass
            self.slider_job = None

        if self.slider_before is not None:
            before = self.slider_before
            self.slider_before = None
            self._push_undo(before)

    def _commit(self, before):
        self._finish_slider_if_needed()
        self._push_undo(before)

    def _push_undo(self, state):
        if state is None:
            return

        self.undo_stack.append(state)
        if len(self.undo_stack) > self.MAX_HISTORY:
            self.undo_stack.pop(0)

        self.redo_stack.clear()
        self._history_buttons()

    def _restore(self, state):
        image = state["image"]
        self.original_image = image.copy() if image is not None else None
        self.settings = state["settings"].copy()
        self.source_path = state["source_path"]
        self.zoom = state["zoom"]

        self._sync_controls()
        self._crop_cancel()
        self._render()
        self._history_buttons()

    def undo(self):
        self._finish_slider_if_needed()

        if not self.undo_stack:
            self._status("Nothing to undo")
            return

        self.redo_stack.append(self._snapshot())
        self._restore(self.undo_stack.pop())
        self._status("Undo completed")

    def redo(self):
        self._finish_slider_if_needed()

        if not self.redo_stack:
            self._status("Nothing to redo")
            return

        self.undo_stack.append(self._snapshot())
        self._restore(self.redo_stack.pop())
        self._status("Redo completed")

    def _history_buttons(self):
        self.undo_button.configure(
            state="normal" if self.undo_stack else "disabled"
        )
        self.redo_button.configure(
            state="normal" if self.redo_stack else "disabled"
        )

    def _sync_controls(self):
        for key in ("brightness", "contrast", "saturation"):
            slider = getattr(self, key + "_slider", None)
            label = getattr(self, key + "_label", None)

            if slider is not None:
                slider.set(self.settings[key])

            if label is not None:
                value = self.settings[key]
                label.configure(
                    text=(
                        str(value)
                        if key == "brightness"
                        else f"{value:.2f}"
                    )
                )

        self.gray_button.configure(
            text=f"Grayscale: {'ON' if self.settings['grayscale'] else 'OFF'}"
        )
        self.face_button.configure(
            text=(
                "YuNet Detection: "
                f"{'ON' if self.settings['face_detection'] else 'OFF'}"
            )
        )

    # =====================================================
    # OPEN / EXPORT
    # =====================================================

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff",
                ),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            with Image.open(path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")

            self._finish_slider_if_needed()

            if self.original_image is not None:
                self._push_undo(self._snapshot())

            self.original_image = image
            self.source_path = path
            self.settings.update(
                brightness=0,
                contrast=1.0,
                saturation=1.0,
                grayscale=False,
                face_detection=False,
                filter="none",
            )
            self.zoom = 1.0

            self._crop_cancel()
            self._sync_controls()
            self._render()
            self._status(f"Opened: {os.path.basename(path)}")

        except (OSError, ValueError) as exc:
            messagebox.showerror("Open Image Error", str(exc))

    def export_image(self):
        if self.current_image is None:
            self._no_image()
            return

        default_name = "codemind_edited.png"
        if self.source_path:
            base = os.path.splitext(os.path.basename(self.source_path))[0]
            default_name = base + "_edited.png"

        path = filedialog.asksaveasfilename(
            title="Export Image",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("WebP", "*.webp"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif *.tiff"),
            ],
        )
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()

            if ext in (".jpg", ".jpeg"):
                self.current_image.save(path, "JPEG", quality=95)
            elif ext == ".webp":
                self.current_image.save(path, "WEBP", quality=95)
            elif ext == ".bmp":
                self.current_image.save(path, "BMP")
            elif ext in (".tif", ".tiff"):
                self.current_image.save(path, "TIFF")
            else:
                self.current_image.save(path, "PNG")

            self._status(f"Exported: {os.path.basename(path)}")

        except (OSError, ValueError) as exc:
            messagebox.showerror("Export Error", str(exc))

    # =====================================================
    # IMAGE PROCESSING
    # =====================================================

    def _render(self):
        if self.original_image is None:
            self.current_image = None
            self.face_count = 0
            self._placeholder()
            self._update_info()
            self._draw_histogram()
            self.analytics_label.configure(
                text="Open an image to analyze its colors."
            )
            self.faces_label.configure(text="Faces: —")
            return

        try:
            image = np.asarray(self.original_image.convert("RGB")).copy()

            image = cv2.convertScaleAbs(
                image,
                alpha=float(self.settings["contrast"]),
                beta=int(self.settings["brightness"]),
            )

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(
                hsv[:, :, 1] * float(self.settings["saturation"]),
                0,
                255,
            )
            image = cv2.cvtColor(
                hsv.astype(np.uint8), cv2.COLOR_HSV2RGB
            )

            mode = self.settings["filter"]

            if mode == "sepia":
                matrix = np.array(
                    [
                        [0.393, 0.769, 0.189],
                        [0.349, 0.686, 0.168],
                        [0.272, 0.534, 0.131],
                    ],
                    dtype=np.float32,
                )
                image = np.clip(
                    image.astype(np.float32) @ matrix.T, 0, 255
                ).astype(np.uint8)

            elif mode == "sharpen":
                kernel = np.array(
                    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                    dtype=np.float32,
                )
                image = cv2.filter2D(image, -1, kernel)

            elif mode == "blur":
                image = cv2.GaussianBlur(image, (9, 9), 0)

            elif mode == "edges":
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 80, 160)
                image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            elif mode == "invert":
                image = cv2.bitwise_not(image)

            if self.settings["grayscale"]:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            self.face_count = 0
            if self.settings["face_detection"]:
                image, self.face_count = self._detect_faces(image)

            self.current_image = Image.fromarray(image)

            self._show_preview()
            self._update_info()
            self._draw_histogram()

            if self.settings["face_detection"]:
                self.faces_label.configure(
                    text=f"Detected faces: {self.face_count}"
                )
            else:
                self.faces_label.configure(text="Faces: —")

        except (cv2.error, ValueError, MemoryError) as exc:
            self._status(f"Processing error: {exc}")
            messagebox.showerror("Processing Error", str(exc))

    def _set_filter(self, mode):
        if self.original_image is None:
            self._no_image()
            return

        before = self._snapshot()
        self.settings["filter"] = mode
        self._commit(before)
        self._render()
        self._status(f"Filter selected: {mode}")

    def _toggle_gray(self):
        if self.original_image is None:
            self._no_image()
            return

        before = self._snapshot()
        self.settings["grayscale"] = not self.settings["grayscale"]
        self._commit(before)
        self._sync_controls()
        self._render()

    def _toggle_faces(self):
        if self.original_image is None:
            self._no_image()
            return

        if not self.settings["face_detection"]:
            if not self._load_yunet():
                self._status("YuNet is unavailable")
                return

        before = self._snapshot()
        self.settings["face_detection"] = not self.settings["face_detection"]
        self._commit(before)
        self._sync_controls()
        self._render()

    # =====================================================
    # YUNET FACE DETECTION
    # =====================================================

    def _load_yunet(self):
        if self.face_detector is not None:
            return True

        if not os.path.isfile(MODEL_PATH):
            download = messagebox.askyesno(
                "YuNet Model Missing",
                "The YuNet model is missing.\n"
                "Download it now? An internet connection is required.",
            )
            if not download:
                return False

            try:
                os.makedirs(MODEL_DIR, exist_ok=True)
                self._status("Downloading YuNet model...")
                urlretrieve(MODEL_URL, MODEL_PATH)

                if os.path.getsize(MODEL_PATH) < 10000:
                    try:
                        os.remove(MODEL_PATH)
                    except OSError:
                        pass
                    raise ValueError(
                        "Downloaded model file appears incomplete."
                    )

            except Exception as exc:
                messagebox.showerror(
                    "Model Download Failed",
                    f"Could not download YuNet:\n{exc}\n\n"
                    "You can download the model from the official "
                    "OpenCV Zoo repository and place it at:\n"
                    f"{os.path.abspath(MODEL_PATH)}",
                )
                return False

        try:
            factory = getattr(cv2, "FaceDetectorYN_create", None)

            if callable(factory):
                detector = factory(
                    MODEL_PATH, "", (320, 320), 0.75, 0.3, 5000
                )
            else:
                detector_class = getattr(cv2, "FaceDetectorYN", None)
                create = (
                    getattr(detector_class, "create", None)
                    if detector_class else None
                )

                if not callable(create):
                    raise RuntimeError(
                        "This OpenCV build does not provide FaceDetectorYN. "
                        "Install a compatible opencv-python version."
                    )

                detector = create(
                    MODEL_PATH, "", (320, 320), 0.75, 0.3, 5000
                )

            self.face_detector = detector
            self._status("YuNet model loaded")
            return True

        except Exception as exc:
            self.face_detector = None
            messagebox.showerror(
                "YuNet Initialization Error",
                f"Could not initialize the detector:\n{exc}",
            )
            return False

    def _detect_faces(self, rgb):
        if self.face_detector is None:
            return rgb, 0

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]

        try:
            self.face_detector.setInputSize((width, height))
            result = self.face_detector.detect(bgr)

            faces = (
                result[1]
                if isinstance(result, tuple) and len(result) > 1
                else None
            )

            count = 0

            if faces is not None:
                for face in faces:
                    x, y, fw, fh = [
                        int(round(v)) for v in face[:4]
                    ]

                    score = float(face[14]) if len(face) > 14 else 0.0

                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    fw = max(1, min(fw, width - x))
                    fh = max(1, min(fh, height - y))

                    cv2.rectangle(
                        bgr,
                        (x, y),
                        (x + fw, y + fh),
                        (40, 220, 150),
                        2,
                    )

                    cv2.putText(
                        bgr,
                        f"Face {score:.0%}",
                        (x, max(20, y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (40, 220, 150),
                        2,
                    )

                    count += 1

            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), count

        except (cv2.error, AttributeError, TypeError) as exc:
            self._status(f"YuNet error: {exc}")
            return rgb, 0

    # =====================================================
    # HISTOGRAM & COLOR ANALYTICS
    # =====================================================

    def _draw_histogram(self):
        if not hasattr(self, "histogram_canvas"):
            return

        canvas = self.histogram_canvas
        canvas.delete("all")

        if self.current_image is None:
            canvas.create_text(
                130,
                60,
                text="NO IMAGE",
                fill="#94A3B8",
                font=("Segoe UI", 11, "bold"),
            )
            return

        try:
            rgb = np.asarray(self.current_image.convert("RGB"))

            width = max(canvas.winfo_width(), 240)
            height = max(canvas.winfo_height(), 130)

            margin = 8
            plot_height = height - 25
            plot_width = width - 2 * margin

            channels = [
                (0, "#F87171"),
                (1, "#4ADE80"),
                (2, "#60A5FA"),
            ]

            for channel, color in channels:
                histogram = cv2.calcHist(
                    [rgb], [channel], None, [64], [0, 256]
                ).flatten()

                peak = max(float(histogram.max()), 1.0)
                points = []

                for i, value in enumerate(histogram):
                    x = margin + i * plot_width / 63
                    y = height - 18 - (
                        float(value) / peak * plot_height
                    )
                    points.extend((x, y))

                if len(points) >= 4:
                    canvas.create_line(
                        *points,
                        fill=color,
                        width=1.5,
                        smooth=True,
                    )

            canvas.create_text(
                8,
                8,
                anchor="nw",
                text="R     G     B",
                fill="#E5EDF8",
                font=("Segoe UI", 9, "bold"),
            )

        except (cv2.error, ValueError, tk.TclError):
            canvas.delete("all")
            canvas.create_text(
                120,
                60,
                text="Histogram unavailable",
                fill="#F87171",
            )

    def _show_color_analysis(self):
        if self.current_image is None:
            self._no_image()
            return

        try:
            rgb = np.asarray(self.current_image.convert("RGB"))

            average = np.mean(rgb, axis=(0, 1))
            median = np.median(rgb, axis=(0, 1))

            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            mean_saturation = float(np.mean(hsv[:, :, 1])) / 255 * 100
            mean_brightness = float(np.mean(hsv[:, :, 2])) / 255 * 100

            r, g, b = [
                int(round(value)) for value in average
            ]
            mr, mg, mb = [
                int(round(value)) for value in median
            ]

            pixels = rgb.shape[0] * rgb.shape[1]

            report = (
                "COLOR ANALYSIS\n\n"
                f"Dimensions: {rgb.shape[1]} × {rgb.shape[0]} px\n\n"
                f"Average RGB: ({r}, {g}, {b})\n"
                f"Median RGB:  ({mr}, {mg}, {mb})\n\n"
                f"Mean saturation: {mean_saturation:.1f}%\n"
                f"Mean brightness: {mean_brightness:.1f}%\n\n"
                f"Pixels analyzed: {pixels:,}"
            )

            self.analytics_label.configure(text=report)
            self._status("Color analysis completed")

        except (cv2.error, ValueError, MemoryError) as exc:
            messagebox.showerror("Color Analysis Error", str(exc))

    # =====================================================
    # TRANSFORM / RESIZE / RESET
    # =====================================================

    def _transform(self, operation):
        if self.original_image is None:
            self._no_image()
            return

        before = self._snapshot()

        if operation == "left":
            self.original_image = self.original_image.rotate(90, expand=True)
        elif operation == "right":
            self.original_image = self.original_image.rotate(-90, expand=True)
        elif operation == "flip_h":
            self.original_image = ImageOps.mirror(self.original_image)
        elif operation == "flip_v":
            self.original_image = ImageOps.flip(self.original_image)

        self._commit(before)
        self._crop_cancel()
        self._render()
        self._status(f"Transform applied: {operation}")

    def _resize(self):
        if self.original_image is None:
            self._no_image()
            return

        try:
            width = int(self.width_entry.get().strip())
            height = int(self.height_entry.get().strip())

            if not (1 <= width <= 20000 and 1 <= height <= 20000):
                raise ValueError("Dimensions must be between 1 and 20000.")

        except ValueError as exc:
            messagebox.showwarning("Invalid Size", str(exc))
            return

        before = self._snapshot()

        self.original_image = self.original_image.resize(
            (width, height), Image.Resampling.LANCZOS
        )

        self._commit(before)
        self._crop_cancel()
        self._render()
        self._status(f"Resized to {width} × {height}")

    def _reset_adjustments(self):
        if self.original_image is None:
            self._no_image()
            return

        before = self._snapshot()

        self.settings.update(
            brightness=0,
            contrast=1.0,
            saturation=1.0,
            grayscale=False,
            face_detection=False,
            filter="none",
        )

        self._commit(before)
        self._sync_controls()
        self._render()
        self._status("Adjustments reset")

    def _reset_original(self):
        if not self.source_path or not os.path.isfile(self.source_path):
            self._reset_adjustments()
            return

        if not messagebox.askyesno(
            "Restore Original",
            "Reload the image from its original file?",
        ):
            return

        try:
            with Image.open(self.source_path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")

            before = self._snapshot()
            self.original_image = image

            self.settings.update(
                brightness=0,
                contrast=1.0,
                saturation=1.0,
                grayscale=False,
                face_detection=False,
                filter="none",
            )

            self.zoom = 1.0
            self._commit(before)
            self._sync_controls()
            self._crop_cancel()
            self._render()
            self._status("Original file restored")

        except OSError as exc:
            messagebox.showerror("Reset Error", str(exc))

    # =====================================================
    # CROP / ZOOM / PREVIEW
    # =====================================================

    def _crop_down(self, event):
        if self.current_image is None or self.preview_bounds is None:
            return

        left, top, width, height = self.preview_bounds

        if (
            left <= event.x <= left + width
            and top <= event.y <= top + height
        ):
            self.crop_start = (event.x, event.y)
            self.crop_rect = (
                event.x, event.y, event.x, event.y
            )
            self._show_preview()

    def _crop_move(self, event):
        if self.crop_start is None or self.preview_bounds is None:
            return

        left, top, width, height = self.preview_bounds

        x = max(left, min(left + width, event.x))
        y = max(top, min(top + height, event.y))

        self.crop_rect = (*self.crop_start, x, y)
        self._show_preview()

    def _crop_up(self, event):
        if self.crop_start is None or self.preview_bounds is None:
            return

        left, top, width, height = self.preview_bounds

        x = max(left, min(left + width, event.x))
        y = max(top, min(top + height, event.y))

        self.crop_rect = (*self.crop_start, x, y)
        self.crop_start = None
        self._show_preview()
        self._status("Crop selected — click Apply Crop")

    def _crop_apply(self):
        if self.original_image is None:
            self._no_image()
            return

        if self.crop_rect is None or self.preview_bounds is None:
            messagebox.showinfo(
                "Crop", "Drag over the image to select a crop area."
            )
            return

        left, top, display_width, display_height = self.preview_bounds

        x1, y1, x2, y2 = self.crop_rect
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        x1 = max(left, x1)
        x2 = min(left + display_width, x2)
        y1 = max(top, y1)
        y2 = min(top + display_height, y2)

        if x2 - x1 < 4 or y2 - y1 < 4:
            messagebox.showwarning("Crop", "Select a larger area.")
            return

        image_width, image_height = self.original_image.size

        box = (
            max(0, int((x1 - left) * image_width / display_width)),
            max(0, int((y1 - top) * image_height / display_height)),
            min(image_width, int((x2 - left) * image_width / display_width)),
            min(image_height, int((y2 - top) * image_height / display_height)),
        )

        if box[2] <= box[0] or box[3] <= box[1]:
            messagebox.showwarning("Crop", "Invalid crop area.")
            return

        before = self._snapshot()
        self.original_image = self.original_image.crop(box)

        self._commit(before)
        self._crop_cancel()
        self._render()
        self._status(
            f"Cropped to {self.original_image.width} × "
            f"{self.original_image.height}"
        )

    def _crop_cancel(self):
        self.crop_start = None
        self.crop_rect = None

        if hasattr(self, "canvas"):
            self.canvas.delete("crop_box")

    def _show_preview(self):
        if self.current_image is None:
            self._placeholder()
            return

        self.update_idletasks()

        canvas_width = max(20, self.canvas.winfo_width())
        canvas_height = max(20, self.canvas.winfo_height())

        image_width, image_height = self.current_image.size

        fit_scale = min(
            (canvas_width - 24) / image_width,
            (canvas_height - 24) / image_height,
        )
        scale = max(0.02, fit_scale * self.zoom)

        preview_width = max(1, int(image_width * scale))
        preview_height = max(1, int(image_height * scale))

        preview = self.current_image.resize(
            (preview_width, preview_height),
            Image.Resampling.LANCZOS,
        )

        self.preview_photo = ImageTk.PhotoImage(preview)

        left = (canvas_width - preview_width) / 2
        top = (canvas_height - preview_height) / 2

        self.preview_bounds = (
            left, top, preview_width, preview_height
        )

        self.canvas.delete("all")
        self.canvas.create_image(
            left, top, image=self.preview_photo, anchor="nw"
        )

        if self.crop_rect is not None:
            self.canvas.create_rectangle(
                *self.crop_rect,
                outline=C["cyan"],
                width=2,
                dash=(5, 3),
                tags="crop_box",
            )

        self.zoom_label.configure(text=f"Zoom {self.zoom:.0%}")

    def _placeholder(self):
        if not hasattr(self, "canvas"):
            return

        self.canvas.delete("all")
        self.preview_bounds = None

        width = max(200, self.canvas.winfo_width())
        height = max(150, self.canvas.winfo_height())

        self.canvas.create_text(
            width / 2,
            height / 2 - 15,
            text="CODEMIND VISION STUDIO",
            fill="#52627B",
            font=("Segoe UI", 19, "bold"),
        )

        self.canvas.create_text(
            width / 2,
            height / 2 + 20,
            text="Open an image to start editing",
            fill="#71829B",
            font=("Segoe UI", 12),
        )

    def _canvas_resize(self, _event):
        if self.current_image is not None:
            self._show_preview()
        else:
            self._placeholder()

    def _zoom(self, delta):
        self.zoom = max(0.1, min(4.0, self.zoom + delta))

        if self.current_image is not None:
            self._show_preview()
            self._update_info()

    def _wheel_zoom(self, event):
        self._zoom(0.1 if event.delta > 0 else -0.1)

    def _fit(self):
        self.zoom = 1.0

        if self.current_image is not None:
            self._show_preview()
            self._update_info()

    # =====================================================
    # HELPERS / SHORTCUTS
    # =====================================================

    def _update_info(self):
        if self.original_image is None:
            self.info_label.configure(text="No image loaded")
            return

        width, height = self.original_image.size
        file_size = ""

        if self.source_path and os.path.isfile(self.source_path):
            try:
                kb = os.path.getsize(self.source_path) / 1024
                file_size = f" • {kb:.1f} KB"
            except OSError:
                pass

        self.info_label.configure(
            text=(
                f"{width:,} × {height:,} px • RGB • "
                f"Zoom {self.zoom:.0%}{file_size}"
            )
        )

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda _e: self.open_image())
        self.bind("<Control-s>", lambda _e: self.export_image())
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Control-y>", lambda _e: self.redo())
        self.bind("<Control-0>", lambda _e: self._fit())
        self.bind("<Escape>", lambda _e: self._crop_cancel())

    def _status(self, text):
        if hasattr(self, "status_label"):
            self.status_label.configure(text=f"  {text}")

    def _no_image(self):
        messagebox.showinfo(
            "No Image", "Please open an image first."
        )

    def _close(self):
        if self.slider_job is not None:
            try:
                self.after_cancel(self.slider_job)
            except tk.TclError:
                pass

        self.destroy()


if __name__ == "__main__":
    try:
        app = VisionStudio()
        app.mainloop()
    except Exception as exc:
        print(f"{APP_NAME} startup failed: {exc}", file=sys.stderr)
        try:
            messagebox.showerror(
                APP_NAME,
                f"Application startup failed:\n{exc}",
            )
        except tk.TclError:
            pass
        raise
