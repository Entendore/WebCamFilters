"""
Webcam Filter Application with OBS Virtual Camera Output (PySide6)
Features dynamically generated, filter-specific options.
"""

import sys
import time
import cv2
import numpy as np
import pyvirtualcam
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton,
                               QFrame, QSizePolicy, QMessageBox, QGridLayout)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QFont

class FilterEngine:
    """All available filters."""
    
    @staticmethod
    def none(frame, **_): return frame

    @staticmethod
    def grayscale(frame, **_):
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def sepia(frame, intensity=1.0, **_):
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        return np.clip(frame * (1 - intensity) + sepia * intensity, 0, 255).astype(np.uint8)

    @staticmethod
    def blur(frame, kernel_size=15, **_):
        k = int(kernel_size) | 1  # ensure odd
        return cv2.GaussianBlur(frame, (k, k), 0)

    @staticmethod
    def invert(frame, **_):
        return cv2.bitwise_not(frame)

    @staticmethod
    def edge(frame, threshold=80, **_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold, threshold * 2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def cartoon(frame, **_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)

    @staticmethod
    def hue_shift(frame, shift=30, **_):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def vignette(frame, intensity=0.6, **_):
        h, w = frame.shape[:2]
        X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        mask = 1 - (X**2 + Y**2) * intensity
        mask = np.clip(mask, 0, 1)
        return (frame * mask[:, :, None]).astype(np.uint8)

    @staticmethod
    def pixelated(frame, block_size=15, **_):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // block_size), max(1, h // block_size)),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def thermal(frame, **_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def sketch(frame, blur_size=21, **_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        k = int(blur_size) | 1
        blur = cv2.GaussianBlur(inv, (k, k), 0)
        return cv2.cvtColor(cv2.divide(gray, 255 - blur, scale=256), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def mirror(frame, **_):
        return cv2.flip(frame, 1)

    @staticmethod
    def brightness_contrast(frame, brightness=0, contrast=1.0, **_):
        return np.clip(frame * contrast + brightness, 0, 255).astype(np.uint8)

    @staticmethod
    def color_pop(frame, saturation=2.0, **_):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def noise(frame, intensity=50, **_):
        """Adds analog TV/static style noise."""
        noise = np.random.randn(*frame.shape) * (intensity / 2)
        noisy_frame = np.clip(frame.astype(np.float32) + noise, 0, 255)
        return noisy_frame.astype(np.uint8)

    @staticmethod
    def psychedelic(frame, t=0, wobble=60, speed=4.0, saturation=1.8, **_):
        """Animated wavy colors and shifting hue."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        time_shift = int((t * speed * 10) % 180) 
        hsv[:, :, 0] = (hsv[:, :, 0] + time_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        h, w = frame.shape[:2]
        map_y, map_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        wobble_amount = np.sin(map_y / 15.0 + t * speed) * (wobble / 5)
        map_x = map_x.astype(np.float32) + wobble_amount.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return distorted

    # Filter mapping and UI configuration
    FILTERS = {
        "None": none.__func__,
        "Pixelated": pixelated.__func__,
        "Noise / Grain": noise.__func__,
        "Psychedelic (Animated)": psychedelic.__func__,
        "Grayscale": grayscale.__func__,
        "Sepia": sepia.__func__,
        "Blur": blur.__func__,
        "Invert/Negative": invert.__func__,
        "Edge Detect": edge.__func__,
        "Cartoon": cartoon.__func__,
        "Hue Shift": hue_shift.__func__,
        "Vignette": vignette.__func__,
        "Thermal Vision": thermal.__func__,
        "Pencil Sketch": sketch.__func__,
        "Mirror Flip": mirror.__func__,
        "Brightness/Contrast": brightness_contrast.__func__,
        "Color Pop (Saturation)": color_pop.__func__,
    }

    # Configuration for dynamically generating UI sliders
    FILTER_CONFIG = {
        "None": {},
        "Pixelated": {
            "block_size": {"label": "Block Size", "min": 4, "max": 60, "default": 15, "scale": 1.0}
        },
        "Noise / Grain": {
            "intensity": {"label": "Noise Amount", "min": 0, "max": 150, "default": 40, "scale": 1.0}
        },
        "Psychedelic (Animated)": {
            "wobble": {"label": "Wobble", "min": 0, "max": 100, "default": 60, "scale": 1.0},
            "speed": {"label": "Speed", "min": 0, "max": 100, "default": 40, "scale": 0.1},
            "saturation": {"label": "Saturation", "min": 0, "max": 300, "default": 180, "scale": 0.01}
        },
        "Grayscale": {},
        "Sepia": {
            "intensity": {"label": "Intensity", "min": 0, "max": 100, "default": 100, "scale": 0.01}
        },
        "Blur": {
            "kernel_size": {"label": "Blur Amount", "min": 1, "max": 60, "default": 15, "scale": 1.0}
        },
        "Invert/Negative": {},
        "Edge Detect": {
            "threshold": {"label": "Threshold", "min": 10, "max": 200, "default": 80, "scale": 1.0}
        },
        "Cartoon": {},
        "Hue Shift": {
            "shift": {"label": "Hue Shift", "min": 0, "max": 180, "default": 60, "scale": 1.0}
        },
        "Vignette": {
            "intensity": {"label": "Darkness", "min": 0, "max": 100, "default": 60, "scale": 0.01}
        },
        "Thermal Vision": {},
        "Pencil Sketch": {
            "blur_size": {"label": "Detail", "min": 3, "max": 51, "default": 21, "scale": 1.0}
        },
        "Mirror Flip": {},
        "Brightness/Contrast": {
            "brightness": {"label": "Brightness", "min": -100, "max": 100, "default": 0, "scale": 1.0},
            "contrast": {"label": "Contrast", "min": 20, "max": 300, "default": 100, "scale": 0.01}
        },
        "Color Pop (Saturation)": {
            "saturation": {"label": "Saturation", "min": 0, "max": 300, "default": 150, "scale": 0.01}
        },
    }


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    fps_updated = Signal(int)
    vcam_status = Signal(bool, str)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.vcam_enabled = False
        self.vcam = None
        self.current_filter = "None"
        self.filter_params = {} # Will be populated by UI dynamically

    def update_filter(self, name):
        self.current_filter = name

    def update_param(self, key, value):
        self.filter_params[key] = value

    def toggle_vcam(self):
        self.vcam_enabled = not self.vcam_enabled

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_occurred.emit("Could not open webcam (index 0).")
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps_t = time.time()
        fps_count = 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            # Apply filter, passing current time `t` for animated filters
            fn = FilterEngine.FILTERS.get(self.current_filter, FilterEngine.none.__func__)
            try:
                frame = fn(frame, t=time.time(), **self.filter_params)
            except Exception as e:
                pass

            # Handle Virtual Cam lifecycle
            if self.vcam_enabled and self.vcam is None:
                try:
                    self.vcam = pyvirtualcam.Camera(width=width, height=height, fps=30, fmt="RGB")
                    self.vcam_status.emit(True, f"Virtual Cam: ON ({width}x{height})")
                except Exception as e:
                    self.vcam_enabled = False
                    self.vcam_status.emit(False, "VCam Error")
                    self.error_occurred.emit(f"Could not start virtual camera.\n{e}\n\nMake sure OBS is installed.")
            
            if not self.vcam_enabled and self.vcam is not None:
                self.vcam.close()
                self.vcam = None
                self.vcam_status.emit(False, "Virtual Cam: OFF")

            # Send to VCam
            if self.vcam is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.vcam.send(rgb)
                except Exception:
                    pass

            # Emit to UI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            q_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(q_img.copy())

            # FPS
            fps_count += 1
            now = time.time()
            if now - fps_t >= 1.0:
                self.fps_updated.emit(fps_count)
                fps_count = 0
                fps_t = now

        # Cleanup
        if self.vcam is not None:
            self.vcam.close()
        cap.release()

    def stop(self):
        self.running = False
        self.wait()


def clear_layout(layout):
    """Utility to clear dynamically generated widgets."""
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Filters → OBS Virtual Camera")
        self.setGeometry(100, 100, 980, 640)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QLabel { color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }
            QComboBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; padding: 4px; border-radius: 4px; }
            QPushButton { background-color: #a6e3a1; color: #11111b; border: none; padding: 8px; border-radius: 4px; font-weight: bold; font-size: 11pt;}
            QPushButton:hover { background-color: #94e089; }
            QPushButton#stopBtn { background-color: #f38ba8; }
            QPushButton#stopBtn:hover { background-color: #eba0ac; }
            QSlider::groove:horizontal { border: 1px solid #45475a; height: 4px; background: #313244; border-radius: 2px; }
            QSlider::handle:horizontal { background: #89b4fa; width: 12px; margin: -5px 0; border-radius: 2px; }
        """)

        self.worker = CameraWorker()
        self.worker.frame_ready.connect(self.update_preview)
        self.worker.fps_updated.connect(self.update_fps)
        self.worker.vcam_status.connect(self.update_vcam_btn)
        self.worker.error_occurred.connect(self.show_error)

        self._build_ui()
        
        # Initialize default params and start worker
        self.rebuild_options("None")
        self.worker.start()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Left Panel ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)
        
        self.preview_label = QLabel("Starting camera…")
        self.preview_label.setStyleSheet("background-color: #11111b; border: 1px solid #313244; color: #6c7086;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_panel.addWidget(self.preview_label)

        status_bar = QHBoxLayout()
        self.status_label = QLabel("● Idle")
        self.status_label.setStyleSheet("color: #a6e3a1; font-family: 'Consolas'; font-weight: bold; font-size: 11pt;")
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #89b4fa; font-family: 'Consolas'; font-weight: bold; font-size: 11pt;")
        self.fps_label.setAlignment(Qt.AlignRight)
        status_bar.addWidget(self.status_label)
        status_bar.addStretch()
        status_bar.addWidget(self.fps_label)
        left_panel.addLayout(status_bar)

        # --- Right Panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)
        right_panel.setContentsMargins(0, 0, 0, 0)
        
        right_panel.addWidget(self._header("🎯 Filter Selection"))
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(list(FilterEngine.FILTERS.keys()))
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        right_panel.addWidget(self.filter_combo)

        info = QLabel("Apply a filter to your webcam feed.\nOBS will see this app as a camera source.")
        info.setStyleSheet("color: #6c7086; font-size: 9pt;")
        right_panel.addWidget(info)

        right_panel.addWidget(self._header("🎚️ Filter Options"))
        
        # Container for dynamic sliders
        self.options_container = QWidget()
        self.options_layout = QVBoxLayout(self.options_container)
        self.options_layout.setContentsMargins(0, 0, 0, 0)
        self.options_layout.setSpacing(8)
        right_panel.addWidget(self.options_container)
        
        right_panel.addStretch()

        self.vcam_btn = QPushButton("▶ Start Virtual Camera")
        self.vcam_btn.setObjectName("startBtn")
        self.vcam_btn.clicked.connect(self.toggle_vcam)
        right_panel.addWidget(self.vcam_btn)

        self.vcam_status_label = QLabel("Virtual Cam: OFF")
        self.vcam_status_label.setStyleSheet("color: #f38ba8; font-family: 'Consolas'; font-weight: bold;")
        self.vcam_status_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.vcam_status_label)

        tips = QLabel("💡 In OBS:\n1. Add Source → Video Capture Device\n2. Select 'OBS Virtual Camera'\n3. Your filtered feed appears!")
        tips.setStyleSheet("color: #6c7086; font-size: 9pt;")
        right_panel.addWidget(tips)

        # Combine layouts
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        main_layout.addWidget(left_widget, 3)

        right_widget = QWidget()
        right_widget.setFixedWidth(320)
        right_widget.setLayout(right_panel)
        main_layout.addWidget(right_widget, 1)

    def _header(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #cba6f7; font-size: 13pt; font-weight: bold;")
        return lbl

    def on_filter_changed(self, filter_name):
        self.worker.update_filter(filter_name)
        self.rebuild_options(filter_name)

    def rebuild_options(self, filter_name):
        # Clear old widgets
        clear_layout(self.options_layout)
        self.worker.filter_params = {} # Reset params for the new filter

        config = FilterEngine.FILTER_CONFIG.get(filter_name, {})
        
        if not config:
            no_opts = QLabel("No options available for this filter.")
            no_opts.setStyleSheet("color: #6c7086; font-style: italic;")
            self.options_layout.addWidget(no_opts)
            return

        # Create new sliders dynamically
        for key, params in config.items():
            row = QHBoxLayout()
            lbl = QLabel(params["label"])
            lbl.setFixedWidth(90)
            lbl.setStyleSheet("color: #cdd6f4; font-size: 10pt;")
            row.addWidget(lbl)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(params["min"], params["max"])
            slider.setValue(params["default"])
            
            scale = params["scale"]
            val_lbl = QLabel(f"{params['default'] * scale:.2f}")
            val_lbl.setFixedWidth(40)
            val_lbl.setStyleSheet("color: #89b4fa; font-family: 'Consolas';")

            def on_change(v, k=key, sc=scale, vl=val_lbl):
                real_val = v * sc
                vl.setText(f"{real_val:.2f}")
                self.worker.update_param(k, real_val)

            slider.valueChanged.connect(on_change)
            # Initialize worker with default value
            on_change(params["default"])
            
            row.addWidget(slider)
            row.addWidget(val_lbl)
            
            container = QWidget()
            container.setLayout(row)
            self.options_layout.addWidget(container)

    def update_preview(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps}")

    def update_vcam_btn(self, is_on, text):
        self.vcam_status_label.setText(text)
        if is_on:
            self.vcam_btn.setText("⏹ Stop Virtual Camera")
            self.vcam_btn.setObjectName("stopBtn")
            self.status_label.setText("● Live to OBS")
            self.status_label.setStyleSheet("color: #a6e3a1; font-family: 'Consolas'; font-weight: bold; font-size: 11pt;")
            self.vcam_status_label.setStyleSheet("color: #a6e3a1; font-family: 'Consolas'; font-weight: bold;")
        else:
            self.vcam_btn.setText("▶ Start Virtual Camera")
            self.vcam_btn.setObjectName("startBtn")
            self.status_label.setText("● Idle")
            self.status_label.setStyleSheet("color: #a6e3a1; font-family: 'Consolas'; font-weight: bold; font-size: 11pt;")
            self.vcam_status_label.setStyleSheet("color: #f38ba8; font-family: 'Consolas'; font-weight: bold;")
        
        # Refresh stylesheet
        self.vcam_btn.style().unpolish(self.vcam_btn)
        self.vcam_btn.style().polish(self.vcam_btn)

    def toggle_vcam(self):
        self.worker.toggle_vcam()

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def resizeEvent(self, event):
        if self.preview_label.pixmap() is not None:
            self.preview_label.setPixmap(self.preview_label.pixmap().scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())