#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np


class VideoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Editor with Zoom")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2b2b2b")
        
        # Video properties
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame = 0
        self.is_playing = False
        self.original_frame = None
        self.width = 0
        self.height = 0
        
        # Zoom properties
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_dragging = False
        
        # Flag to prevent callback loops
        self.updating_slider = False
        
        # Initialize UI widget references
        self.zoom_level_label = None
        self.zoom_slider = None
        self.canvas = None
        self.timeline_slider = None
        self.time_label = None
        self.frame_label = None
        self.play_btn = None
        self.info_label = None
        self.photo = None
        self.main_frame = None
        
        # Color scheme
        self.colors = {
            'bg_dark': '#2b2b2b',
            'bg_medium': '#3c3c3c',
            'bg_light': '#555555',
            'accent_blue': '#4a9eff',
            'accent_green': '#28a745',
            'btn_gray': '#666666',
            'btn_dark': '#444444',
            'text_light': '#ffffff',
            'text_dark': '#1a1a1a',  # Dark text for buttons
            'text_muted': '#666666'
        }
        
        # Create UI
        self.create_ui()
        
        # Bind resize event
        self.root.bind("<Configure>", self.on_resize)
        
    def create_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top toolbar
        self.create_toolbar()
        
        # Video display area
        self.create_video_canvas()
        
        # Zoom controls
        self.create_zoom_controls()
        
        # Timeline and playback controls
        self.create_playback_controls()
        
    def create_toolbar(self):
        toolbar = tk.Frame(self.main_frame, bg=self.colors['bg_medium'], height=50)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        toolbar.pack_propagate(False)
        
        # Open button
        open_btn = tk.Button(
            toolbar, text="Open Video", command=self.open_video,
            bg=self.colors['accent_blue'], fg=self.colors['text_dark'],
            font=("Arial", 10, "bold"),
            relief=tk.FLAT, padx=15, pady=5
        )
        open_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Export button
        export_btn = tk.Button(
            toolbar, text="Export", command=self.export_video,
            bg=self.colors['accent_green'], fg=self.colors['text_dark'],
            font=("Arial", 10, "bold"),
            relief=tk.FLAT, padx=15, pady=5
        )
        export_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Video info label
        self.info_label = tk.Label(
            toolbar, text="No video loaded",
            bg=self.colors['bg_medium'], fg=self.colors['text_light'],
            font=("Arial", 10)
        )
        self.info_label.pack(side=tk.RIGHT, padx=10)
        
    def create_video_canvas(self):
        # Canvas frame
        canvas_frame = tk.Frame(self.main_frame, bg="#1e1e1e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Video canvas
        self.canvas = tk.Canvas(
            canvas_frame, bg="#1e1e1e", highlightthickness=0,
            width=800, height=450
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for panning
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-1>", self.stop_pan)
        self.canvas.bind("<MouseWheel>", self.mouse_zoom)  # Windows
        self.canvas.bind("<Button-4>", self.mouse_zoom)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.mouse_zoom)    # Linux scroll down
        
        # Placeholder text
        self.canvas.create_text(
            400, 250, text="Open a video to start editing",
            fill=self.colors['text_muted'], font=("Arial", 16), tags="placeholder"
        )
        
    def create_zoom_controls(self):
        zoom_frame = tk.Frame(self.main_frame, bg=self.colors['bg_medium'])
        zoom_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Zoom text label
        zoom_text_label = tk.Label(
            zoom_frame, text="Zoom:",
            bg=self.colors['bg_medium'], fg=self.colors['text_light'],
            font=("Arial", 10)
        )
        zoom_text_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Zoom out button
        zoom_out_btn = tk.Button(
            zoom_frame, text="-", command=self.zoom_out,
            bg=self.colors['bg_light'], fg=self.colors['text_dark'],
            font=("Arial", 12, "bold"),
            width=3, relief=tk.FLAT
        )
        zoom_out_btn.pack(side=tk.LEFT, padx=2)
        
        # Zoom slider
        self.zoom_slider = tk.Scale(
            zoom_frame, from_=10, to=500, orient=tk.HORIZONTAL,
            length=200, command=self.on_zoom_slider,
            bg=self.colors['bg_medium'], fg=self.colors['text_light'],
            highlightthickness=0,
            troughcolor=self.colors['bg_light'], showvalue=False
        )
        self.zoom_slider.set(100)
        self.zoom_slider.pack(side=tk.LEFT, padx=5)
        
        # Zoom in button
        zoom_in_btn = tk.Button(
            zoom_frame, text="+", command=self.zoom_in,
            bg=self.colors['bg_light'], fg=self.colors['text_dark'],
            font=("Arial", 12, "bold"),
            width=3, relief=tk.FLAT
        )
        zoom_in_btn.pack(side=tk.LEFT, padx=2)
        
        # Zoom percentage label
        self.zoom_level_label = tk.Label(
            zoom_frame, text="100%",
            bg=self.colors['bg_medium'], fg=self.colors['text_light'],
            font=("Arial", 10), width=6
        )
        self.zoom_level_label.pack(side=tk.LEFT, padx=10)
        
        # Reset zoom button
        reset_btn = tk.Button(
            zoom_frame, text="Reset", command=self.reset_zoom,
            bg=self.colors['btn_gray'], fg=self.colors['text_dark'],
            font=("Arial", 9),
            relief=tk.FLAT, padx=10
        )
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Fit to window button
        fit_btn = tk.Button(
            zoom_frame, text="Fit", command=self.fit_to_window,
            bg=self.colors['btn_gray'], fg=self.colors['text_dark'],
            font=("Arial", 9),
            relief=tk.FLAT, padx=10
        )
        fit_btn.pack(side=tk.LEFT, padx=5)
        
        # Zoom presets
        presets_label = tk.Label(
            zoom_frame, text="Presets:",
            bg=self.colors['bg_medium'], fg=self.colors['text_light'],
            font=("Arial", 9)
        )
        presets_label.pack(side=tk.LEFT, padx=(20, 5))
        
        # FIXED: Changed fg from text_light to text_dark
        # Also changed bg from btn_dark to bg_light for better readability
        for preset in [50, 100, 150, 200]:
            btn = tk.Button(
                zoom_frame, text=f"{preset}%",
                command=lambda p=preset: self.set_zoom_percent(p),
                bg=self.colors['bg_light'], fg=self.colors['text_dark'],
                font=("Arial", 8),
                relief=tk.FLAT, padx=5
            )
            btn.pack(side=tk.LEFT, padx=2)
        
    def create_playback_controls(self):
        # Timeline frame
        timeline_frame = tk.Frame(self.main_frame, bg=self.colors['bg_medium'])
        timeline_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Timeline slider
        self.timeline_slider = tk.Scale(
            timeline_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            command=self.on_timeline_seek,
            bg=self.colors['bg_medium'], fg=self.colors['text_light'],
            highlightthickness=0,
            troughcolor=self.colors['bg_light'], showvalue=False, length=800
        )
        self.timeline_slider.pack(fill=tk.X, padx=10, pady=5)
        
        # Controls frame
        controls_frame = tk.Frame(self.main_frame, bg=self.colors['bg_medium'])
        controls_frame.pack(fill=tk.X)
        
        # Time label
        self.time_label = tk.Label(
            controls_frame, text="00:00:00 / 00:00:00",
            bg=self.colors['bg_medium'], fg=self.colors['text_light'],
            font=("Arial", 10)
        )
        self.time_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Center buttons frame
        btn_frame = tk.Frame(controls_frame, bg=self.colors['bg_medium'])
        btn_frame.pack(expand=True)
        
        # Previous frame button
        prev_btn = tk.Button(
            btn_frame, text="<<", command=self.prev_frame,
            bg=self.colors['bg_light'], fg=self.colors['text_dark'],
            font=("Arial", 12),
            width=3, relief=tk.FLAT
        )
        prev_btn.pack(side=tk.LEFT, padx=2)
        
        # Step back button
        step_back_btn = tk.Button(
            btn_frame, text="-10", command=lambda: self.step_frames(-10),
            bg=self.colors['bg_light'], fg=self.colors['text_dark'],
            font=("Arial", 10),
            width=4, relief=tk.FLAT
        )
        step_back_btn.pack(side=tk.LEFT, padx=2)
        
        # Play/Pause button
        self.play_btn = tk.Button(
            btn_frame, text="Play", command=self.toggle_play,
            bg=self.colors['accent_blue'], fg=self.colors['text_dark'],
            font=("Arial", 10, "bold"),
            width=6, relief=tk.FLAT
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        # Step forward button
        step_fwd_btn = tk.Button(
            btn_frame, text="+10", command=lambda: self.step_frames(10),
            bg=self.colors['bg_light'], fg=self.colors['text_dark'],
            font=("Arial", 10),
            width=4, relief=tk.FLAT
        )
        step_fwd_btn.pack(side=tk.LEFT, padx=2)
        
        # Next frame button
        next_btn = tk.Button(
            btn_frame, text=">>", command=self.next_frame,
            bg=self.colors['bg_light'], fg=self.colors['text_dark'],
            font=("Arial", 12),
            width=3, relief=tk.FLAT
        )
        next_btn.pack(side=tk.LEFT, padx=2)
        
        # Frame counter
        self.frame_label = tk.Label(
            controls_frame, text="Frame: 0 / 0",
            bg=self.colors['bg_medium'], fg=self.colors['text_light'],
            font=("Arial", 10)
        )
        self.frame_label.pack(side=tk.RIGHT, padx=10, pady=10)
        
    def on_resize(self, event):
        """Handle window resize"""
        if self.original_frame is not None and event.widget == self.root:
            self.root.after(100, lambda: self.apply_zoom_and_display(self.original_frame))
        
    def open_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_video(file_path)
            
    def load_video(self, path):
        try:
            if self.cap:
                self.cap.release()
                
            self.video_path = path
            self.cap = cv2.VideoCapture(path)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.current_frame = 0
            
            # Update UI
            if self.timeline_slider:
                self.timeline_slider.configure(to=max(1, self.total_frames - 1))
            if self.info_label:
                self.info_label.config(
                    text=f"{self.width}x{self.height} | {self.fps:.2f} FPS | {self.total_frames} frames"
                )
            
            # Remove placeholder
            if self.canvas:
                self.canvas.delete("placeholder")
            
            # Reset zoom and pan
            self.reset_zoom()
            
            # Display first frame
            self.root.after(100, self.display_frame)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
        
    def display_frame(self):
        if self.cap is None:
            return
            
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            
            if ret:
                self.original_frame = frame.copy()
                self.apply_zoom_and_display(frame)
                self.update_time_display()
        except Exception as e:
            print(f"Error displaying frame: {e}")
            
    def apply_zoom_and_display(self, frame):
        if frame is None or self.canvas is None:
            return
            
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update canvas to get current dimensions
            self.canvas.update_idletasks()
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 450
                
            # Calculate zoomed dimensions
            zoomed_width = max(1, int(self.width * self.zoom_level))
            zoomed_height = max(1, int(self.height * self.zoom_level))
            
            # Resize frame according to zoom level
            zoomed_frame = cv2.resize(
                frame_rgb, (zoomed_width, zoomed_height),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Calculate crop region for panning
            max_pan_x = max(0, zoomed_width - canvas_width)
            max_pan_y = max(0, zoomed_height - canvas_height)
            
            self.pan_x = max(0, min(self.pan_x, max_pan_x))
            self.pan_y = max(0, min(self.pan_y, max_pan_y))
            
            # Crop the visible region
            x1 = int(self.pan_x)
            y1 = int(self.pan_y)
            x2 = min(x1 + canvas_width, zoomed_width)
            y2 = min(y1 + canvas_height, zoomed_height)
            
            if x2 > x1 and y2 > y1:
                visible_region = zoomed_frame[y1:y2, x1:x2]
                
                # Create image for display
                image = Image.fromarray(visible_region)
                self.photo = ImageTk.PhotoImage(image)
                
                # Clear canvas and display
                self.canvas.delete("video")
                
                # Center the image if smaller than canvas
                display_x = max(0, (canvas_width - (x2 - x1)) // 2)
                display_y = max(0, (canvas_height - (y2 - y1)) // 2)
                
                self.canvas.create_image(
                    display_x, display_y, anchor=tk.NW,
                    image=self.photo, tags="video"
                )
        except Exception as e:
            print(f"Error in apply_zoom_and_display: {e}")
        
    def update_time_display(self):
        try:
            current_time = self.current_frame / self.fps if self.fps > 0 else 0
            total_time = self.total_frames / self.fps if self.fps > 0 else 0
            
            current_str = self.format_time(current_time)
            total_str = self.format_time(total_time)
            
            if self.time_label:
                self.time_label.config(text=f"{current_str} / {total_str}")
            if self.frame_label:
                self.frame_label.config(text=f"Frame: {self.current_frame} / {self.total_frames}")
            
            # Update timeline without triggering callback
            if self.timeline_slider and not self.updating_slider:
                self.updating_slider = True
                self.timeline_slider.set(self.current_frame)
                self.updating_slider = False
        except Exception as e:
            print(f"Error updating time display: {e}")
        
    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
    # Zoom functions
    def set_zoom_percent(self, percent):
        """Set zoom from percentage value"""
        self.set_zoom(percent / 100.0)
        
    def set_zoom(self, level):
        try:
            self.zoom_level = max(0.1, min(5.0, level))
            
            if self.zoom_slider and not self.updating_slider:
                self.updating_slider = True
                self.zoom_slider.set(int(self.zoom_level * 100))
                self.updating_slider = False
                
            if self.zoom_level_label:
                self.zoom_level_label.config(text=f"{int(self.zoom_level * 100)}%")
            
            if self.original_frame is not None:
                self.apply_zoom_and_display(self.original_frame)
        except Exception as e:
            print(f"Error setting zoom: {e}")
            
    def on_zoom_slider(self, value):
        if not self.updating_slider:
            try:
                percent = int(float(value))
                self.zoom_level = percent / 100.0
                
                if self.zoom_level_label:
                    self.zoom_level_label.config(text=f"{percent}%")
                
                if self.original_frame is not None:
                    self.apply_zoom_and_display(self.original_frame)
            except Exception as e:
                print(f"Error in zoom slider: {e}")
        
    def zoom_in(self):
        self.set_zoom(self.zoom_level + 0.1)
        
    def zoom_out(self):
        self.set_zoom(self.zoom_level - 0.1)
        
    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        if self.zoom_slider:
            self.updating_slider = True
            self.zoom_slider.set(100)
            self.updating_slider = False
            
        if self.zoom_level_label:
            self.zoom_level_label.config(text="100%")
        
        if self.original_frame is not None:
            self.apply_zoom_and_display(self.original_frame)
            
    def fit_to_window(self):
        if self.cap is None or self.canvas is None:
            return
            
        try:
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if self.width <= 0 or self.height <= 0:
                return
            if canvas_width <= 1 or canvas_height <= 1:
                return
                
            scale_x = canvas_width / self.width
            scale_y = canvas_height / self.height
            
            self.pan_x = 0
            self.pan_y = 0
            self.set_zoom(min(scale_x, scale_y))
        except Exception as e:
            print(f"Error in fit_to_window: {e}")
            
    def mouse_zoom(self, event):
        try:
            # Determine zoom direction
            if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
                self.set_zoom(self.zoom_level * 1.1)
            elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
                self.set_zoom(self.zoom_level / 1.1)
        except Exception as e:
            print(f"Error in mouse_zoom: {e}")
            
    # Pan functions
    def start_pan(self, event):
        self.drag_start_x = event.x + self.pan_x
        self.drag_start_y = event.y + self.pan_y
        self.is_dragging = True
        
    def do_pan(self, event):
        if self.is_dragging and self.original_frame is not None:
            try:
                self.pan_x = self.drag_start_x - event.x
                self.pan_y = self.drag_start_y - event.y
                self.apply_zoom_and_display(self.original_frame)
            except Exception as e:
                print(f"Error in do_pan: {e}")
            
    def stop_pan(self, event):
        self.is_dragging = False
        
    # Playback functions
    def toggle_play(self):
        if self.cap is None:
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            if self.play_btn:
                self.play_btn.config(text="Pause")
            self.play_video()
        else:
            if self.play_btn:
                self.play_btn.config(text="Play")
            
    def play_video(self):
        if not self.is_playing or self.cap is None:
            return
            
        try:
            if self.current_frame < self.total_frames - 1:
                self.current_frame += 1
                self.display_frame()
                
                # Schedule next frame
                delay = max(1, int(1000 / self.fps)) if self.fps > 0 else 33
                self.root.after(delay, self.play_video)
            else:
                self.is_playing = False
                if self.play_btn:
                    self.play_btn.config(text="Play")
        except Exception as e:
            print(f"Error in play_video: {e}")
            self.is_playing = False
            
    def prev_frame(self):
        if self.cap and self.current_frame > 0:
            self.current_frame -= 1
            self.display_frame()
            
    def next_frame(self):
        if self.cap and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.display_frame()
            
    def step_frames(self, count):
        if self.cap:
            self.current_frame = max(0, min(
                self.total_frames - 1,
                self.current_frame + count
            ))
            self.display_frame()
            
    def on_timeline_seek(self, value):
        if self.cap and not self.updating_slider:
            try:
                self.current_frame = int(float(value))
                self.display_frame()
            except Exception as e:
                print(f"Error in timeline_seek: {e}")
            
    def export_video(self):
        if self.cap is None:
            messagebox.showwarning("Warning", "No video loaded")
            return
            
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
        )
        
        if output_path:
            self.export_with_zoom(output_path)
            
    def export_with_zoom(self, output_path):
        """Export video with current zoom applied"""
        if self.cap is None:
            return
            
        try:
            # Calculate output dimensions
            out_width = max(1, int(self.width * self.zoom_level))
            out_height = max(1, int(self.height * self.zoom_level))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (out_width, out_height))
            
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Exporting...")
            progress_window.geometry("300x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_bar = ttk.Progressbar(
                progress_window, orient=tk.HORIZONTAL,
                length=250, mode='determinate'
            )
            progress_bar.pack(pady=20)
            progress_text_label = tk.Label(progress_window, text="Exporting: 0%")
            progress_text_label.pack()
            
            for i in range(self.total_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Apply zoom
                zoomed_frame = cv2.resize(
                    frame, (out_width, out_height),
                    interpolation=cv2.INTER_LINEAR
                )
                out.write(zoomed_frame)
                
                # Update progress
                progress = (i + 1) / self.total_frames * 100
                progress_bar['value'] = progress
                progress_text_label.config(text=f"Exporting: {int(progress)}%")
                progress_window.update()
                
            out.release()
            progress_window.destroy()
            
            # Reset to current frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            messagebox.showinfo("Success", f"Video exported to:\n{output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
        
    def on_closing(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VideoEditor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()