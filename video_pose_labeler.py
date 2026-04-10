#!/usr/bin/env python3
"""Video Pose Repetition Labeller using Tkinter and OpenCV."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV (cv2) is required. Install it with 'pip install opencv-python'."
    ) from exc

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it with 'pip install Pillow'."
    ) from exc

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class Segment:
    """Represents a labeled segment."""

    start: int
    end: int
    label: str

    def as_dict(self) -> dict:
        return {"start": int(self.start), "end": int(self.end), "label": self.label}


class VideoPoseLabellerApp:
    """Main Tkinter application for labeling repetition segments."""

    MAX_DISPLAY_WIDTH = 960
    MAX_DISPLAY_HEIGHT = 540

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video Pose Repetition Labeller")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Path management
        self.json_root: Optional[Path] = None
        self.dataset_root: Optional[Path] = None
        self.sample_json_paths: List[Path] = []
        self.binary_label: str = ""
        self.state_sequence: List[str] = []
        self.new_video_mode: bool = False
        self.new_video_path: Optional[Path] = None

        # Video playback state
        self.capture: Optional[cv2.VideoCapture] = None
        self.current_frame: int = 0
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.frame_delay_ms: int = 33
        self.playing: bool = False
        self.after_id: Optional[str] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.slider_updating: bool = False

        # Original video dimensions (pixels)
        self.video_width: int = 0
        self.video_height: int = 0
        self.original_frame = None

        # ----------------------------------------------------------
        # Zoom & Pan state
        # ----------------------------------------------------------
        self.zoom_level_label = None
        self.zoom_slider = None
        self.zoom_level: float = 1.0
        self.focal_x: float = 0.5
        self.focal_y: float = 0.5
        self.drag_start_x: int = 0
        self.drag_start_y: int = 0
        self.drag_start_focal_x: float = 0.0
        self.drag_start_focal_y: float = 0.0
        self.is_dragging: bool = False
        self.zoom_slider_updating: bool = False

        # Annotation state
        self.recorded_segments: List[Segment] = []
        self.current_state_index: int = 0
        self.state_start_frame: int = 0
        
        # Annotation mode — one of:
        #   "classic"   Classic State Sequence
        #   "boundary"  Rep Boundary Mode
        #   "manual"    Manual Windowing Mode
        #   "flexible"  Flexible Binary Labels
        self.annotation_mode: str = "classic"

        # Rep boundary sub-state
        self.awaiting_rep_end: bool = False
        self.current_rep_start: Optional[int] = None

        # UI state variables
        self.root_dir_var = tk.StringVar(value="Choose a json_keypoints root folder")
        self.current_state_var = tk.StringVar(value="No sample loaded")
        self.sequence_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Select a folder to begin")
        self.frame_info_var = tk.StringVar(value="Frame: - / -")
        self.current_layout: str = "default"
        self._segment_end: int = 0          # used by _segment_play_loop
        self._unsaved_changes: bool = False
        self._undo_stack: list[list[Segment]] = []
        self._redo_stack: list[list[Segment]] = []
        self._last_browse_dir: Optional[str] = None
       
        # Realtime playback state
        self._rt_playing: bool = False
        self._rt_after_id: Optional[str] = None
        self._rt_start_time: float = 0.0
        self._rt_start_frame: int = 0
        
        # Sequential read optimisation for realtime playback —
        # avoid expensive seeks when reading frames in order.
        self._last_read_frame: int = -1

        # Standard playback speed multiplier (1.0 = full fps, 0.5 = half, etc.)
        self.playback_speed: float = 0.5

        # Build the UI widgets
        self._build_ui()

        # Pre-select default json root if it exists
        default_root = Path.cwd() / "CFRep" / "json_keypoints"
        if default_root.exists():
            self.set_json_root(default_root)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Root grid: 3 rows — content | controls | status bar
        self.root.columnconfigure(0, weight=0)  # sidebar
        self.root.columnconfigure(1, weight=1)  # canvas
        self.root.columnconfigure(2, weight=0)  # controls (vertical layout only)
        self.root.rowconfigure(0, weight=1)     # canvas row
        self.root.rowconfigure(1, weight=0)     # controls row (default layout)
        self.root.rowconfigure(2, weight=0)     # status bar

        self._build_menu()

        # ---- Sidebar (always column 0, row 0) ----
        self.sidebar = ttk.Frame(self.root, padding=10)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.columnconfigure(0, weight=1)

        ttk.Button(
            self.sidebar,
            text="Select json_keypoints folder…",
            command=self.choose_json_root,
        ).grid(row=0, column=0, pady=(0, 6), sticky="ew")

        ttk.Button(
            self.sidebar,
            text="Load New Video…",
            command=self.load_new_video,
        ).grid(row=1, column=0, pady=(0, 6), sticky="ew")

        ttk.Label(
            self.sidebar, textvariable=self.root_dir_var, wraplength=220
        ).grid(row=2, column=0, sticky="ew")

        ttk.Label(
            self.sidebar, text="Exercises", padding=(0, 10, 0, 0)
        ).grid(row=3, column=0, sticky="w")
        self.exercise_list = tk.Listbox(
            self.sidebar, exportselection=False
        )
        self.exercise_list.grid(row=4, column=0, sticky="nsew")
        self.exercise_list.bind("<<ListboxSelect>>", self.on_exercise_select)

        ttk.Label(
            self.sidebar, text="Samples", padding=(0, 10, 0, 0)
        ).grid(row=5, column=0, sticky="w")
        self.sample_list = tk.Listbox(
            self.sidebar, exportselection=False
        )
        self.sample_list.grid(row=6, column=0, sticky="nsew")
        self.sample_list.bind("<Double-Button-1>", self.on_sample_double_click)

        ttk.Button(
            self.sidebar,
            text="Load selected sample",
            command=self.load_selected_sample,
        ).grid(row=7, column=0, pady=(10, 0), sticky="ew")

        ttk.Button(
            self.sidebar,
            text="Rename sample…",
            command=self.rename_sample,
        ).grid(row=8, column=0, pady=(4, 0), sticky="ew")

        ttk.Button(
            self.sidebar,
            text="Build video_config.json",
            command=self.build_video_config,
        ).grid(row=9, column=0, pady=(6, 0), sticky="ew")

        self.sidebar.rowconfigure(4, weight=1)
        self.sidebar.rowconfigure(6, weight=1)

        # No fixed height — the listbox grows/shrinks with the window
        self.exercise_list = tk.Listbox(
            self.sidebar, exportselection=False
        )
        self.exercise_list.grid(row=4, column=0, sticky="nsew")
        self.exercise_list.bind("<<ListboxSelect>>", self.on_exercise_select)

        ttk.Label(
            self.sidebar, text="Samples", padding=(0, 10, 0, 0)
        ).grid(row=5, column=0, sticky="w")

        # No fixed height — the listbox grows/shrinks with the window
        self.sample_list = tk.Listbox(
            self.sidebar, exportselection=False
        )
        self.sample_list.grid(row=6, column=0, sticky="nsew")
        self.sample_list.bind("<Double-Button-1>", self.on_sample_double_click)


        # Both listboxes share the available vertical space equally
        self.sidebar.rowconfigure(4, weight=1)
        self.sidebar.rowconfigure(6, weight=1)

        # ---- Canvas container (always column 1, row 0) ----
        self.canvas_outer = ttk.Frame(self.root, padding=10)
        self.canvas_outer.grid(row=0, column=1, sticky="nsew")
        self.canvas_outer.columnconfigure(0, weight=1)
        self.canvas_outer.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(self.canvas_outer)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(
            canvas_frame,
            bg="#1e1e1e",
            highlightthickness=0,
            width=self.MAX_DISPLAY_WIDTH,
            height=self.MAX_DISPLAY_HEIGHT,
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        self.video_canvas.create_text(
            self.MAX_DISPLAY_WIDTH // 2,
            self.MAX_DISPLAY_HEIGHT // 2,
            text="Open a video to start",
            fill="#666666",
            font=("Arial", 16),
            tags="placeholder",
        )

        self.video_canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.video_canvas.bind("<B1-Motion>", self._on_pan_move)
        self.video_canvas.bind("<ButtonRelease-1>", self._on_pan_stop)
        self.video_canvas.bind("<MouseWheel>", self._on_mouse_zoom)
        self.video_canvas.bind("<Button-4>", self._on_mouse_zoom)
        self.video_canvas.bind("<Button-5>", self._on_mouse_zoom)
        self.video_canvas.bind("<Control-MouseWheel>", self._on_mouse_zoom)
        try:
            self.video_canvas.bind("<Gesture-Pinch>", self._on_pinch_zoom)
        except tk.TclError:
            pass
        self.video_canvas.bind("<Configure>", self._on_canvas_resize)

        # ---- Controls panel — permanent child of self.root.
        #      Re-gridded by layout presets; never destroyed. ----
        self.controls_panel = ttk.Frame(self.root, padding=(6, 6))

        # -- Zoom controls (wrapping row) --
        zoom_wrap = WrapFrame(self.controls_panel, padx=2, pady=2, bg="#f0f0f0")
        zoom_wrap.pack(fill=tk.X, pady=(0, 2))

        zoom_wrap.add(ttk.Label(zoom_wrap, text="Zoom:"))
        zoom_wrap.add(ttk.Button(zoom_wrap, text="−", width=3, command=self.zoom_out))

        self.zoom_slider = ttk.Scale(
            zoom_wrap, from_=10, to=500, orient=tk.HORIZONTAL,
            command=self._on_zoom_slider, length=120,
        )
        self.zoom_slider.set(100)
        zoom_wrap.add(self.zoom_slider)

        zoom_wrap.add(ttk.Button(zoom_wrap, text="+", width=3, command=self.zoom_in))

        self.zoom_level_label = ttk.Label(zoom_wrap, text="100%", width=5)
        zoom_wrap.add(self.zoom_level_label)

        zoom_wrap.add(ttk.Button(zoom_wrap, text="Reset", command=self.reset_zoom))
        zoom_wrap.add(ttk.Button(zoom_wrap, text="Fit",   command=self.fit_to_window))

        for preset in (50, 100, 150, 200):
            zoom_wrap.add(
                ttk.Button(
                    zoom_wrap, text=f"{preset}%", width=4,
                    command=lambda p=preset: self.set_zoom_percent(p),
                )
            )

        # -- Playback + annotation controls (wrapping row) --
        ctrl_wrap = WrapFrame(self.controls_panel, padx=2, pady=2, bg="#f0f0f0")
        ctrl_wrap.pack(fill=tk.X, pady=(0, 2))

        self.play_button = ttk.Button(
            ctrl_wrap, text="▶", width=3, command=self.toggle_play
        )
        ctrl_wrap.add(self.play_button)

        # Realtime play button — time-anchored, skips frames to stay in sync
        self.rt_play_button = ttk.Button(
            ctrl_wrap, text="▶▶", width=3, command=self.toggle_realtime_play
        )
        ctrl_wrap.add(self.rt_play_button)

        ctrl_wrap.add(
            ttk.Button(ctrl_wrap, text="⏮", width=3,
                       command=lambda: self.step_frame(-1))
        )
        ctrl_wrap.add(
            ttk.Button(ctrl_wrap, text="⏭", width=3,
                       command=lambda: self.step_frame(1))
        )

        # Speed selector for standard playback
        ctrl_wrap.add(ttk.Label(ctrl_wrap, text="Speed:"))
        self.speed_var = tk.StringVar(value="0.5x")
        speed_combo = ttk.Combobox(
            ctrl_wrap,
            textvariable=self.speed_var,
            values=["0.25x", "0.5x", "0.75x", "1x", "1.5x", "2x"],
            width=5,
            state="readonly",
        )
        speed_combo.bind("<<ComboboxSelected>>", self._on_speed_changed)
        ctrl_wrap.add(speed_combo)

        self.mark_button = ttk.Button(
            ctrl_wrap, text="Mark end of state",
            command=self.mark_current_state,
        )
        ctrl_wrap.add(self.mark_button)

        self.mark_prep_button = ttk.Button(
            ctrl_wrap, text="Mark as prep",
            command=lambda: self.mark_manual_segment("prep"),
        )
        self.mark_rep_button = ttk.Button(
            ctrl_wrap, text="Mark as rep",
            command=lambda: self.mark_manual_segment("rep"),
        )
        self.mark_norep_button = ttk.Button(
            ctrl_wrap, text="Mark as no-rep",
            command=lambda: self.mark_manual_segment("no-rep"),
        )
        self.mark_finish_button = ttk.Button(
            ctrl_wrap, text="Mark as finish",
            command=lambda: self.mark_manual_segment("finish"),
        )
        
        # Rep boundary button — shown only in boundary mode
        self.mark_rep_boundary_button = ttk.Button(
            ctrl_wrap, text="Mark rep start",
            command=self.mark_rep_boundary,
        )
        # Do NOT add to ctrl_wrap yet — _update_buttons manages visibility
        # Register the manual buttons so they wrap too, but keep them
        # hidden until new_video_mode activates them.

        for btn in (
            self.mark_prep_button, self.mark_rep_button,
            self.mark_norep_button, self.mark_finish_button,
        ):
            ctrl_wrap.add(btn)

        self.undo_button = ttk.Button(
            ctrl_wrap, text="Undo", command=self.undo_last_mark
        )
        ctrl_wrap.add(self.undo_button)

        self.redo_button = ttk.Button(
            ctrl_wrap, text="Redo", command=self.redo_last_mark
        )
        ctrl_wrap.add(self.redo_button)

        self.clear_button = ttk.Button(
            ctrl_wrap, text="Clear", command=self.clear_annotations
        )
        ctrl_wrap.add(self.clear_button)

        self.save_button = ttk.Button(
            ctrl_wrap, text="Save", command=self.save_annotations
        )
        ctrl_wrap.add(self.save_button)

        # Keep references so _update_buttons can show/hide them
        self._ctrl_wrap = ctrl_wrap

        # -- Frame slider --
        slider_frame = ttk.Frame(self.controls_panel)
        slider_frame.pack(fill=tk.X, pady=(2, 2))
        slider_frame.columnconfigure(0, weight=1)

        self.frame_slider = ttk.Scale(
            slider_frame, from_=0, to=1,
            orient=tk.HORIZONTAL, command=self.on_slider_moved,
        )
        self.frame_slider.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            slider_frame, textvariable=self.frame_info_var,
            width=18, anchor="e",
        ).grid(row=0, column=1, padx=(6, 0))

        # -- State / sequence info --
        info_frame = ttk.Frame(self.controls_panel)
        info_frame.pack(fill=tk.X, pady=(2, 0))
        info_frame.columnconfigure(0, weight=1)

        ttk.Label(
            info_frame,
            textvariable=self.current_state_var,
            anchor="w",
            wraplength=400,     # never forces the window wider
        ).grid(row=0, column=0, sticky="ew")

        ttk.Label(
            info_frame,
            textvariable=self.sequence_var,
            anchor="w",
            foreground="#444",
            wraplength=400,     # long sequences wrap instead of stretching
        ).grid(row=1, column=0, sticky="ew")

        # -- Annotation tree --
        self.annotation_tree = ttk.Treeview(
            self.controls_panel,
            columns=("play", "start", "end", "label"),
            show="headings",
            height=1, # was 6 — no longer forces a minimum window height
        )
        self.annotation_tree.heading("play",  text="")
        self.annotation_tree.heading("start", text="Start")
        self.annotation_tree.heading("end",   text="End")
        self.annotation_tree.heading("label", text="Label")
        self.annotation_tree.column("play",  width=32,  anchor="center", stretch=False)
        self.annotation_tree.column("start", width=70,  anchor="center")
        self.annotation_tree.column("end",   width=70,  anchor="center")
        self.annotation_tree.column("label", width=100, anchor="center")
        self.annotation_tree.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.annotation_tree.bind("<Double-1>",       self.on_annotation_double_click)
        self.annotation_tree.bind("<ButtonRelease-1>", self.on_annotation_click)

        # Annotation management buttons (also wrapping)
        ann_wrap = WrapFrame(self.controls_panel, padx=2, pady=2, bg="#f0f0f0")
        ann_wrap.pack(fill=tk.X, pady=(2, 0))

        ann_wrap.add(
            ttk.Button(ann_wrap, text="Edit Selected",
                       command=self.edit_selected_annotation)
        )
        ann_wrap.add(
            ttk.Button(ann_wrap, text="Delete Selected",
                       command=self.delete_selected_annotation)
        )
        ann_wrap.add(
            ttk.Button(ann_wrap, text="Insert Segment",
                       command=self.insert_segment)
        )

        # -- Binary label display (compact, wraps onto its own row if needed) --
        bin_wrap = WrapFrame(self.controls_panel, padx=4, pady=2, bg="#f0f0f0")
        bin_wrap.pack(fill=tk.X, pady=(6, 0))

        bin_wrap.add(
            ttk.Label(
                bin_wrap,
                text="Binary Label:",
                font=("TkDefaultFont", 9, "bold"),
            )
        )

        self.binary_label_var = tk.StringVar(value="—")
        bin_wrap.add(
            ttk.Label(
                bin_wrap,
                textvariable=self.binary_label_var,
                foreground="#0055cc",
                font=("Courier", 9, "bold"),
                width=20,
            )
        )

        self.rep_count_var = tk.StringVar(value="")
        bin_wrap.add(
            ttk.Label(
                bin_wrap,
                textvariable=self.rep_count_var,
                foreground="#555555",
                font=("TkDefaultFont", 9),
            )
        )

        # ---- Status bar (always row 2, spans all columns) ----
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            padding=6,
            wraplength=800,     # long status messages wrap, never widen window
        )
        self.status_bar.grid(row=2, column=0, columnspan=3, sticky="ew")

        # Apply default layout — all widgets exist by this point
        self._apply_default_layout()
        self._update_buttons()

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        # ---- File menu ----
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Select json_keypoints folder…",
            command=self.choose_json_root,
        )
        file_menu.add_command(
            label="Load New Video…",
            command=self.load_new_video,
        )
        file_menu.add_command(
            label="Load Selected Sample",
            command=self.load_selected_sample,
        )
        file_menu.add_command(
            label="Rename Sample…",
            command=self.rename_sample,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Save Annotations",
            command=self.save_annotations,
        )
        file_menu.add_command(
            label="Build video_config.json",
            command=self.build_video_config,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Clean duplicate config entries…",
            command=self.deduplicate_video_configs,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit",
            command=self.on_close,
        )
        menubar.add_cascade(label="File", menu=file_menu)

        # ---- Edit menu ----
        self._edit_menu = tk.Menu(menubar, tearoff=0)
        self._edit_menu.add_command(
            label="Undo",
            command=self.undo_last_mark,
            accelerator="Cmd+Z" if sys.platform == "darwin" else "Ctrl+Z",
            state="disabled",
        )
        self._edit_menu.add_command(
            label="Redo",
            command=self.redo_last_mark,
            accelerator=(
                "Cmd+Shift+Z" if sys.platform == "darwin" else "Ctrl+Y"
            ),
            state="disabled",
        )
        self._edit_menu.add_separator()
        self._edit_menu.add_command(
            label="Clear all annotations",
            command=self.clear_annotations,
        )
        menubar.add_cascade(label="Edit", menu=self._edit_menu)

        # ---- Mode menu ----
        self._mode_menu = tk.Menu(menubar, tearoff=0)
        self.annotation_mode_var = tk.StringVar(value="classic")

        self._mode_menu.add_radiobutton(
            label="Classic State Sequence",
            variable=self.annotation_mode_var,
            value="classic",
            command=self._on_annotation_mode_changed,
        )
        self._mode_menu.add_radiobutton(
            label="Rep Boundary Mode",
            variable=self.annotation_mode_var,
            value="boundary",
            command=self._on_annotation_mode_changed,
        )
        self._mode_menu.add_radiobutton(
            label="Manual Windowing Mode",
            variable=self.annotation_mode_var,
            value="manual",
            command=self._on_annotation_mode_changed,
        )
        self._mode_menu.add_radiobutton(
            label="Flexible Binary Labels",
            variable=self.annotation_mode_var,
            value="flexible",
            command=self._on_annotation_mode_changed,
        )
        self._mode_menu.add_separator()
        self._mode_menu.add_command(
            label="About modes…",
            command=self._show_mode_help,
        )
        menubar.add_cascade(label="Mode", menu=self._mode_menu)

        # ---- View menu ----
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(
            label="Default Layout  (landscape)",
            command=self._apply_default_layout,
        )
        view_menu.add_command(
            label="Vertical Player Layout  (portrait)",
            command=self._apply_vertical_layout,
        )
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

        # Keyboard shortcuts
        modifier = "Command" if sys.platform == "darwin" else "Control"
        self.root.bind_all(
            f"<{modifier}-z>", lambda e: self.undo_last_mark()
        )
        self.root.bind_all(
            f"<{modifier}-Z>", lambda e: self.redo_last_mark()
        )
        if sys.platform != "darwin":
            self.root.bind_all(
                "<Control-y>", lambda e: self.redo_last_mark()
            )
    
    def _on_annotation_mode_changed(self) -> None:
        """Called when the user switches annotation mode via the Mode menu."""
        new_mode = self.annotation_mode_var.get()

        # If switching away from boundary mode mid-rep, warn the user
        if (
            self.annotation_mode == "boundary"
            and self.awaiting_rep_end
            and new_mode != "boundary"
        ):
            if not messagebox.askyesno(
                "Incomplete rep",
                "You have an unfinished rep boundary (start marked, "
                "no end yet).\nDiscard it and switch mode?"
            ):
                # Revert the menu selection
                self.annotation_mode_var.set(self.annotation_mode)
                return
            self.awaiting_rep_end  = False
            self.current_rep_start = None

        self.annotation_mode = new_mode

        mode_descriptions = {
            "classic":  "Classic State Sequence — mark prep → rep → no-rep → finish in order",
            "boundary": "Rep Boundary Mode — mark individual rep start/end points",
            "manual":   "Manual Windowing Mode — create custom segments freely",
            "flexible": "Flexible Binary Labels — edit binary labels and segments on the fly",
        }
        self.status_var.set(f"Mode: {mode_descriptions[new_mode]}")
        self._refresh_state_ui()
        self._update_buttons()
    
    def mark_rep_boundary(self) -> None:
        """Mark rep start or end in Rep Boundary Mode."""
        if not self.capture:
            messagebox.showwarning("No video", "Please load a video first.")
            return
        self.pause_video()
        if not self.awaiting_rep_end:
            # Mark start
            self.current_rep_start = self.current_frame
            self.awaiting_rep_end  = True
            self.mark_rep_boundary_button.config(text="Mark rep end")
            self.status_var.set(
                f"Rep start marked at frame {self.current_frame} — "
                f"now mark the rep end"
            )
            self._update_buttons()
        else:
            # Mark end
            if self.current_frame <= self.current_rep_start:
                messagebox.showwarning(
                    "Invalid range",
                    f"Rep end ({self.current_frame}) must be after "
                    f"rep start ({self.current_rep_start})"
                )
                return
            label = self._determine_rep_label()
            if label is None:
                return
            self._push_undo()
            new_segment = Segment(
                self.current_rep_start, self.current_frame, label
            )
            self.recorded_segments.append(new_segment)
            self.recorded_segments.sort(key=lambda s: s.start)
            seg_count = len([
                s for s in self.recorded_segments
                if s.label in ("rep", "no-rep")
            ])
            self.status_var.set(
                f"Segment {seg_count} ({label}) recorded: "
                f"{self.current_rep_start}–{self.current_frame}"
            )
            self.awaiting_rep_end  = False
            self.current_rep_start = None
            self._update_annotation_view()
            self._refresh_binary_label_display()
            self._update_buttons()
            self._mark_unsaved()

    def _determine_rep_label(self) -> Optional[str]:
        """Determine if the current boundary segment is rep or no-rep."""
        if self.binary_label:
            existing_count = len([
                s for s in self.recorded_segments
                if s.label in ("rep", "no-rep")
            ])
            if existing_count < len(self.binary_label):
                bit = self.binary_label[existing_count]
                return "rep" if bit == "1" else "no-rep"
            else:
                messagebox.showwarning(
                    "All segments marked",
                    f"All {len(self.binary_label)} segments have already "
                    "been marked.\nUpdate the binary label to add more."
                )
                return None
        else:
            result = messagebox.askyesno(
                "Classify segment",
                f"Is frames {self.current_rep_start}–{self.current_frame} "
                "a successful rep?\n\nYes = rep    No = no-rep"
            )
            return "rep" if result else "no-rep"

    def _show_mode_help(self) -> None:
        """Show a brief description of each annotation mode."""
        help_text = (
            "CLASSIC STATE SEQUENCE\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Mark continuous state transitions in order:\n"
            "prep → rep/no-rep → ... → finish\n"
            "Best for samples with known binary labels.\n"
            "\n"
            "REP BOUNDARY MODE\n"
            "━━━━━━━━━━━━━━━━━\n"
            "Mark the start and end of each rep individually.\n"
            "The app classifies each as rep or no-rep automatically\n"
            "using the binary label, or asks you if no label exists.\n"
            "\n"
            "MANUAL WINDOWING MODE\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "Create any segment (prep/rep/no-rep/finish) freely.\n"
            "Ideal for new videos without existing pose data.\n"
            "Binary label is derived automatically when finish is marked.\n"
            "\n"
            "FLEXIBLE BINARY LABELS\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Edit the binary label at any time and re-annotate on the fly.\n"
            "Segments and binary label stay in sync automatically.\n"
            "Best for correcting or refining existing annotations."
        )
        dialog = tk.Toplevel(self.root)
        dialog.title("About Annotation Modes")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth()  // 2) - 220
        y = (dialog.winfo_screenheight() // 2) - 220
        dialog.geometry(f"440x480+{x}+{y}")

        main = ttk.Frame(dialog, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(
            main,
            wrap=tk.WORD,
            font=("TkDefaultFont", 10),
            relief=tk.FLAT,
            bg=dialog.cget("bg"),
            state=tk.NORMAL,
            width=50,
            height=22,
        )
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(
            main, text="Close", command=dialog.destroy
        ).pack(pady=(12, 0))

    # ------------------------------------------------------------------
    # Layout presets
    # ------------------------------------------------------------------
    def _apply_default_layout(self) -> None:
        """Default landscape layout:
        sidebar (col 0) | video canvas (col 1, row 0)
                          controls panel (col 1, row 1)
        status bar spans all columns at row 2.
        """
        self.current_layout = "default"

        # Only two content columns needed
        self.root.columnconfigure(0, weight=0)   # sidebar
        self.root.columnconfigure(1, weight=1)   # canvas + controls
        self.root.columnconfigure(2, weight=0)   # unused — collapse it
        self.root.rowconfigure(0, weight=1)      # canvas stretches
        self.root.rowconfigure(1, weight=0)      # controls fixed height

        # Sidebar
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="ns",
                          padx=0, pady=0)

        # Canvas — row 0, column 1
        self.canvas_outer.grid(row=0, column=1, columnspan=2,
                                sticky="nsew", padx=0, pady=0)
        self.canvas_outer.columnconfigure(0, weight=1)
        self.canvas_outer.rowconfigure(0, weight=1)

        # Landscape canvas dimensions
        self.video_canvas.config(
            width=self.MAX_DISPLAY_WIDTH,
            height=self.MAX_DISPLAY_HEIGHT,
        )

        # Controls panel — row 1, column 1 (below canvas)
        self.controls_panel.grid(row=1, column=1, columnspan=2,
                                  sticky="nsew", padx=0, pady=0)

        # Status bar — row 2
        self.status_bar.grid(row=2, column=0, columnspan=3, sticky="ew")

        if self.original_frame is not None:
            self._apply_zoom_and_display(self.original_frame)

        self.status_var.set("Default layout applied")

    def _apply_vertical_layout(self) -> None:
        """Vertical portrait layout:
        sidebar (col 0) | portrait canvas (col 1) | controls panel (col 2)
        status bar spans all columns at row 2.
        """
        self.current_layout = "vertical"

        self.root.columnconfigure(0, weight=0, minsize=180)  # sidebar
        self.root.columnconfigure(1, weight=0)               # portrait canvas
        self.root.columnconfigure(2, weight=1)               # controls
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        # Sidebar
        self.sidebar.grid(
            row=0, column=0, rowspan=1, sticky="nsew", padx=0, pady=0
        )

        # Canvas — strict 9:16, column 1
        self.canvas_outer.grid(
            row=0, column=1, columnspan=1,
            sticky="ns", padx=(4, 4), pady=0
        )
        self.canvas_outer.columnconfigure(0, weight=1)
        self.canvas_outer.rowconfigure(0, weight=1)

        portrait_w = 320
        portrait_h = int(portrait_w * 16 / 9)
        self.video_canvas.config(width=portrait_w, height=portrait_h)

        # Controls panel — column 2
        self.controls_panel.grid(
            row=0, column=2, columnspan=1,
            sticky="nsew", padx=(4, 0), pady=0
        )

        # Status bar — row 2
        self.status_bar.grid(row=2, column=0, columnspan=3, sticky="ew")

        if self.original_frame is not None:
            self._apply_zoom_and_display(self.original_frame)

        # Wait until all widgets have resolved their minimum sizes before
        # setting the window geometry — this prevents other widgets from
        # immediately overriding the height we set.
        def _set_geometry() -> None:
            self.root.update_idletasks()
            screen_h = self.root.winfo_screenheight()
            # Derive window height from the canvas 9:16 ratio plus chrome
            sidebar_w    = self.sidebar.winfo_reqwidth()
            canvas_col_w = portrait_w + 8          # canvas + its padding
            controls_w   = 480                     # comfortable controls width
            win_w        = sidebar_w + canvas_col_w + controls_w
            # Height = canvas height + status bar + a little padding
            status_h     = self.status_bar.winfo_reqheight()
            win_h        = min(portrait_h + status_h + 24,
                               int(screen_h * 0.92))
            self.root.geometry(f"{win_w}x{win_h}")
            # Re-enforce ratio after geometry settles
            self.root.after(
                50, lambda: self._enforce_portrait_ratio(portrait_w, portrait_h)
            )

        self.root.after_idle(_set_geometry)
        self.status_var.set("Vertical player layout applied")

    def _on_canvas_resize(self, event: tk.Event) -> None:
        """Re-render the current frame on resize.
        In vertical layout enforce a strict 9:16 ratio by deriving the
        canvas height from its actual rendered width every time.
        """
        if event.widget is not self.video_canvas:
            return

        if self.current_layout == "vertical":
            # event.width is the new width Tkinter is proposing.
            # Derive the correct 9:16 height from it.
            new_w = max(1, event.width)
            new_h = int(new_w * 16 / 9)

            # Use after() to break the Configure feedback loop —
            # setting config() inside a Configure handler can cause
            # infinite recursion on some platforms.
            self.root.after(1, lambda w=new_w, h=new_h:
                            self._enforce_portrait_ratio(w, h))
            return

        if self.original_frame is not None:
            self._apply_zoom_and_display(self.original_frame)

    def _enforce_portrait_ratio(self, w: int, h: int) -> None:
        """Apply 9:16 canvas dimensions and re-render, called via after()
        to avoid recursive Configure events."""
        self.video_canvas.config(width=w, height=h)
        if self.original_frame is not None:
            self._apply_zoom_and_display(self.original_frame)

    def _reparent_controls(self, new_parent: tk.Widget) -> ttk.Frame:
        """Detach the controls panel from wherever it currently lives and
        re-attach it as a child of *new_parent*, preserving all child widgets.

        Tkinter does not allow true widget re-parenting, so we create a fresh
        container in the new parent and physically move (grid/pack) each
        direct child widget into it.
        """
        old_panel = self.controls_panel
        new_panel = ttk.Frame(new_parent, padding=10)

        # Collect children in their current pack/grid order
        children = old_panel.winfo_children()
        for child in children:
            info = {}
            try:
                info = child.pack_info()
                child.pack_forget()
                child.pack(
                    in_=new_panel,
                    fill=info.get("fill", tk.NONE),
                    expand=info.get("expand", False),
                    side=info.get("side", tk.TOP),
                    padx=info.get("padx", 0),
                    pady=info.get("pady", 0),
                    anchor=info.get("anchor", tk.CENTER),
                )
            except tk.TclError:
                try:
                    info = child.grid_info()
                    child.grid_forget()
                    child.grid(
                        in_=new_panel,
                        row=info.get("row", 0),
                        column=info.get("column", 0),
                        rowspan=info.get("rowspan", 1),
                        columnspan=info.get("columnspan", 1),
                        sticky=info.get("sticky", ""),
                        padx=info.get("padx", 0),
                        pady=info.get("pady", 0),
                    )
                except tk.TclError:
                    pass

        old_panel.destroy()
        return new_panel

    # ------------------------------------------------------------------
    # Binary label display helper
    # ------------------------------------------------------------------
    def _refresh_binary_label_display(self) -> None:
        """Recompute and display the binary label from current segments."""
        middle = [
            seg for seg in self.recorded_segments
            if seg.label in ("rep", "no-rep")
        ]
        middle.sort(key=lambda s: s.start)

        if middle:
            derived      = "".join(
                "1" if s.label == "rep" else "0" for s in middle
            )
            rep_count    = sum(1 for s in middle if s.label == "rep")
            no_rep_count = sum(1 for s in middle if s.label == "no-rep")
            self.rep_count_var.set(
                f"({rep_count} rep{'s' if rep_count != 1 else ''}, "
                f"{no_rep_count} no-rep{'s' if no_rep_count != 1 else ''})"
            )
        elif self.binary_label and self.annotation_mode not in (
            "manual", "flexible"
        ):
            derived      = self.binary_label
            rep_count    = self.binary_label.count("1")
            no_rep_count = self.binary_label.count("0")
            self.rep_count_var.set(
                f"({rep_count} rep{'s' if rep_count != 1 else ''}, "
                f"{no_rep_count} no-rep{'s' if no_rep_count != 1 else ''})"
            )
        else:
            derived = "—"
            self.rep_count_var.set("")

        # Commit binary_label based on mode
        if self.annotation_mode in ("manual",):
            # Only commit when finish is present
            has_finish = any(
                seg.label == "finish" for seg in self.recorded_segments
            )
            if has_finish and derived != "—":
                self.binary_label = derived
            else:
                self.binary_label = ""
        elif self.annotation_mode == "flexible":
            # Always keep in sync in flexible mode
            if derived != "—":
                self.binary_label = derived
        else:
            # Classic and boundary — sync when segments exist
            if derived != "—":
                self.binary_label = derived

        self.binary_label_var.set(derived)

    # ------------------------------------------------------------------
    # Folder and sample selection
    # ------------------------------------------------------------------
    def choose_json_root(self) -> None:
        selected = filedialog.askdirectory(
            title="Select json_keypoints folder",
            initialdir=self._last_browse_dir,
        )
        if not selected:
            return
        # Remember this folder for next time
        self._last_browse_dir = str(Path(selected).parent)
        self.set_json_root(Path(selected))

    def set_json_root(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if not path.exists() or not path.is_dir():
            messagebox.showerror("Invalid folder", f"{path} is not a valid directory")
            return
        self.json_root = path
        self.dataset_root = path.parent
        self.root_dir_var.set(str(path))
        self.status_var.set("Pick an exercise to continue")
        self.populate_exercises()

    def populate_exercises(self) -> None:
        self.exercise_list.delete(0, tk.END)
        self.sample_list.delete(0, tk.END)
        if not self.json_root:
            return
        for entry in sorted(self.json_root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                self.exercise_list.insert(tk.END, entry.name)

    def on_exercise_select(self, event: tk.Event) -> None:
        del event
        if not self.json_root:
            return
        try:
            selection = self.exercise_list.curselection()
            if not selection:
                return
            exercise = self.exercise_list.get(selection[0])
        except tk.TclError:
            return
        self.populate_samples(exercise)

    def populate_samples(self, exercise: str) -> None:
        self.sample_list.delete(0, tk.END)
        if not self.json_root:
            return
        exercise_dir = self.json_root / exercise
        if not exercise_dir.exists():
            return
        for entry in sorted(exercise_dir.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                self.sample_list.insert(tk.END, entry.name)

    def on_sample_double_click(self, event: tk.Event) -> None:
        del event
        self.load_selected_sample()

    def load_selected_sample(self) -> None:
        if not self.json_root:
            messagebox.showwarning(
                "Select folder", "Please choose a json_keypoints folder first"
            )
            return
        try:
            exercise_idx = self.exercise_list.curselection()
            if not exercise_idx:
                messagebox.showinfo("Select exercise", "Please select an exercise")
                return
            exercise = self.exercise_list.get(exercise_idx[0])
            sample_idx = self.sample_list.curselection()
            if not sample_idx:
                messagebox.showinfo("Select sample", "Please select a sample")
                return
            sample = self.sample_list.get(sample_idx[0])
        except tk.TclError:
            return
        self.load_sample(exercise, sample)

    # ------------------------------------------------------------------
    # Sample loading and validation
    # ------------------------------------------------------------------
    def load_sample(self, exercise: str, sample: str) -> None:
        assert self.json_root is not None
        self.pause_video()
        self.close_video()

        # Explicitly reset ALL mode flags before anything else so
        # _update_buttons always sees a clean normal-sample state.
        self.new_video_mode      = False
        self.new_video_path      = None
        self.binary_label        = ""
        self.state_sequence      = []
        self.recorded_segments   = []
        self.sample_json_paths   = []
        self.current_state_index = 0
        self.state_start_frame   = 0
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.awaiting_rep_end  = False
        self.current_rep_start = None

        sample_dir = self.json_root / exercise / sample
        json_files = sorted(sample_dir.glob("*.json"))
        if not json_files:
            messagebox.showerror(
                "Missing JSON", "No JSON files found for the selected sample"
            )
            return

        # Find the first JSON file whose top-level structure is a dict
        primary_json_path = None
        primary_data      = None
        for candidate in json_files:
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    primary_json_path = candidate
                    primary_data      = loaded
                    break
            except json.JSONDecodeError:
                continue

        if primary_data is None or primary_json_path is None:
            messagebox.showerror(
                "Invalid JSON",
                "No valid JSON object (dict) found in the selected sample folder.\n"
                "All JSON files either failed to parse or contain a list at "
                "the top level."
            )
            return

        binary_label = primary_data.get("binary_label", "")
        if not binary_label:
            binary_label = self.prompt_binary_label(sample)
            if binary_label is None:
                self.status_var.set("Binary label required to proceed")
                return

        if not all(ch in "01" for ch in binary_label):
            messagebox.showerror(
                "Invalid binary label",
                "Binary label must contain only 0 and 1"
            )
            return

        self.binary_label   = binary_label
        self.state_sequence = self._build_state_sequence(binary_label)

        # Only keep JSON files whose top-level structure is a dict
        self.sample_json_paths = []
        for jf in json_files:
            try:
                with jf.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    self.sample_json_paths.append(jf)
            except Exception:
                continue

        video_path = self._resolve_video_path(primary_data, sample)
        if video_path is None or not video_path.exists():
            messagebox.showerror(
                "Video not found",
                "Unable to locate the source video for this sample"
            )
            return

        if not self._open_video(video_path):
            return

        existing_segments = self._validate_existing_annotations(primary_data)
        if existing_segments:
            if messagebox.askyesno(
                "Existing annotations",
                "Existing annotations were found. Do you want to load them?"
            ):
                self._apply_existing_segments(existing_segments)

        self.current_state_index = len(self.recorded_segments)
        self.state_start_frame = (
            0 if not self.recorded_segments
            else self.recorded_segments[-1].end + 1
        )
        self.state_start_frame = min(
            self.state_start_frame, max(self.total_frames - 1, 0)
        )
        self.seek_to_frame(0)
        self.status_var.set(f"Loaded {exercise} / {sample}")
        self._refresh_state_ui()
        self._update_annotation_view()
        self._refresh_binary_label_display()
        self._unsaved_changes = False
        self._mark_saved()
        self._update_buttons()

        if self.current_layout == "vertical":
            self.root.after_idle(self._reapply_vertical_geometry)

        self.root.after(150, self._update_buttons)

    def _reapply_vertical_geometry(self) -> None:
        """Re-lock the window to the correct 9:16 portrait geometry after
        any content change that might have nudged the window size."""
        self.root.update_idletasks()
        portrait_w = self.video_canvas.winfo_reqwidth()
        portrait_h = self.video_canvas.winfo_reqheight()
        if portrait_w < 1 or portrait_h < 1:
            return
        screen_h     = self.root.winfo_screenheight()
        sidebar_w    = self.sidebar.winfo_reqwidth()
        canvas_col_w = portrait_w + 8
        controls_w   = 480
        win_w        = sidebar_w + canvas_col_w + controls_w
        status_h     = self.status_bar.winfo_reqheight()
        win_h        = min(
            portrait_h + status_h + 24, int(screen_h * 0.92)
        )
        self.root.geometry(f"{win_w}x{win_h}")
        self._enforce_portrait_ratio(portrait_w, portrait_h)

    # ------------------------------------------------------------------
    # Binary label and annotation editing
    # ------------------------------------------------------------------
    def on_annotation_double_click(self, event) -> None:
        selection = self.annotation_tree.selection()
        if selection:
            self.edit_selected_annotation()

    def on_annotation_click(self, event: tk.Event) -> None:
        """Handle single clicks on the annotation table:
        - Column 1 (▶)     → play segment from start to end frame
        - Column 2 (Start) → seek to start frame
        - Column 3 (End)   → seek to end frame
        """
        region = self.annotation_tree.identify_region(event.x, event.y)
        if region != "cell":
            return

        column_id = self.annotation_tree.identify_column(event.x)
        row_id    = self.annotation_tree.identify_row(event.y)
        if not row_id:
            return

        values = self.annotation_tree.item(row_id, "values")
        if not values or len(values) < 4:
            return

        # Column mapping: #1=play, #2=start, #3=end, #4=label
        if column_id == "#1":
            # Play segment
            try:
                start = int(values[1])
                end   = int(values[2])
            except ValueError:
                return
            self._play_segment(start, end)

        elif column_id in ("#2", "#3"):
            # Seek to start or end frame
            if not self.capture:
                return
            try:
                frame_index = int(values[1] if column_id == "#2" else values[2])
            except ValueError:
                return
            self.pause_video()
            self.seek_to_frame(frame_index)
            self.status_var.set(
                f"Seeked to {'start' if column_id == '#2' else 'end'}"
                f" frame {frame_index}"
            )

    def _play_segment(self, start: int, end: int) -> None:
        """Seek to *start* and play until *end*, then pause."""
        if not self.capture:
            return
        self.pause_video()
        self.seek_to_frame(start)
        self._segment_end = end
        self.playing = True
        self.play_button.configure(text="⏸")
        self._segment_play_loop()

    def _segment_play_loop(self) -> None:
        """Playback loop that stops automatically at self._segment_end."""
        if not self.playing or not self.capture:
            return
        if self.current_frame >= self._segment_end:
            self.pause_video()
            self.status_var.set(
                f"Finished playing segment "
                f"(frames {self._segment_end - (self._segment_end - self.current_frame)}"
                f"–{self._segment_end})"
            )
            return
        self.current_frame += 1
        self.show_frame(self.current_frame)
        self.after_id = self.root.after(
            self.frame_delay_ms, self._segment_play_loop
        )

    def toggle_realtime_play(self) -> None:
        """Toggle realtime (time-anchored) playback on/off."""
        if not self.capture:
            return
        if self._rt_playing:
            self._stop_realtime_play()
        else:
            self._start_realtime_play()

    def _start_realtime_play(self) -> None:
        """Begin realtime playback from the current frame."""
        if not self.capture or self.total_frames <= 0:
            return
        # Stop normal playback if running
        self.pause_video()

        self._rt_playing      = True
        self._rt_start_frame  = self.current_frame
        self._rt_start_time   = time.perf_counter()
        self.rt_play_button.configure(text="⏸▶")
        self._rt_loop()

    def _stop_realtime_play(self) -> None:
        """Stop realtime playback."""
        self._rt_playing = False
        self.rt_play_button.configure(text="▶▶")
        if self._rt_after_id is not None:
            self.root.after_cancel(self._rt_after_id)
            self._rt_after_id = None

    def _on_speed_changed(self, event=None) -> None:
        """Update playback speed from the speed selector combobox."""
        val = self.speed_var.get().replace("x", "")
        try:
            self.playback_speed = float(val)
        except ValueError:
            self.playback_speed = 0.5
        self.status_var.set(f"Playback speed set to {self.speed_var.get()}")

    def _rt_loop(self) -> None:
        """Time-anchored realtime playback loop.

        Key optimisations vs the naive approach:
        - Sequential reads instead of seeks (handled in show_frame)
        - Slider and frame-info label updated only every N frames to
          reduce UI thread load
        - Next-tick scheduling uses perf_counter for sub-millisecond
          accuracy rather than a fixed after() delay
        """
        if not self._rt_playing or not self.capture:
            return

        now          = time.perf_counter()
        elapsed      = now - self._rt_start_time
        target_frame = self._rt_start_frame + int(elapsed * self.fps)

        if target_frame >= self.total_frames:
            self.seek_to_frame(self.total_frames - 1)
            self._stop_realtime_play()
            self.status_var.set("Realtime playback complete")
            return

        # Render every frame we need to catch up, but cap the catch-up
        # at 3 frames per tick to avoid stalling the UI thread when the
        # system hiccups.
        frames_to_render = min(
            target_frame - self.current_frame, 3
        )

        if frames_to_render > 0:
            for _ in range(frames_to_render):
                next_f = self.current_frame + 1
                if next_f >= self.total_frames:
                    break
                self.current_frame = next_f
                self.show_frame(self.current_frame)

        # Update slider and frame counter only every 6 frames (~5 Hz at
        # 30 fps) to reduce StringVar / Scale overhead significantly.
        if self.current_frame % 6 == 0:
            self.slider_updating = True
            self.frame_slider.set(self.current_frame)
            self.slider_updating = False
            total = max(self.total_frames - 1, 0)
            self.frame_info_var.set(
                f"Frame: {self.current_frame} / {total}"
            )

        # Schedule next tick precisely
        next_frame    = self.current_frame + 1
        time_for_next = (
            (next_frame - self._rt_start_frame) / self.fps
        ) - (time.perf_counter() - self._rt_start_time)

        # Clamp: never wait more than one frame period, never less than 1 ms
        frame_period_ms = int(1000 / self.fps)
        delay_ms = max(1, min(frame_period_ms, int(time_for_next * 1000)))

        self._rt_after_id = self.root.after(delay_ms, self._rt_loop)

    def edit_selected_annotation(self) -> None:
        selection = self.annotation_tree.selection()
        if not selection:
            messagebox.showinfo("No selection", "Please select an annotation to edit.")
            return
        item = selection[0]
        values = self.annotation_tree.item(item, "values")
        if not values or len(values) < 4:
            return
        # values[0]=▶  values[1]=start  values[2]=end  values[3]=label
        start_frame = int(values[1])
        end_frame   = int(values[2])
        label       = values[3]

        segment_index = None
        for i, seg in enumerate(self.recorded_segments):
            if (
                seg.start == start_frame
                and seg.end == end_frame
                and seg.label == label
            ):
                segment_index = i
                break
        if segment_index is None:
            messagebox.showerror("Error", "Could not find the selected segment.")
            return
        dialog = SegmentEditDialog(
            self.root, start_frame, end_frame, label, self.total_frames
        )
        if dialog.result:
            self._push_undo()   # ← snapshot before change
            new_start, new_end, new_label = dialog.result
            self.recorded_segments[segment_index].start = new_start
            self.recorded_segments[segment_index].end   = new_end
            self.recorded_segments[segment_index].label = new_label
            self.recorded_segments.sort(key=lambda seg: seg.start)
            self._update_annotation_view()
            self._refresh_binary_label_display()
            self.status_var.set("Annotation updated")
            self._mark_unsaved()

    def delete_selected_annotation(self) -> None:
        selection = self.annotation_tree.selection()
        if not selection:
            messagebox.showinfo(
                "No selection", "Please select an annotation to delete."
            )
            return
        item = selection[0]
        values = self.annotation_tree.item(item, "values")
        if not values or len(values) < 4:
            return
        # values[0]=▶  values[1]=start  values[2]=end  values[3]=label
        start_frame = int(values[1])
        end_frame   = int(values[2])
        label       = values[3]

        response = messagebox.askyesno(
            "Delete segment",
            f"Delete segment '{label}' ({start_frame}-{end_frame})?"
        )
        if not response:
            return

        for i, seg in enumerate(self.recorded_segments):
            if (
                seg.start == start_frame
                and seg.end == end_frame
                and seg.label == label
            ):
                self._push_undo()   # ← snapshot before change
                self.recorded_segments.pop(i)
                break

        self.current_state_index = len(self.recorded_segments)
        self.state_start_frame = (
            0 if not self.recorded_segments
            else self.recorded_segments[-1].end + 1
        )
        self.state_start_frame = min(
            self.state_start_frame, max(self.total_frames - 1, 0)
        )
        self._refresh_state_ui()
        self._update_annotation_view()
        # Refresh binary label BEFORE _update_buttons so self.binary_label
        # is up to date when _update_buttons checks it for button visibility.
        self._refresh_binary_label_display()
        self._update_buttons()
        self.status_var.set("Annotation deleted")
        self._mark_unsaved()

    def insert_segment(self) -> None:
        if not self.capture:
            messagebox.showwarning("No video", "Please load a video first.")
            return
        # In new_video_mode state_sequence is intentionally empty —
        # that is not an error condition, so we only block on no video.
        if not self.state_sequence and not self.new_video_mode:
            messagebox.showwarning("No sample", "Please load a sample first.")
            return
        dialog = SegmentEditDialog(
            self.root,
            self.current_frame,
            min(self.current_frame + 30, self.total_frames - 1),
            "rep",
            self.total_frames,
        )
        if dialog.result:
            self._push_undo()   # ← snapshot before change
            new_start, new_end, new_label = dialog.result
            new_segment = Segment(new_start, new_end, new_label)
            self.recorded_segments.append(new_segment)
            self.recorded_segments.sort(key=lambda seg: seg.start)
            self._update_annotation_view()
            self._refresh_binary_label_display()
            self.status_var.set("Segment inserted")
            self._mark_unsaved()

    # ------------------------------------------------------------------
    # Build aggregated video_config.json
    # ------------------------------------------------------------------
    def build_video_config(self) -> None:
        if not self.json_root:
            messagebox.showwarning(
                "Select folder", "Please choose a json_keypoints folder first"
            )
            return
        output_path = self.json_root.parent / "video_config.json"
        existing_by_filename: dict[str, dict] = {}
        if output_path.exists():
            try:
                with output_path.open("r", encoding="utf-8") as f:
                    existing_list = json.load(f)
                if isinstance(existing_list, list):
                    for item in existing_list:
                        fn = item.get("filename")
                        if fn:
                            existing_by_filename[str(fn)] = item
            except Exception:
                existing_by_filename = {}
        processed = 0
        skipped = 0
        try:
            exercise_dirs = [
                d for d in sorted(self.json_root.iterdir())
                if d.is_dir() and not d.name.startswith(".")
            ]
        except FileNotFoundError:
            messagebox.showerror(
                "Folder not found", f"Cannot access {self.json_root}"
            )
            return
        for exercise_dir in exercise_dirs:
            for sample_dir in sorted(exercise_dir.iterdir()):
                if not sample_dir.is_dir() or sample_dir.name.startswith("."):
                    continue
                json_files = sorted(sample_dir.glob("*.json"))
                if not json_files:
                    continue
                primary = json_files[0]
                try:
                    with primary.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    skipped += 1
                    continue
                annotations = data.get("annotations")
                if not isinstance(annotations, list) or not annotations:
                    skipped += 1
                    continue
                segments: list[dict] = []
                try:
                    for seg in annotations:
                        start = int(seg["start"])
                        end = int(seg["end"])
                        label = str(seg["label"])
                        segments.append({"start": start, "end": end, "label": label})
                except Exception:
                    skipped += 1
                    continue
                rep_count = sum(1 for s in segments if s.get("label") == "rep")
                video_path_val = data.get("video_path")
                if video_path_val:
                    filename = Path(str(video_path_val)).name
                else:
                    filename = f"{sample_dir.name}.mp4"
                binary_label = str(data.get("binary_label") or "")
                if not binary_label:
                    labels_in_order = [s.get("label") for s in segments]
                    middle = [
                        lbl for lbl in labels_in_order if lbl in ("rep", "no-rep")
                    ]
                    try:
                        binary_label = "".join(
                            "1" if lbl == "rep" else "0" for lbl in middle
                        )
                    except Exception:
                        binary_label = ""
                entry = {
                    "filename": filename,
                    "exercise": exercise_dir.name,
                    "binary_label": binary_label,
                    "rep_count": rep_count,
                    "segments": segments,
                }
                existing_by_filename[filename] = entry
                processed += 1
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            aggregated_list = [
                existing_by_filename[k]
                for k in sorted(existing_by_filename.keys())
            ]
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(aggregated_list, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Write error", f"Failed to write {output_path}: {exc}")
            return
        self.status_var.set(
            f"Updated video_config.json — processed {processed}, skipped {skipped}"
        )
        messagebox.showinfo(
            "Compilation complete",
            f"video_config.json written at:\n{output_path}\n\n"
            f"Processed: {processed}\nSkipped (no annotations): {skipped}",
        )

    def prompt_binary_label(self, sample: str) -> Optional[str]:
        return simpledialog.askstring(
            "Binary label missing",
            f"Enter the binary label (e.g. 1010) for sample '{sample}':",
            parent=self.root,
        )

    def _build_state_sequence(self, binary_label: str) -> List[str]:
        sequence = ["prep"]
        sequence.extend("rep" if bit == "1" else "no-rep" for bit in binary_label)
        sequence.append("finish")
        return sequence

    def _resolve_video_path(
        self, primary_data: dict, sample: str
    ) -> Optional[Path]:
        if self.dataset_root is None:
            return None
        candidate = self.dataset_root / f"{sample}.mp4"
        if candidate.exists():
            return candidate
        video_path_str = primary_data.get("video_path")
        if video_path_str:
            path_candidate = Path(video_path_str)
            if path_candidate.is_absolute():
                return path_candidate
            candidates = [
                self.dataset_root / video_path_str,
                self.dataset_root.parent / video_path_str,
            ]
            for option in candidates:
                if option.exists():
                    return option
        return None

    def _open_video(self, video_path: Path) -> bool:
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            messagebox.showerror("Video error", f"Could not open video: {video_path}")
            self.capture = None
            return False
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 1e-3:
            self.fps = 30.0
        self.frame_delay_ms = max(15, int(1000 / self.fps))
        self.current_frame = 0
        self.frame_slider.configure(from_=0, to=max(self.total_frames - 1, 1))
        self.video_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.video_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.video_canvas.delete("placeholder")
        self.reset_zoom()
        return True

    def _validate_existing_annotations(
        self, primary_data: dict
    ) -> Optional[List[Segment]]:
        annotations = primary_data.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            return None
        segments: List[Segment] = []
        try:
            for item in annotations:
                if item.get("label") == "finish":
                    continue
                start = int(item["start"])
                end = int(item["end"])
                label = str(item["label"])
                if start < 0 or end < start:
                    return None
                segments.append(Segment(start, end, label))
        except (KeyError, TypeError, ValueError):
            return None
        return segments

    def _apply_existing_segments(self, segments: List[Segment]) -> None:
        self.recorded_segments = [
            Segment(seg.start, seg.end, seg.label) for seg in segments
        ]
        self.recorded_segments.sort(key=lambda seg: seg.start)
        if self.total_frames:
            for seg in self.recorded_segments:
                seg.start = max(0, min(seg.start, self.total_frames - 1))
                seg.end = max(0, min(seg.end, self.total_frames - 1))

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------
    def toggle_play(self) -> None:
        if not self.capture:
            return
        if self.playing:
            self.pause_video()
        else:
            self.playing = True
            self.play_button.configure(text="⏸")
            self._play_loop()

    def pause_video(self) -> None:
        if self.playing:
            self.playing = False
            self.play_button.configure(text="▶")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        # Also stop realtime playback if it was running
        if self._rt_playing:
            self._stop_realtime_play()

    def _play_loop(self) -> None:
        """Standard playback loop — honours self.playback_speed so the
        user can watch at a comfortable sub-realtime rate."""
        if not self.playing or not self.capture:
            return
        if self.current_frame >= self.total_frames - 1:
            self.pause_video()
            return
        self.current_frame += 1
        self.show_frame(self.current_frame)
        # Derive delay from fps and speed multiplier — never below 15 ms
        delay_ms = max(15, int(1000 / (self.fps * self.playback_speed)))
        self.after_id = self.root.after(delay_ms, self._play_loop)

    def step_frame(self, delta: int) -> None:
        if not self.capture or self.total_frames <= 0:
            return
        self.pause_video()
        new_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
        self.seek_to_frame(new_frame)

    def seek_to_frame(self, frame_index: int) -> None:
        if not self.capture:
            return
        self.current_frame    = max(0, min(self.total_frames - 1, frame_index))
        self._last_read_frame = self.current_frame - 1  # prime sequential read
        self.show_frame(self.current_frame)

    def on_slider_moved(self, value: str) -> None:
        if self.slider_updating:
            return
        try:
            frame_index = int(float(value))
        except ValueError:
            return
        self.pause_video()
        self.seek_to_frame(frame_index)

    # ------------------------------------------------------------------
    # Frame display — Canvas with zoom & pan
    # ------------------------------------------------------------------
    def show_frame(self, frame_index: int) -> None:
        """Read and display the requested frame.

        If the requested frame is exactly one ahead of the last read
        frame we use a plain sequential read instead of a seek, which
        is significantly faster on most codecs and file systems.
        """
        if not self.capture:
            return

        # Use sequential read when possible to avoid costly seek ops
        if frame_index == self._last_read_frame + 1:
            ok, frame = self.capture.read()
        else:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.capture.read()

        if not ok:
            return

        self._last_read_frame = frame_index
        self.original_frame   = frame.copy()
        self._apply_zoom_and_display(frame)

        self.slider_updating = True
        self.frame_slider.set(frame_index)
        self.slider_updating = False

        total = max(self.total_frames - 1, 0)
        self.frame_info_var.set(f"Frame: {frame_index} / {total}")

    # ------------------------------------------------------------------
    # Zoom & Pan — canvas-based display engine
    # ------------------------------------------------------------------
    def _get_canvas_dimensions(self) -> tuple[int, int]:
        self.video_canvas.update_idletasks()
        w = self.video_canvas.winfo_width()
        h = self.video_canvas.winfo_height()
        if w <= 1 or h <= 1:
            return self.MAX_DISPLAY_WIDTH, self.MAX_DISPLAY_HEIGHT
        return w, h

    def _get_video_display_info(self) -> tuple[int, int, int, int, float, float]:
        canvas_w, canvas_h = self._get_canvas_dimensions()
        zoomed_w = max(1, int(self.video_width * self.zoom_level))
        zoomed_h = max(1, int(self.video_height * self.zoom_level))
        focal_x_px = self.focal_x * zoomed_w
        focal_y_px = self.focal_y * zoomed_h
        view_x = focal_x_px - canvas_w / 2
        view_y = focal_y_px - canvas_h / 2
        max_vx = max(0, zoomed_w - canvas_w)
        max_vy = max(0, zoomed_h - canvas_h)
        view_x = max(0.0, min(view_x, max_vx))
        view_y = max(0.0, min(view_y, max_vy))
        offset_x = max(0, (canvas_w - zoomed_w) // 2)
        offset_y = max(0, (canvas_h - zoomed_h) // 2)
        return zoomed_w, zoomed_h, offset_x, offset_y, view_x, view_y

    def _canvas_to_video_coords(
        self, cx: int, cy: int
    ) -> tuple[Optional[float], Optional[float]]:
        if self.video_width <= 0 or self.video_height <= 0:
            return None, None
        zw, zh, ox, oy, vx, vy = self._get_video_display_info()
        ax = cx - ox
        ay = cy - oy
        if ax < 0 or ay < 0 or ax > zw or ay > zh:
            return None, None
        norm_x = max(0.0, min(1.0, (vx + ax) / zw))
        norm_y = max(0.0, min(1.0, (vy + ay) / zh))
        return norm_x, norm_y

    def _apply_zoom_and_display(self, frame) -> None:
        """Render *frame* (BGR numpy array) onto the canvas respecting
        the current zoom level and focal point.

        Optimised path: when zoom == 1.0 and no pan offset is needed
        we skip the resize and crop to a direct canvas-sized slice,
        saving significant CPU time during realtime playback.
        """
        if frame is None:
            return

        canvas_w, canvas_h = self._get_canvas_dimensions()
        zw, zh, off_x, off_y, vx, vy = self._get_video_display_info()

        # Fast path — no zoom, no pan, frame fits canvas exactly
        if (
            self.zoom_level == 1.0
            and off_x == 0 and off_y == 0
            and vx == 0.0 and vy == 0.0
            and self.video_width == canvas_w
            and self.video_height == canvas_h
        ):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.display_image = ImageTk.PhotoImage(image=image)
            self.video_canvas.delete("video")
            self.video_canvas.create_image(
                0, 0, anchor=tk.NW,
                image=self.display_image, tags="video"
            )
            return

        # Standard path — resize and crop
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize only to the visible region rather than the full
        # zoomed size — avoids allocating a huge intermediate array.
        x1, y1 = int(vx), int(vy)
        x2 = min(x1 + canvas_w, zw)
        y2 = min(y1 + canvas_h, zh)

        if x2 <= x1 or y2 <= y1:
            return

        # Scale source region to destination size directly
        src_x1 = int(x1 * self.video_width  / zw)
        src_y1 = int(y1 * self.video_height / zh)
        src_x2 = int(x2 * self.video_width  / zw)
        src_y2 = int(y2 * self.video_height / zh)

        src_x1 = max(0, min(src_x1, self.video_width))
        src_y1 = max(0, min(src_y1, self.video_height))
        src_x2 = max(0, min(src_x2, self.video_width))
        src_y2 = max(0, min(src_y2, self.video_height))

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return

        dest_w = x2 - x1
        dest_h = y2 - y1

        cropped = frame_rgb[src_y1:src_y2, src_x1:src_x2]
        if cropped.size == 0:
            return

        resized = cv2.resize(
            cropped, (dest_w, dest_h),
            interpolation=cv2.INTER_LINEAR,
        )

        image = Image.fromarray(resized)
        self.display_image = ImageTk.PhotoImage(image=image)

        self.video_canvas.delete("video")
        self.video_canvas.create_image(
            off_x, off_y, anchor=tk.NW,
            image=self.display_image, tags="video"
        )

    # ---- Zoom helpers ----
    def set_zoom_percent(self, percent: int) -> None:
        self._set_zoom(percent / 100.0)

    def _set_zoom(self, level: float) -> None:
        self.zoom_level = max(0.1, min(5.0, level))
        self._sync_zoom_ui()
        if self.original_frame is not None:
            self._apply_zoom_and_display(self.original_frame)

    def _set_zoom_at_point(
        self, new_level: float, cursor_x: int, cursor_y: int
    ) -> None:
        if self.video_width <= 0 or self.video_height <= 0:
            self._set_zoom(new_level)
            return
        video_x, video_y = self._canvas_to_video_coords(cursor_x, cursor_y)
        if video_x is None or video_y is None:
            self._set_zoom(new_level)
            return
        canvas_w, canvas_h = self._get_canvas_dimensions()
        new_level = max(0.1, min(5.0, new_level))
        new_zw = self.video_width * new_level
        new_zh = self.video_height * new_level
        point_x = video_x * new_zw
        point_y = video_y * new_zh
        new_off_x = max(0, (canvas_w - new_zw) / 2)
        new_off_y = max(0, (canvas_h - new_zh) / 2)
        cid_x = cursor_x - new_off_x
        cid_y = cursor_y - new_off_y
        nvx = point_x - cid_x
        nvy = point_y - cid_y
        if new_zw > canvas_w:
            self.focal_x = max(0.0, min(1.0, (nvx + canvas_w / 2) / new_zw))
        else:
            self.focal_x = 0.5
        if new_zh > canvas_h:
            self.focal_y = max(0.0, min(1.0, (nvy + canvas_h / 2) / new_zh))
        else:
            self.focal_y = 0.5
        self.zoom_level = new_level
        self._sync_zoom_ui()
        if self.original_frame is not None:
            self._apply_zoom_and_display(self.original_frame)

    def _sync_zoom_ui(self) -> None:
        if self.zoom_slider is not None and not self.zoom_slider_updating:
            self.zoom_slider_updating = True
            self.zoom_slider.set(int(self.zoom_level * 100))
            self.zoom_slider_updating = False
        if self.zoom_level_label is not None:
            self.zoom_level_label.configure(text=f"{int(self.zoom_level * 100)}%")

    def _on_zoom_slider(self, value: str) -> None:
        if self.zoom_slider_updating:
            return
        try:
            pct = int(float(value))
        except ValueError:
            return
        self.zoom_level = max(0.1, min(5.0, pct / 100.0))
        if self.zoom_level_label is not None:
            self.zoom_level_label.configure(text=f"{pct}%")
        if self.original_frame is not None:
            self._apply_zoom_and_display(self.original_frame)

    def zoom_in(self) -> None:
        self._set_zoom(self.zoom_level + 0.1)

    def zoom_out(self) -> None:
        self._set_zoom(self.zoom_level - 0.1)

    def reset_zoom(self) -> None:
        self.zoom_level = 1.0
        self.focal_x = 0.5
        self.focal_y = 0.5
        self._sync_zoom_ui()
        if self.original_frame is not None:
            self._apply_zoom_and_display(self.original_frame)

    def fit_to_window(self) -> None:
        if self.video_width <= 0 or self.video_height <= 0:
            return
        cw, ch = self._get_canvas_dimensions()
        self.zoom_level = min(cw / self.video_width, ch / self.video_height)
        self.focal_x = 0.5
        self.focal_y = 0.5
        self._sync_zoom_ui()
        if self.original_frame is not None:
            self._apply_zoom_and_display(self.original_frame)

    # ---- Mouse / trackpad event handlers ----
    def _on_mouse_zoom(self, event: tk.Event) -> None:
        zoom_factor = 1.1
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            new_zoom = self.zoom_level * zoom_factor
        elif event.num == 5 or (hasattr(event, "delta") and event.delta < 0):
            new_zoom = self.zoom_level / zoom_factor
        else:
            return
        self._set_zoom_at_point(new_zoom, event.x, event.y)

    def _on_pinch_zoom(self, event: tk.Event) -> None:
        if hasattr(event, "delta"):
            factor = 1.05 if event.delta > 0 else 1 / 1.05
            self._set_zoom_at_point(self.zoom_level * factor, event.x, event.y)

    def _on_pan_start(self, event: tk.Event) -> None:
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_start_focal_x = self.focal_x
        self.drag_start_focal_y = self.focal_y
        self.is_dragging = True

    def _on_pan_move(self, event: tk.Event) -> None:
        if not self.is_dragging or self.original_frame is None:
            return
        zw = self.video_width * self.zoom_level
        zh = self.video_height * self.zoom_level
        if zw < 1 or zh < 1:
            return
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.focal_x = max(
            0.0, min(1.0, self.drag_start_focal_x - dx / zw)
        )
        self.focal_y = max(
            0.0, min(1.0, self.drag_start_focal_y - dy / zh)
        )
        self._apply_zoom_and_display(self.original_frame)

    def _on_pan_stop(self, event: tk.Event) -> None:
        self.is_dragging = False

    def _on_canvas_resize(self, event: tk.Event) -> None:
        if self.original_frame is not None and event.widget is self.video_canvas:
            self._apply_zoom_and_display(self.original_frame)

    # ------------------------------------------------------------------
    # Annotation workflow
    # ------------------------------------------------------------------
    def mark_current_state(self) -> None:
        if not self.capture or not self.state_sequence:
            return
        if self.current_state_index >= len(self.state_sequence):
            messagebox.showinfo(
                "All states marked", "All states have already been marked."
            )
            return
        self._push_undo()   # ← snapshot before change
        self.pause_video()
        start_frame = (
            0 if self.current_state_index == 0 else self.state_start_frame
        )
        end_frame = min(self.current_frame, self.total_frames - 1)
        if end_frame < start_frame:
            end_frame = start_frame
        label = self.state_sequence[self.current_state_index]
        self.recorded_segments = self.recorded_segments[: self.current_state_index]
        self.recorded_segments.append(Segment(start_frame, end_frame, label))
        self.current_state_index += 1
        self.state_start_frame = min(
            end_frame + 1, max(self.total_frames - 1, 0)
        )
        if self.current_state_index < len(self.state_sequence):
            self.seek_to_frame(self.state_start_frame)
        self._refresh_state_ui()
        self._update_annotation_view()
        self._refresh_binary_label_display()
        self._update_buttons()
        self._mark_unsaved()

    def undo_last_mark(self) -> None:
        """Restore the previous annotation state from the undo stack."""
        if not self._undo_stack:
            return
        import copy
        # Push current state onto redo before restoring
        self._redo_stack.append(copy.deepcopy(self.recorded_segments))
        self.recorded_segments = self._undo_stack.pop()

        self.pause_video()
        self.current_state_index = len(self.recorded_segments)
        self.state_start_frame = (
            0 if not self.recorded_segments
            else self.recorded_segments[-1].end + 1
        )
        self.state_start_frame = min(
            self.state_start_frame, max(self.total_frames - 1, 0)
        )
        if self.new_video_mode:
            has_finish = any(
                seg.label == "finish" for seg in self.recorded_segments
            )
            if not has_finish:
                self.binary_label = ""
        self.seek_to_frame(self.state_start_frame)
        self._refresh_state_ui()
        self._update_annotation_view()
        self._refresh_binary_label_display()
        self._update_undo_redo_buttons()
        self._update_buttons()
        self._mark_unsaved()
        self.status_var.set("Undo")

    def redo_last_mark(self) -> None:
        """Re-apply the next annotation state from the redo stack."""
        if not self._redo_stack:
            return
        import copy
        # Push current state onto undo before re-applying
        self._undo_stack.append(copy.deepcopy(self.recorded_segments))
        self.recorded_segments = self._redo_stack.pop()

        self.pause_video()
        self.current_state_index = len(self.recorded_segments)
        self.state_start_frame = (
            0 if not self.recorded_segments
            else self.recorded_segments[-1].end + 1
        )
        self.state_start_frame = min(
            self.state_start_frame, max(self.total_frames - 1, 0)
        )
        if self.new_video_mode:
            has_finish = any(
                seg.label == "finish" for seg in self.recorded_segments
            )
            if not has_finish:
                self.binary_label = ""
        self.seek_to_frame(self.state_start_frame)
        self._refresh_state_ui()
        self._update_annotation_view()
        self._refresh_binary_label_display()
        self._update_undo_redo_buttons()
        self._update_buttons()
        self._mark_unsaved()
        self.status_var.set("Redo")

    def clear_annotations(self) -> None:
        if not self.state_sequence and not self.new_video_mode:
            return
        if not messagebox.askyesno(
            "Clear annotations", "Discard all marks for this sample?"
        ):
            return
        self._push_undo()   # ← snapshot before change
        self.pause_video()
        # ... rest unchanged ...
        self.pause_video()
        self.recorded_segments.clear()
        self.current_state_index = 0
        self.state_start_frame   = 0
        # In new_video_mode reset binary_label so manual buttons reappear.
        if self.new_video_mode:
            self.binary_label = ""
        self.seek_to_frame(0)
        self._refresh_state_ui()
        self._update_annotation_view()
        self._refresh_binary_label_display()
        self._update_buttons()
        self._mark_unsaved()

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------
    def _refresh_state_ui(self) -> None:
        mode_labels = {
            "classic":  "Classic State Sequence",
            "boundary": "Rep Boundary Mode",
            "manual":   "Manual Windowing Mode",
            "flexible": "Flexible Binary Labels",
        }
        mode_name = mode_labels.get(self.annotation_mode, "")

        if self.annotation_mode == "classic":
            if not self.state_sequence:
                self.current_state_var.set(
                    f"[{mode_name}]  No sample loaded"
                )
                self.sequence_var.set("")
                return
            if self.current_state_index < len(self.state_sequence):
                label = self.state_sequence[self.current_state_index]
                self.current_state_var.set(
                    f"[{mode_name}]  Current state: {label} — mark its end"
                )
            else:
                self.current_state_var.set(
                    f"[{mode_name}]  All states marked. Ready to save."
                )
            decorated = []
            for idx, label in enumerate(self.state_sequence):
                if idx < self.current_state_index:
                    decorated.append(f"✓ {label}")
                elif idx == self.current_state_index:
                    decorated.append(f"→ {label}")
                else:
                    decorated.append(label)
            self.sequence_var.set(" | ".join(decorated))

        elif self.annotation_mode == "boundary":
            if self.awaiting_rep_end:
                self.current_state_var.set(
                    f"[{mode_name}]  "
                    f"Awaiting rep end (start: frame {self.current_rep_start})"
                )
            else:
                count = len([
                    s for s in self.recorded_segments
                    if s.label in ("rep", "no-rep")
                ])
                self.current_state_var.set(
                    f"[{mode_name}]  {count} segment(s) marked — "
                    "click 'Mark rep start' to add another"
                )
            self.sequence_var.set("")

        elif self.annotation_mode == "manual":
            seg_count = len(self.recorded_segments)
            self.current_state_var.set(
                f"[{mode_name}]  {seg_count} segment(s) — "
                "use the marking buttons to add segments"
            )
            self.sequence_var.set("")

        elif self.annotation_mode == "flexible":
            seg_count = len(self.recorded_segments)
            self.current_state_var.set(
                f"[{mode_name}]  {seg_count} segment(s) — "
                "edit segments and binary label freely"
            )
            self.sequence_var.set("")

    def _update_annotation_view(self) -> None:
        for child in self.annotation_tree.get_children():
            self.annotation_tree.delete(child)
        for seg in self.recorded_segments:
            self.annotation_tree.insert(
                "", tk.END,
                values=("▶", seg.start, seg.end, seg.label),
            )

    def _update_buttons(self) -> None:
        has_video    = self.capture is not None
        has_segments = bool(self.recorded_segments)
        can_save     = has_video and has_segments

        # Which buttons are valid depends on mode
        can_mark_classic = (
            has_video
            and self.annotation_mode == "classic"
            and self.current_state_index < len(self.state_sequence)
        )
        can_mark_boundary = (
            has_video and self.annotation_mode == "boundary"
        )
        can_mark_manual = (
            has_video and self.annotation_mode in ("manual", "flexible")
        )

        self.play_button.config(state="normal" if has_video else "disabled")
        self.rt_play_button.config(state="normal" if has_video else "disabled")

        # Button sets for each mode
        manual_btns = [
            self.mark_prep_button, self.mark_rep_button,
            self.mark_norep_button, self.mark_finish_button,
        ]

        # Hide everything first then show what is needed
        all_special = manual_btns + [
            self.mark_button, self.mark_rep_boundary_button
        ]
        for btn in all_special:
            if btn in self._ctrl_wrap._children:
                self._ctrl_wrap._children.remove(btn)
            btn.place_forget()

        if self.annotation_mode == "classic":
            if self.mark_button not in self._ctrl_wrap._children:
                idx = self._ctrl_wrap._children.index(self.undo_button)
                self._ctrl_wrap._children.insert(idx, self.mark_button)
            self.mark_button.config(
                state="normal" if can_mark_classic else "disabled"
            )

        elif self.annotation_mode == "boundary":
            if self.mark_rep_boundary_button not in self._ctrl_wrap._children:
                idx = self._ctrl_wrap._children.index(self.undo_button)
                self._ctrl_wrap._children.insert(
                    idx, self.mark_rep_boundary_button
                )
            self.mark_rep_boundary_button.config(
                state="normal" if can_mark_boundary else "disabled",
                text=(
                    "Mark rep end"
                    if self.awaiting_rep_end
                    else "Mark rep start"
                ),
            )

        elif self.annotation_mode in ("manual", "flexible"):
            for btn in manual_btns:
                if btn not in self._ctrl_wrap._children:
                    idx = self._ctrl_wrap._children.index(self.undo_button)
                    self._ctrl_wrap._children.insert(idx, btn)
            for btn in manual_btns:
                btn.config(state="normal" if can_mark_manual else "disabled")

        self.undo_button.config(
            state="normal" if has_segments else "disabled"
        )
        self.clear_button.config(
            state="normal" if has_segments else "disabled"
        )
        self.save_button.config(
            state="normal" if can_save else "disabled"
        )
        self._update_undo_redo_buttons()

        def _deferred_reflow(attempt: int = 0) -> None:
            self._ctrl_wrap.update_idletasks()
            w = self._ctrl_wrap.winfo_width()
            if w <= 1:
                w = self.controls_panel.winfo_width()
            if w <= 1:
                w = self.root.winfo_width()
            if w <= 1:
                if attempt < 20:
                    self.root.after(20, lambda: _deferred_reflow(attempt + 1))
                return
            self._ctrl_wrap._reflow(w)

        self.root.after_idle(lambda: _deferred_reflow(0))

    # ------------------------------------------------------------------
    # New video loading functionality
    # ------------------------------------------------------------------
    def load_new_video(self) -> None:
        video_path = filedialog.askopenfilename(
            title="Select video file",
            initialdir=self._last_browse_dir,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*"),
            ],
        )
        if not video_path:
            return
        video_path = Path(video_path)
        if not video_path.exists():
            messagebox.showerror(
                "Video not found", f"Video file not found: {video_path}"
            )
            return

        # Remember the folder this video came from for next time
        self._last_browse_dir = str(video_path.parent)

        self.pause_video()
        self.close_video()
        if not self._open_video(video_path):
            messagebox.showerror("Video load error", "Failed to open video")
            return
        self.new_video_mode = True
        self.new_video_path = video_path
        self.binary_label = ""  # stays empty until "finish" is marked
        self.state_sequence = []
        self.recorded_segments = []
        self.awaiting_rep_end  = False
        self.current_rep_start = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.sample_json_paths = []
        self.current_state_index = 0
        self.state_start_frame = 0
        self.seek_to_frame(0)
        self.status_var.set(f"Loaded new video: {video_path.name}")
        self._refresh_state_ui()
        self._update_annotation_view()
        self._refresh_binary_label_display()
        self._update_buttons()

        self.root.after(150, self._update_buttons)

        messagebox.showinfo(
            "New video loaded",
            "You can now either:\n\n"
            "1. Enter a binary label (0s and 1s) and follow the normal workflow, OR\n"
            "2. Use the manual windowing buttons (Mark as prep/rep/no-rep/finish) "
            "to create segments directly",
        )

    def mark_manual_segment(self, label: str) -> None:
        """Mark a segment in Manual Windowing or Flexible Binary Labels mode."""
        if not self.capture:
            messagebox.showwarning("No video", "Please load a video first.")
            return
        start_frame = (
            0 if not self.recorded_segments
            else self.recorded_segments[-1].end + 1
        )
        end_frame = self.current_frame
        if end_frame <= start_frame:
            messagebox.showwarning(
                "Invalid segment", "End frame must be after start frame."
            )
            return
        self._push_undo()
        new_segment = Segment(start_frame, end_frame, label)
        self.recorded_segments.append(new_segment)
        self._update_annotation_view()
        self._refresh_binary_label_display()
        self._update_buttons()
        self._refresh_state_ui()
        self.status_var.set(
            f"Added {label} segment: {start_frame}–{end_frame}"
        )
        self._mark_unsaved()
        if label == "finish":
            self._derive_binary_from_segments()

    def _derive_binary_from_segments(self) -> None:
        """Derive and commit binary label from manually created segments.
        Called only when the user marks a 'finish' segment in new_video_mode,
        at which point the annotation is considered complete and the binary
        label is locked in."""
        if not self.recorded_segments:
            return
        middle_segments = [
            seg for seg in self.recorded_segments
            if seg.label in ("rep", "no-rep")
        ]
        middle_segments.sort(key=lambda s: s.start)
        binary_label = "".join(
            "1" if seg.label == "rep" else "0"
            for seg in middle_segments
        )
        # Commit the binary label now that annotation is complete
        self.binary_label = binary_label
        self._refresh_binary_label_display()
        self.status_var.set(f"Derived binary label: {binary_label}")

    def rename_sample(self) -> None:
        """Rename the currently loaded sample — its folder, all JSON files
        inside it, the video file, and every reference inside those JSON
        files, then update video_config.json / .csv."""
        if not self.sample_json_paths:
            messagebox.showwarning(
                "No sample loaded",
                "Please load a sample before trying to rename it."
            )
            return
        if not self.json_root:
            messagebox.showwarning(
                "No root folder",
                "Please select a json_keypoints folder first."
            )
            return

        # Current sample folder:  json_keypoints/<exercise>/<sample>/
        sample_dir   = self.sample_json_paths[0].parent
        exercise_dir = sample_dir.parent
        old_sample   = sample_dir.name          # e.g. "push_ups_person1_cam1"
        exercise     = exercise_dir.name        # e.g. "push_ups"

        # Read current video filename from the first JSON so we can
        # rename the video file and update the path reference.
        try:
            with self.sample_json_paths[0].open("r", encoding="utf-8") as f:
                primary_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Read error", f"Could not read sample JSON: {e}")
            return

        old_video_path_str = primary_data.get("video_path", "")
        old_video_filename = Path(old_video_path_str).name if old_video_path_str \
            else f"{old_sample}.mp4"

        # Warn about unsaved changes
        if self._unsaved_changes:
            if not messagebox.askyesno(
                "Unsaved changes",
                "You have unsaved changes. Renaming will save the current "
                "annotations as part of the renamed files.\n\nContinue?"
            ):
                return

        # Show the rename dialog
        dialog = SampleRenameDialog(
            self.root,
            exercise=exercise,
            old_sample=old_sample,
        )
        if not dialog.result:
            return

        angle_person, rep_norep = dialog.result
        new_sample     = f"{exercise}_{angle_person}_{rep_norep}"
        new_video_name = f"{new_sample}.mp4"

        if new_sample == old_sample:
            messagebox.showinfo("No change", "The new name is the same as the old name.")
            return

        new_sample_dir = exercise_dir / new_sample

        if new_sample_dir.exists():
            messagebox.showerror(
                "Name conflict",
                f"A sample named '{new_sample}' already exists.\n"
                "Please choose a different name."
            )
            return

        # ------------------------------------------------------------------
        # 1. Rename the video file on disk
        # ------------------------------------------------------------------
        dataset_root      = self.json_root.parent
        old_video_abspath = dataset_root / old_video_filename
        new_video_abspath = dataset_root / new_video_name

        video_renamed = False
        if old_video_abspath.exists():
            try:
                old_video_abspath.rename(new_video_abspath)
                video_renamed = True
            except Exception as e:
                messagebox.showerror(
                    "Rename error",
                    f"Could not rename video file:\n{e}"
                )
                return
        else:
            # Video file not found at expected location — warn but continue
            messagebox.showwarning(
                "Video file not found",
                f"Could not find '{old_video_filename}' at:\n{old_video_abspath}\n\n"
                "JSON files and config will still be updated."
            )

        # ------------------------------------------------------------------
        # 2. Rename each JSON file inside the sample folder and update
        #    its internal video_path / exercise_type references
        # ------------------------------------------------------------------
        updated_json_paths: list[Path] = []
        annotations = [seg.as_dict() for seg in self.recorded_segments]

        for old_json_path in self.sample_json_paths:
            try:
                with old_json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue

                # Update internal references
                data["video_path"]    = f"CFRep/CFRep/{new_video_name}"
                data["exercise_type"] = exercise.upper()
                data["binary_label"]  = self.binary_label
                data["annotations"]   = annotations

                # Derive a new filename by replacing old_sample with new_sample
                old_stem    = old_json_path.stem    # e.g. push_ups_p1_cam1_minimal
                new_stem    = old_stem.replace(old_sample, new_sample, 1)
                new_json_fn = new_stem + ".json"

                # Write to the same folder under the new filename
                new_json_path = old_json_path.parent / new_json_fn
                with new_json_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                # Remove old file only if new path differs
                if new_json_path != old_json_path:
                    old_json_path.unlink()

                updated_json_paths.append(new_json_path)

            except Exception as e:
                messagebox.showwarning(
                    "JSON update warning",
                    f"Could not update {old_json_path.name}:\n{e}"
                )

        # ------------------------------------------------------------------
        # 3. Rename the sample folder itself
        # ------------------------------------------------------------------
        try:
            sample_dir.rename(new_sample_dir)
        except Exception as e:
            messagebox.showerror(
                "Folder rename error",
                f"Could not rename sample folder:\n{e}"
            )
            # Undo video rename if folder rename failed
            if video_renamed and new_video_abspath.exists():
                try:
                    new_video_abspath.rename(old_video_abspath)
                except Exception:
                    pass
            return

        # ------------------------------------------------------------------
        # 4. Update video_config.json / .csv
        #    Remove the old entry and upsert the new one
        # ------------------------------------------------------------------
        self._remove_video_config_entry(old_video_filename)

        if updated_json_paths:
            # Re-read the first successfully written JSON for config data
            final_json = new_sample_dir / updated_json_paths[0].name
            try:
                with final_json.open("r", encoding="utf-8") as f:
                    final_data = json.load(f)
                self._update_video_configs(new_video_name, exercise, final_data)
            except Exception as e:
                print(f"Warning: could not update video_config after rename: {e}")

        # ------------------------------------------------------------------
        # 5. Refresh the in-memory state to point at the renamed files
        # ------------------------------------------------------------------
        self.sample_json_paths = [
            new_sample_dir / p.name for p in updated_json_paths
        ]
        self._unsaved_changes = False
        self._mark_saved()

        # Refresh the exercise/sample lists in the sidebar
        self.populate_samples(exercise)
        # Re-select the renamed sample in the list
        for i in range(self.sample_list.size()):
            if self.sample_list.get(i) == new_sample:
                self.sample_list.selection_clear(0, tk.END)
                self.sample_list.selection_set(i)
                self.sample_list.see(i)
                break

        self.status_var.set(
            f"Sample renamed: '{old_sample}' → '{new_sample}'"
        )
        messagebox.showinfo(
            "Rename successful",
            f"Sample renamed to: {new_sample}\n"
            f"Video renamed to:  {new_video_name}\n"
            f"video_config.json updated."
        )

    def _remove_video_config_entry(self, filename: str) -> None:
        """Remove ALL entries for *filename* from video_config.json
        and video_config.csv (handles pre-existing duplicates too)."""
        if not self.json_root:
            return
        config_dir       = self.json_root.parent
        json_config_path = config_dir / "video_config.json"
        csv_config_path  = config_dir / "video_config.csv"

        # JSON
        if json_config_path.exists():
            try:
                with json_config_path.open("r", encoding="utf-8") as f:
                    config_list = json.load(f)
                if isinstance(config_list, list):
                    before = len(config_list)
                    config_list = [
                        e for e in config_list
                        if e.get("filename") != filename
                    ]
                    after = len(config_list)
                    with json_config_path.open("w", encoding="utf-8") as f:
                        json.dump(config_list, f, indent=2)
                    print(
                        f"video_config.json: removed {before - after} "
                        f"entry/entries for '{filename}'"
                    )
            except Exception as e:
                print(
                    f"Warning: could not remove entry from "
                    f"video_config.json: {e}"
                )

        # CSV
        if csv_config_path.exists():
            try:
                with csv_config_path.open("r", encoding="utf-8") as f:
                    rows = f.readlines()
                new_rows: list[str] = []
                removed = 0
                for row in rows:
                    stripped = row.strip()
                    if not stripped:
                        new_rows.append(row)
                        continue
                    parts = stripped.split(",")
                    if parts[0].strip() == filename:
                        removed += 1
                        continue
                    new_rows.append(row)
                with csv_config_path.open("w", encoding="utf-8") as f:
                    f.writelines(new_rows)
                print(
                    f"video_config.csv: removed {removed} "
                    f"row(s) for '{filename}'"
                )
            except Exception as e:
                print(
                    f"Warning: could not remove entry from "
                    f"video_config.csv: {e}"
                )
    # ------------------------------------------------------------------
    # Save annotations
    # ------------------------------------------------------------------
    def save_annotations(self) -> None:
        if not self.capture or not self.recorded_segments:
            messagebox.showwarning("Nothing to save", "No annotations to save.")
            return
        if self.new_video_mode:
            self._save_new_video_annotations()
        else:
            self._save_existing_annotations()

    def _save_new_video_annotations(self) -> None:
        if not self.new_video_path:
            messagebox.showerror("Missing data", "No video path found for saving.")
            return
        naming_dialog = VideoNamingDialog(self.root)
        if not naming_dialog.result:
            return
        exercise, angle_person, rep_norep = naming_dialog.result
        new_video_name = f"{exercise}_{angle_person}_{rep_norep}.mp4"
        if self.json_root:
            dataset_root = self.json_root.parent
        else:
            dataset_root = filedialog.askdirectory(
                title="Select dataset root (CFRep/CFRep)"
            )
            if not dataset_root:
                return
            dataset_root = Path(dataset_root)
        video_dest = dataset_root / new_video_name
        json_dir = (
            dataset_root
            / "json_keypoints"
            / exercise
            / f"{exercise}_{angle_person}_{rep_norep}"
        )
        try:
            json_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(self.new_video_path, video_dest)
            json_data = {
                "video_path": f"CFRep/CFRep/{new_video_name}",
                "dataset_type": "CocoDataset",
                "exercise_type": exercise.upper(),
                "binary_label": self.binary_label,
                "frames": [],
                "annotations": [
                    seg.as_dict() for seg in self.recorded_segments
                ],
            }
            json_path = (
                json_dir
                / f"{exercise}_{angle_person}_{rep_norep}_minimal.json"
            )
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
            self._update_video_configs(new_video_name, exercise, json_data)
            messagebox.showinfo(
                "Save successful",
                f"Video saved as: {new_video_name}\n"
                f"JSON created:   {json_path.name}\n"
                f"Video configs updated.",
            )
            self._mark_saved()
            self.new_video_mode = False
            self.new_video_path = None
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save: {e}")

    def _ask_save_mode(self) -> Optional[str]:
        """Ask the user whether to overwrite the existing file or save as new.

        Returns
        -------
        'overwrite' — replace the original JSON file(s)
        'new'       — save to a brand-new file chosen by the user
        None        — user cancelled
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Annotations")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 175
        y = (dialog.winfo_screenheight() // 2) - 80
        dialog.geometry(f"350x160+{x}+{y}")

        result: list[Optional[str]] = [None]  # mutable container for the choice

        ttk.Label(
            dialog,
            text="How would you like to save your annotations?",
            wraplength=310,
            padding=(16, 14, 16, 6),
        ).pack()

        btn_frame = ttk.Frame(dialog, padding=(16, 8))
        btn_frame.pack(fill=tk.X)

        def choose(value: str) -> None:
            result[0] = value
            dialog.destroy()

        ttk.Button(
            btn_frame,
            text="Overwrite existing file(s)",
            command=lambda: choose("overwrite"),
        ).grid(row=0, column=0, padx=6, pady=4, sticky="ew")

        ttk.Button(
            btn_frame,
            text="Save as new file…",
            command=lambda: choose("new"),
        ).grid(row=0, column=1, padx=6, pady=4, sticky="ew")

        ttk.Button(
            btn_frame,
            text="Cancel",
            command=dialog.destroy,
        ).grid(row=1, column=0, columnspan=2, pady=(0, 4))

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        dialog.wait_window()
        return result[0]
    
    def _sync_video_config_for_existing(
        self, json_data: dict, json_path: Path
    ) -> None:
        """Update video_config.json (and .csv) after saving an existing
        sample's annotations.

        Derives the filename and exercise name from the saved JSON data
        and the path on disk, then delegates to _update_video_configs.
        """
        if not self.json_root:
            return

        # Derive the video filename from the saved data, falling back to
        # the sample directory name if video_path is absent.
        video_path_val = json_data.get("video_path")
        if video_path_val:
            filename = Path(str(video_path_val)).name
        else:
            # json_path is  …/json_keypoints/<exercise>/<sample>/<file>.json
            # so the sample folder name makes a reasonable fallback filename
            filename = json_path.parent.name + ".mp4"

        # Derive the exercise name from the folder two levels above the JSON
        # i.e.  json_keypoints / <exercise> / <sample> / file.json
        try:
            exercise = json_path.parent.parent.name
        except Exception:
            exercise = json_data.get("exercise_type", "unknown")

        try:
            self._update_video_configs(filename, exercise, json_data)
            self.status_var.set(
                self.status_var.get() + " | video_config.json updated"
            )
        except Exception as e:
            print(f"Warning: failed to update video_config.json: {e}")

    def _save_existing_annotations(self) -> None:
        if not self.sample_json_paths:
            messagebox.showerror("No files", "No JSON files loaded to save to.")
            return

        save_choice = self._ask_save_mode()
        if save_choice is None:
            return

        annotations = [seg.as_dict() for seg in self.recorded_segments]

        if save_choice == "overwrite":
            success_count = 0
            skipped_count = 0
            saved_data    = None
            saved_path    = None

            for json_path in self.sample_json_paths:
                try:
                    with json_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        skipped_count += 1
                        continue
                    data["annotations"]  = annotations
                    data["binary_label"] = self.binary_label
                    with json_path.open("w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    success_count += 1
                    # Keep a reference to the first successfully saved
                    # file so we can update video_config.json afterwards
                    if saved_data is None:
                        saved_data = data
                        saved_path = json_path
                except Exception:
                    skipped_count += 1
                    continue

            if success_count > 0:
                # --- Sync video_config.json ---
                if saved_data is not None and saved_path is not None:
                    self._sync_video_config_for_existing(saved_data, saved_path)

                msg = f"Saved annotations to {success_count} JSON file(s)."
                if skipped_count:
                    msg += f"\n({skipped_count} file(s) skipped — not a valid dict)"
                self.status_var.set(f"Annotations saved to {success_count} files")
                messagebox.showinfo("Save complete", msg)
                self._mark_saved()
            else:
                messagebox.showerror(
                    "Save failed",
                    "Failed to save annotations to any files.\n"
                    "All target files may be invalid or inaccessible."
                )

        elif save_choice == "new":
            dest_dir = filedialog.askdirectory(
                title="Select folder to save new annotation file"
            )
            if not dest_dir:
                return
            dest_dir = Path(dest_dir)
            original_name  = self.sample_json_paths[0].stem
            suggested_name = f"{original_name}_copy.json"
            new_filename = simpledialog.askstring(
                "New filename",
                "Enter filename for the new annotation file:",
                initialvalue=suggested_name,
                parent=self.root,
            )
            if not new_filename:
                return
            if not new_filename.endswith(".json"):
                new_filename += ".json"
            new_path = dest_dir / new_filename
            if new_path.exists():
                if not messagebox.askyesno(
                    "File exists",
                    f"{new_filename} already exists. Overwrite it?",
                ):
                    return
            try:
                with self.sample_json_paths[0].open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    messagebox.showerror(
                        "Invalid source",
                        "The source JSON file does not contain a valid object."
                    )
                    return
                data["annotations"]  = annotations
                data["binary_label"] = self.binary_label
                with new_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                # --- Sync video_config.json for the new file too ---
                self._sync_video_config_for_existing(data, new_path)

                self.status_var.set(
                    f"Annotations saved as new file: {new_filename}"
                )
                messagebox.showinfo(
                    "Save complete", f"New annotation file saved:\n{new_path}"
                )
                self._mark_saved()
            except Exception as e:
                messagebox.showerror("Save error", f"Failed to save new file: {e}")

    def deduplicate_video_configs(self) -> None:
        """Scan video_config.json and video_config.csv for duplicate
        filename entries and remove them, keeping only the last
        occurrence of each (which has the most recent data).
        Shows a summary of what was cleaned up."""
        if not self.json_root:
            messagebox.showwarning(
                "No folder",
                "Please select a json_keypoints folder first."
            )
            return

        config_dir       = self.json_root.parent
        json_config_path = config_dir / "video_config.json"
        csv_config_path  = config_dir / "video_config.csv"

        json_dupes = 0
        csv_dupes  = 0

        # ------------------------------------------------------------------
        # JSON deduplication
        # ------------------------------------------------------------------
        if json_config_path.exists():
            try:
                with json_config_path.open("r", encoding="utf-8") as f:
                    config_list = json.load(f)

                if isinstance(config_list, list):
                    # Walk in reverse: first seen (from the end) wins
                    seen_fns: set[str] = set()
                    deduped: list[dict] = []
                    for entry in reversed(config_list):
                        fn = entry.get("filename", "")
                        if fn in seen_fns:
                            json_dupes += 1
                            continue
                        seen_fns.add(fn)
                        deduped.append(entry)
                    # Restore original forward order and sort
                    deduped.reverse()
                    deduped.sort(key=lambda x: x.get("filename", ""))

                    if json_dupes > 0:
                        with json_config_path.open("w", encoding="utf-8") as f:
                            json.dump(deduped, f, indent=2)
                        print(
                            f"video_config.json: removed {json_dupes} duplicate(s)"
                        )
                    else:
                        print("video_config.json: no duplicates found")

            except Exception as e:
                messagebox.showerror(
                    "JSON error",
                    f"Could not process video_config.json:\n{e}"
                )
                return

        # ------------------------------------------------------------------
        # CSV deduplication
        # ------------------------------------------------------------------
        if csv_config_path.exists():
            try:
                with csv_config_path.open("r", encoding="utf-8") as f:
                    rows = f.readlines()

                # Walk in reverse: first seen (from the end) wins
                seen_fns_csv: set[str] = set()
                deduped_rows: list[str] = []
                for row in reversed(rows):
                    stripped = row.strip()
                    if not stripped:
                        deduped_rows.append(row)
                        continue
                    parts = stripped.split(",")
                    key = parts[0].strip()
                    if key in seen_fns_csv:
                        csv_dupes += 1
                        continue
                    seen_fns_csv.add(key)
                    deduped_rows.append(row)

                deduped_rows.reverse()

                if csv_dupes > 0:
                    with csv_config_path.open("w", encoding="utf-8") as f:
                        f.writelines(deduped_rows)
                    print(
                        f"video_config.csv: removed {csv_dupes} duplicate(s)"
                    )
                else:
                    print("video_config.csv: no duplicates found")

            except Exception as e:
                messagebox.showerror(
                    "CSV error",
                    f"Could not process video_config.csv:\n{e}"
                )
                return

        # Summary
        total = json_dupes + csv_dupes
        if total > 0:
            messagebox.showinfo(
                "Deduplication complete",
                f"Removed {json_dupes} duplicate(s) from video_config.json\n"
                f"Removed {csv_dupes} duplicate(s) from video_config.csv"
            )
        else:
            messagebox.showinfo(
                "Deduplication complete",
                "No duplicate entries found in either config file."
            )
        self.status_var.set(
            f"Config deduplication done — "
            f"{json_dupes} JSON + {csv_dupes} CSV duplicates removed"
        )

    def _update_video_configs(
        self, filename: str, exercise: str, json_data: dict
    ) -> None:
        """Upsert the entry for *filename* in both video_config.json
        and video_config.csv, deduplicating both files on every write."""
        if not self.json_root:
            return

        config_dir       = self.json_root.parent
        json_config_path = config_dir / "video_config.json"
        csv_config_path  = config_dir / "video_config.csv"

        segments  = json_data.get("annotations", [])
        rep_count = sum(1 for s in segments if s.get("label") == "rep")
        binary    = json_data.get("binary_label", "")

        new_entry = {
            "filename":     filename,
            "exercise":     exercise,
            "binary_label": binary,
            "rep_count":    rep_count,
            "segments":     segments,
        }

        # ------------------------------------------------------------------
        # JSON — read, deduplicate by filename, upsert, sort, write
        # ------------------------------------------------------------------
        config_list: list[dict] = []
        if json_config_path.exists():
            try:
                with json_config_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    config_list = loaded
            except Exception as e:
                print(f"Warning: could not read video_config.json: {e}")

        # Deduplicate: keep only the last occurrence of each filename,
        # then we will upsert our new entry on top.
        seen: dict[str, int] = {}
        deduped: list[dict] = []
        for entry in config_list:
            fn = entry.get("filename", "")
            if fn in seen:
                # Remove the earlier duplicate
                deduped[seen[fn]] = None    # type: ignore[call-overload]
            seen[fn] = len(deduped)
            deduped.append(entry)
        config_list = [e for e in deduped if e is not None]

        # Upsert our new entry
        updated = False
        for i, entry in enumerate(config_list):
            if entry.get("filename") == filename:
                config_list[i] = new_entry
                updated = True
                break
        if not updated:
            config_list.append(new_entry)

        config_list.sort(key=lambda x: x.get("filename", ""))

        try:
            json_config_path.parent.mkdir(parents=True, exist_ok=True)
            with json_config_path.open("w", encoding="utf-8") as f:
                json.dump(config_list, f, indent=2)
            print(f"video_config.json updated for: {filename}")
        except Exception as e:
            print(f"Failed to write video_config.json: {e}")

        # ------------------------------------------------------------------
        # CSV — read, deduplicate by first field, upsert, write
        # ------------------------------------------------------------------
        new_csv_row = f"{filename},{exercise},{rep_count},{binary}\n"

        existing_rows: list[str] = []
        if csv_config_path.exists():
            try:
                with csv_config_path.open("r", encoding="utf-8") as f:
                    existing_rows = f.readlines()
            except Exception as e:
                print(f"Warning: could not read video_config.csv: {e}")

        # Deduplicate: build an ordered dict keyed by filename field,
        # last writer wins (keeps most recent data).
        csv_dict: dict[str, str] = {}
        blank_rows: list[str] = []
        for row in existing_rows:
            stripped = row.strip()
            if not stripped:
                blank_rows.append(row)
                continue
            parts = stripped.split(",")
            key = parts[0].strip()
            csv_dict[key] = row   # later row overwrites earlier duplicate

        # Upsert our new row
        csv_dict[filename] = new_csv_row

        # Rebuild: sort by filename key, preserve a single trailing newline
        new_rows = [csv_dict[k] for k in sorted(csv_dict.keys())]

        try:
            csv_config_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_config_path.open("w", encoding="utf-8") as f:
                f.writelines(new_rows)
            print(f"video_config.csv updated for: {filename}")
        except Exception as e:
            print(f"Failed to write video_config.csv: {e}")

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------
    def close_video(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self._stop_realtime_play()
        self.current_frame = 0
        self.total_frames = 0
        self.video_width = 0
        self.video_height = 0
        self.original_frame = None
        self.display_image = None
        self.video_canvas.delete("video")
        self.frame_info_var.set("Frame: - / -")
        self._last_read_frame = -1

    def on_close(self) -> None:
        if self._unsaved_changes and self.recorded_segments:
            response = messagebox.askyesnocancel(
                "Unsaved changes",
                "You have unsaved annotation changes.\n\n"
                "Do you want to save before exiting?",
                parent=self.root,
            )
            if response is None:
                # Cancel — do not close
                return
            if response:
                # Yes — attempt save, only close if save completes
                self.save_annotations()
                # If segments still flagged dirty the save was cancelled
                if self._unsaved_changes:
                    return
        self.pause_video()
        self.close_video()
        self.root.destroy()

    def _mark_unsaved(self) -> None:
        """Call whenever annotations are modified but not yet saved."""
        self._unsaved_changes = True
        # Reflect dirty state in the window title
        title = self.root.title()
        if not title.startswith("* "):
            self.root.title(f"* {title}")

    def _mark_saved(self) -> None:
        """Call after a successful save to clear the dirty flag."""
        self._unsaved_changes = False
        # Remove the dirty asterisk from the window title
        title = self.root.title()
        if title.startswith("* "):
            self.root.title(title[2:])

    def _push_undo(self) -> None:
        """Snapshot the current segments onto the undo stack and clear redo."""
        import copy
        self._undo_stack.append(copy.deepcopy(self.recorded_segments))
        self._redo_stack.clear()
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self) -> None:
        """Sync the enabled/disabled state of undo and redo buttons
        and their menu entries."""
        can_undo = bool(self._undo_stack)
        can_redo = bool(self._redo_stack)
        self.undo_button.config(
            state="normal" if can_undo else "disabled"
        )
        self.redo_button.config(
            state="normal" if can_redo else "disabled"
        )
        # Menu entries
        if hasattr(self, "_edit_menu"):
            self._edit_menu.entryconfig(
                "Undo", state="normal" if can_undo else "disabled"
            )
            self._edit_menu.entryconfig(
                "Redo", state="normal" if can_redo else "disabled"
            )

# ======================================================================
# Dialog classes
# ======================================================================

class WrapFrame(tk.Frame):
    """A frame that arranges child widgets in left-to-right rows and
    automatically wraps them onto the next row when there is not enough
    horizontal space.  Use add() to append widgets instead of packing
    or gridding them directly.
    """

    def __init__(self, parent, padx: int = 2, pady: int = 2, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._children: list[tk.Widget] = []
        self._padx = padx
        self._pady = pady
        self.bind("<Configure>", self._on_configure)

    def add(self, widget: tk.Widget) -> tk.Widget:
        """Register *widget* as a managed child and return it."""
        self._children.append(widget)
        return widget

    def _on_configure(self, event: tk.Event) -> None:
        self._reflow(event.width)

    def _reflow(self, max_width: int) -> None:
        if max_width <= 1:
            return
        # Place every child, wrapping when the row would exceed max_width
        x = self._padx
        y = self._pady
        row_height = 0

        for widget in self._children:
            widget.update_idletasks()
            w = widget.winfo_reqwidth()
            h = widget.winfo_reqheight()

            if x + w + self._padx > max_width and x > self._padx:
                # Wrap to next row
                x = self._padx
                y += row_height + self._pady
                row_height = 0

            widget.place(x=x, y=y)
            x += w + self._padx
            row_height = max(row_height, h)

        # Tell the frame how tall it needs to be to show all rows
        total_height = y + row_height + self._pady
        self.configure(height=total_height)
        
class SampleRenameDialog:
    """Dialog for renaming an existing sample."""

    def __init__(self, parent, exercise: str, old_sample: str):
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Rename Sample")
        self.dialog.geometry("420x280")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth()  // 2) - 210
        y = (self.dialog.winfo_screenheight() // 2) - 140
        self.dialog.geometry(f"+{x}+{y}")

        main = ttk.Frame(self.dialog, padding=20)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)

        ttk.Label(
            main,
            text="Rename Sample",
            font=("TkDefaultFont", 11, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        # Current name (read-only)
        ttk.Label(main, text="Current name:").grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Label(
            main, text=old_sample, foreground="#666"
        ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=3)

        # Exercise (read-only)
        ttk.Label(main, text="Exercise:").grid(
            row=2, column=0, sticky="w", pady=3
        )
        ttk.Label(
            main, text=exercise, foreground="#666"
        ).grid(row=2, column=1, sticky="w", padx=(10, 0), pady=3)

        # New angle_personID
        ttk.Label(main, text="New angle_personID:").grid(
            row=3, column=0, sticky="w", pady=3
        )
        self.angle_person_var = tk.StringVar()
        angle_person_entry = ttk.Entry(
            main, textvariable=self.angle_person_var, width=22
        )
        angle_person_entry.grid(
            row=3, column=1, sticky="ew", padx=(10, 0), pady=3
        )

        # New rep_norep
        ttk.Label(main, text="New rep_norep:").grid(
            row=4, column=0, sticky="w", pady=3
        )
        self.rep_norep_var = tk.StringVar()
        rep_norep_entry = ttk.Entry(
            main, textvariable=self.rep_norep_var, width=22
        )
        rep_norep_entry.grid(
            row=4, column=1, sticky="ew", padx=(10, 0), pady=3
        )

        # Live preview
        self.preview_var = tk.StringVar(value="")
        ttk.Label(
            main,
            textvariable=self.preview_var,
            foreground="#0055cc",
            font=("Courier", 9),
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.angle_person_var.trace_add(
            "write", lambda *_: self._update_preview(exercise)
        )
        self.rep_norep_var.trace_add(
            "write", lambda *_: self._update_preview(exercise)
        )

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(16, 0)
        )
        ttk.Button(
            btn_frame, text="Rename",
            command=lambda: self._ok(exercise),
        ).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(
            btn_frame, text="Cancel",
            command=self.dialog.destroy,
        ).grid(row=0, column=1)

        angle_person_entry.focus()
        self.dialog.wait_window()

    def _update_preview(self, exercise: str) -> None:
        angle_person = self.angle_person_var.get().strip()
        rep_norep    = self.rep_norep_var.get().strip()
        if angle_person or rep_norep:
            self.preview_var.set(
                f"New name: {exercise}_{angle_person}_{rep_norep}"
            )
        else:
            self.preview_var.set("")

    def _ok(self, exercise: str) -> None:
        angle_person = self.angle_person_var.get().strip()
        rep_norep    = self.rep_norep_var.get().strip()
        if not angle_person:
            messagebox.showerror(
                "Missing input", "Please enter an angle_personID."
            )
            return
        if not rep_norep:
            messagebox.showerror(
                "Missing input", "Please enter a rep_norep value."
            )
            return
        for value, name in [
            (angle_person, "angle_personID"),
            (rep_norep,    "rep_norep"),
        ]:
            if not all(c.isalnum() or c in "_-" for c in value):
                messagebox.showerror(
                    "Invalid input",
                    f"'{name}' must contain only letters, numbers, "
                    "underscores, and hyphens."
                )
                return
        self.result = (angle_person, rep_norep)
        self.dialog.destroy()

class SegmentEditDialog:
    """Dialog for editing segment start/end frames and labels."""

    def __init__(
        self,
        parent,
        start_frame: int,
        end_frame: int,
        label: str,
        max_frames: int,
    ):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Edit Segment")
        self.dialog.geometry("300x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(main_frame, text="Start Frame:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.start_var = tk.StringVar(value=str(start_frame))
        start_entry = ttk.Entry(main_frame, textvariable=self.start_var, width=10)
        start_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=2)

        ttk.Label(main_frame, text="End Frame:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.end_var = tk.StringVar(value=str(end_frame))
        end_entry = ttk.Entry(main_frame, textvariable=self.end_var, width=10)
        end_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=2)

        ttk.Label(main_frame, text="Label:").grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.label_var = tk.StringVar(value=label)
        label_combo = ttk.Combobox(
            main_frame, textvariable=self.label_var, width=10
        )
        label_combo["values"] = ("prep", "rep", "no-rep", "finish")
        label_combo.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=2)

        info_label = ttk.Label(
            main_frame,
            text=f"Max frame: {max_frames - 1}",
            foreground="gray",
        )
        info_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(20, 0))
        ttk.Button(
            button_frame, text="OK", command=self.ok_clicked
        ).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(
            button_frame, text="Cancel", command=self.cancel_clicked
        ).grid(row=0, column=1)

        main_frame.columnconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)
        self.max_frames = max_frames

        start_entry.focus()
        start_entry.select_range(0, tk.END)
        self.dialog.wait_window()

    def ok_clicked(self):
        try:
            start_frame = int(self.start_var.get())
            end_frame = int(self.end_var.get())
            label = self.label_var.get().strip()
            if start_frame < 0 or start_frame >= self.max_frames:
                messagebox.showerror(
                    "Invalid input",
                    f"Start frame must be between 0 and {self.max_frames - 1}",
                )
                return
            if end_frame < 0 or end_frame >= self.max_frames:
                messagebox.showerror(
                    "Invalid input",
                    f"End frame must be between 0 and {self.max_frames - 1}",
                )
                return
            if start_frame >= end_frame:
                messagebox.showerror(
                    "Invalid input", "Start frame must be less than end frame"
                )
                return
            if not label or label not in ("prep", "rep", "no-rep", "finish"):
                messagebox.showerror(
                    "Invalid input", "Please select a valid label"
                )
                return
            self.result = (start_frame, end_frame, label)
            self.dialog.destroy()
        except ValueError:
            messagebox.showerror(
                "Invalid input", "Frame numbers must be integers"
            )

    def cancel_clicked(self):
        self.dialog.destroy()


class VideoNamingDialog:
    """Dialog for collecting video naming information for new videos."""

    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Video Naming")
        self.dialog.geometry("400x320")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth()  // 2) - 200
        y = (self.dialog.winfo_screenheight() // 2) - 160
        self.dialog.geometry(f"+{x}+{y}")

        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)

        ttk.Label(
            main_frame,
            text="Enter video naming information:",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

        # Exercise type
        ttk.Label(main_frame, text="Exercise type:").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.exercise_var = tk.StringVar()
        exercise_combo = ttk.Combobox(
            main_frame, textvariable=self.exercise_var, width=20
        )
        exercise_combo["values"] = (
            "double_unders",
            "push_ups",
            "pull_ups",
            "squats",
            "burpees",
            "other",
        )
        exercise_combo.grid(
            row=1, column=1, sticky="ew", padx=(10, 0), pady=5
        )

        # angle_personID
        ttk.Label(main_frame, text="angle_personID:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.angle_person_var = tk.StringVar()
        angle_person_entry = ttk.Entry(
            main_frame, textvariable=self.angle_person_var, width=20
        )
        angle_person_entry.grid(
            row=2, column=1, sticky="ew", padx=(10, 0), pady=5
        )

        # rep_norep
        ttk.Label(main_frame, text="rep_norep:").grid(
            row=3, column=0, sticky="w", pady=5
        )
        self.rep_norep_var = tk.StringVar()
        rep_norep_entry = ttk.Entry(
            main_frame, textvariable=self.rep_norep_var, width=20
        )
        rep_norep_entry.grid(
            row=3, column=1, sticky="ew", padx=(10, 0), pady=5
        )

        # Example frame
        example_frame = ttk.LabelFrame(
            main_frame, text="Example", padding=10
        )
        example_frame.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(20, 10)
        )
        ttk.Label(
            example_frame,
            text="Exercise:      double_unders",
            foreground="gray",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            example_frame,
            text="angle_personID: diag_m2",
            foreground="gray",
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(
            example_frame,
            text="rep_norep:      9_7",
            foreground="gray",
        ).grid(row=2, column=0, sticky="w")
        ttk.Label(
            example_frame,
            text="Result: double_unders_diag_m2_9_7.mp4",
            foreground="blue",
        ).grid(row=3, column=0, sticky="w", pady=(5, 0))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(20, 0)
        )
        ttk.Button(
            button_frame, text="OK", command=self.ok_clicked
        ).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(
            button_frame, text="Cancel", command=self.cancel_clicked
        ).grid(row=0, column=1)

        main_frame.columnconfigure(1, weight=1)
        exercise_combo.focus()
        self.dialog.wait_window()

    def ok_clicked(self):
        exercise     = self.exercise_var.get().strip()
        angle_person = self.angle_person_var.get().strip()
        rep_norep    = self.rep_norep_var.get().strip()
        if not exercise:
            messagebox.showerror(
                "Missing input", "Please enter an exercise type."
            )
            return
        if not angle_person:
            messagebox.showerror(
                "Missing input", "Please enter an angle_personID."
            )
            return
        if not rep_norep:
            messagebox.showerror(
                "Missing input", "Please enter a rep_norep value."
            )
            return
        for value, name in [
            (exercise,     "exercise"),
            (angle_person, "angle_personID"),
            (rep_norep,    "rep_norep"),
        ]:
            if not all(c.isalnum() or c in "_-" for c in value):
                messagebox.showerror(
                    "Invalid input",
                    f"'{name}' must contain only letters, numbers, "
                    "underscores, and hyphens."
                )
                return
        self.result = (exercise, angle_person, rep_norep)
        self.dialog.destroy()

    def cancel_clicked(self):
        self.dialog.destroy()


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if sys.platform == "darwin":
        style.theme_use("clam")
    app = VideoPoseLabellerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()