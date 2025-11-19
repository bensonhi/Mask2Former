"""
Myotube segmentation tab for the Fiji integration GUI.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Any, Optional
import threading

from fiji_integration.gui.base_tab import TabInterface
from fiji_integration.gui.output_stream import GUIOutputStream
from fiji_integration.utils.constants import DEFAULT_GUI_CONFIG, IMAGE_EXTENSIONS
from fiji_integration.utils.path_utils import ensure_mask2former_loaded


__all__ = ['MyotubeTab']


class MyotubeTab(TabInterface):
    """Tab for myotube instance segmentation."""

    def __init__(self, config_file=None):
        """
        Initialize the myotube segmentation tab.

        Args:
            config_file: Path to config file (default: auto-detect)
        """
        super().__init__()

        # Config file location
        if config_file is None:
            # Try to find fiji_integration directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fiji_integration_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            config_file = os.path.join(fiji_integration_dir, '.myotube_gui_config.json')
        self.config_file = config_file

        # Default parameters with workflow paths
        home_dir = os.path.expanduser('~')
        workflow_base = os.path.join(home_dir, 'fiji_workflow')

        self.default_params = DEFAULT_GUI_CONFIG.copy()
        self.default_params['input_path'] = os.path.join(workflow_base, '1_max_projection', 'myotube_channel')
        self.default_params['output_dir'] = os.path.join(workflow_base, '2_myotube_segmentation')

        # Load saved parameters or use defaults
        self.params = self.load_config()

        # GUI widgets (will be created in build_ui)
        self.input_var = None
        self.output_var = None
        self.config_var = None
        self.weights_var = None
        self.mask2former_path_var = None
        self.confidence_var = None
        self.min_area_var = None
        self.max_area_var = None
        self.final_min_area_var = None
        self.cpu_var = None
        self.max_image_size_var = None
        self.force_1024_var = None
        self.use_tiling_var = None
        self.grid_size_var = None
        self.tile_overlap_var = None
        self.skip_merged_var = None
        self.save_measurements_var = None
        self.confidence_label = None
        self.tile_overlap_label = None
        self.run_button = None
        self.stop_button = None
        self.restore_button = None

    def get_tab_name(self) -> str:
        """Return the display name for this tab."""
        return "Myotube Segmentation"

    def build_ui(self, parent_frame: ttk.Frame, console_text: tk.Text) -> None:
        """Build the myotube segmentation UI."""
        self.console_text = console_text

        # Create tkinter variables
        self.input_var = tk.StringVar(value=self.params['input_path'])
        self.output_var = tk.StringVar(value=self.params['output_dir'])
        self.config_var = tk.StringVar(value=self.params['config'])
        self.weights_var = tk.StringVar(value=self.params['weights'])
        self.mask2former_path_var = tk.StringVar(value=self.params['mask2former_path'])
        self.confidence_var = tk.DoubleVar(value=self.params['confidence'])
        self.min_area_var = tk.IntVar(value=self.params['min_area'])
        self.max_area_var = tk.IntVar(value=self.params['max_area'])
        self.final_min_area_var = tk.IntVar(value=self.params['final_min_area'])
        self.cpu_var = tk.BooleanVar(value=self.params['cpu'])
        self.max_image_size_var = tk.StringVar(value=str(self.params['max_image_size']) if self.params['max_image_size'] else '')
        self.force_1024_var = tk.BooleanVar(value=self.params['force_1024'])
        self.use_tiling_var = tk.BooleanVar(value=self.params['use_tiling'])
        self.grid_size_var = tk.IntVar(value=self.params['grid_size'])
        self.tile_overlap_var = tk.DoubleVar(value=self.params['tile_overlap'] * 100)
        self.skip_merged_var = tk.BooleanVar(value=self.params['skip_merged_masks'])
        self.save_measurements_var = tk.BooleanVar(value=self.params['save_measurements'])

        row = 0

        # ===== Paths Section =====
        ttk.Label(parent_frame, text="Input/Output Paths", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Label(parent_frame, text="Input (Image/Directory):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.input_var, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_input).grid(row=row, column=2)
        row += 1

        ttk.Label(parent_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.output_var, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_output).grid(row=row, column=2)
        row += 1

        # Separator
        ttk.Separator(parent_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Model Configuration =====
        ttk.Label(parent_frame, text="Model Configuration", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Label(parent_frame, text="Config File:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.config_var, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_config).grid(row=row, column=2)
        row += 1

        ttk.Label(parent_frame, text="Model Weights:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.weights_var, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_weights).grid(row=row, column=2)
        row += 1

        ttk.Label(parent_frame, text="Mask2Former Path:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.mask2former_path_var, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_mask2former_path).grid(row=row, column=2)
        row += 1

        ttk.Label(parent_frame, text="(Leave empty for auto-detection)", font=('Arial', 9, 'italic')).grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        # Separator
        ttk.Separator(parent_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Detection Parameters =====
        ttk.Label(parent_frame, text="Detection Parameters", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Label(parent_frame, text="Confidence Threshold (0-1):").grid(row=row, column=0, sticky=tk.W)
        confidence_scale = ttk.Scale(parent_frame, from_=0.0, to=1.0, variable=self.confidence_var, orient='horizontal', length=300)
        confidence_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        self.confidence_label = ttk.Label(parent_frame, text=f"{self.confidence_var.get():.2f}")
        self.confidence_label.grid(row=row, column=2)
        confidence_scale.configure(command=lambda v: self.confidence_label.configure(text=f"{float(v):.2f}"))
        row += 1

        ttk.Label(parent_frame, text="Minimum Area (pixels):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.min_area_var, width=20).grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent_frame, text="Maximum Area (pixels):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.max_area_var, width=20).grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent_frame, text="Final Min Area (pixels):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.final_min_area_var, width=20).grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        # Separator
        ttk.Separator(parent_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Performance Options =====
        ttk.Label(parent_frame, text="Performance Options", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Checkbutton(parent_frame, text="Use CPU (slower, less memory)", variable=self.cpu_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Checkbutton(parent_frame, text="Force 1024px input (memory optimization)", variable=self.force_1024_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(parent_frame, text="Max Image Size (optional):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.max_image_size_var, width=20).grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        # Separator
        ttk.Separator(parent_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Tiling Options =====
        ttk.Label(parent_frame, text="Tiling Options", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Checkbutton(parent_frame, text="Use tiled inference (for images with many myotubes)", variable=self.use_tiling_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(parent_frame, text="Grid Size (1=no split, 2=2×2, etc.):").grid(row=row, column=0, sticky=tk.W)
        grid_size_spinbox = ttk.Spinbox(parent_frame, from_=1, to=10, textvariable=self.grid_size_var, width=10)
        grid_size_spinbox.grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent_frame, text="Tile Overlap (%):").grid(row=row, column=0, sticky=tk.W)
        tile_overlap_scale = ttk.Scale(parent_frame, from_=10, to=50, variable=self.tile_overlap_var, orient='horizontal', length=300)
        tile_overlap_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        self.tile_overlap_label = ttk.Label(parent_frame, text=f"{self.tile_overlap_var.get():.1f}")
        self.tile_overlap_label.grid(row=row, column=2)
        tile_overlap_scale.configure(command=lambda v: self.tile_overlap_label.configure(text=f"{float(v):.1f}"))
        row += 1

        # Separator
        ttk.Separator(parent_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Output Options =====
        ttk.Label(parent_frame, text="Output Options", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Checkbutton(parent_frame, text="Skip merged masks (skip imaginary boundary generation)", variable=self.skip_merged_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Checkbutton(parent_frame, text="Save measurements CSV (includes area, length, width, etc.)", variable=self.save_measurements_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        # Configure column weights
        parent_frame.columnconfigure(1, weight=1)

        # Create buttons (will be added to shared button frame)
        self.restore_button = ttk.Button(self.button_frame, text="Restore Defaults", command=self.restore_defaults)
        self.run_button = ttk.Button(self.button_frame, text="Run Segmentation", command=self.on_run_threaded)
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.on_stop, state='disabled')

    def get_button_frame_widgets(self) -> list:
        """Return buttons for the shared button area."""
        return [
            (self.restore_button, tk.LEFT),
            (self.run_button, tk.LEFT),
            (self.stop_button, tk.LEFT),
        ]

    def load_config(self) -> Dict[str, Any]:
        """Load saved configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                # Merge with defaults to handle new parameters
                params = self.default_params.copy()
                params.update(saved)
                print(f"[LOADED] Loaded saved configuration from: {self.config_file}")
                return params
            except Exception as e:
                print(f"[WARNING]  Could not load config file: {e}")
                return self.default_params.copy()
        else:
            return self.default_params.copy()

    def save_config(self, config: Dict[str, Any] = None) -> None:
        """Save configuration to file."""
        if config is None:
            config = self.params
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[SAVED] Saved configuration to: {self.config_file}")
        except Exception as e:
            print(f"[WARNING]  Could not save config file: {e}")

    def validate_parameters(self) -> tuple[bool, Optional[str]]:
        """Validate current parameters before running."""
        # Update parameters from GUI
        self.update_params_from_gui()

        # Validate required fields
        if not self.params['input_path']:
            return False, "Please select an input image or directory"

        if not self.params['output_dir']:
            return False, "Please select an output directory"

        # Validate numeric parameters
        try:
            if not (0 <= self.params['confidence'] <= 1):
                return False, "Confidence must be between 0 and 1"
            if self.params['min_area'] <= 0:
                return False, "Minimum area must be positive"
            if self.params['max_area'] <= self.params['min_area']:
                return False, "Maximum area must be greater than minimum area"
            if self.params['final_min_area'] < 0:
                return False, "Final minimum area must be non-negative"
            if self.params['grid_size'] < 1:
                return False, "Grid size must be at least 1"
            if not (0 < self.params['tile_overlap'] < 1):
                return False, "Tile overlap must be between 0 and 1"
        except ValueError as e:
            return False, str(e)

        return True, None

    # GUI helper methods

    def update_params_from_gui(self):
        """Update parameters from GUI widgets."""
        self.params['input_path'] = self.input_var.get()
        self.params['output_dir'] = self.output_var.get()
        self.params['config'] = self.config_var.get()
        self.params['weights'] = self.weights_var.get()
        self.params['mask2former_path'] = self.mask2former_path_var.get()

        self.params['confidence'] = float(self.confidence_var.get())
        self.params['min_area'] = int(self.min_area_var.get())
        self.params['max_area'] = int(self.max_area_var.get())
        self.params['final_min_area'] = int(self.final_min_area_var.get())

        self.params['cpu'] = self.cpu_var.get()
        self.params['force_1024'] = self.force_1024_var.get()
        self.params['use_tiling'] = self.use_tiling_var.get()
        self.params['skip_merged_masks'] = self.skip_merged_var.get()
        self.params['save_measurements'] = self.save_measurements_var.get()

        # Optional max_image_size
        max_size_str = self.max_image_size_var.get().strip()
        self.params['max_image_size'] = int(max_size_str) if max_size_str else ''

        # Grid size and tile overlap
        self.params['grid_size'] = int(self.grid_size_var.get())
        self.params['tile_overlap'] = float(self.tile_overlap_var.get()) / 100.0

    def update_gui_from_params(self):
        """Update GUI widgets from current parameters."""
        # Paths
        self.input_var.set(self.params['input_path'])
        self.output_var.set(self.params['output_dir'])
        self.config_var.set(self.params['config'])
        self.weights_var.set(self.params['weights'])
        self.mask2former_path_var.set(self.params['mask2former_path'])

        # Processing parameters
        self.confidence_var.set(self.params['confidence'])
        self.min_area_var.set(self.params['min_area'])
        self.max_area_var.set(self.params['max_area'])
        self.final_min_area_var.set(self.params['final_min_area'])

        # Flags
        self.cpu_var.set(self.params['cpu'])
        self.force_1024_var.set(self.params['force_1024'])
        self.use_tiling_var.set(self.params['use_tiling'])
        self.skip_merged_var.set(self.params['skip_merged_masks'])

        # Optional parameters
        self.max_image_size_var.set(str(self.params['max_image_size']) if self.params['max_image_size'] else '')
        self.grid_size_var.set(self.params['grid_size'])
        self.tile_overlap_var.set(self.params['tile_overlap'] * 100)  # Display as percentage

        # Update formatted labels if they exist
        if self.confidence_label:
            self.confidence_label.configure(text=f"{self.params['confidence']:.2f}")
        if self.tile_overlap_label:
            self.tile_overlap_label.configure(text=f"{self.params['tile_overlap'] * 100:.1f}")

    def restore_defaults(self):
        """Restore all parameters to default values."""
        self.params = self.default_params.copy()
        self.update_gui_from_params()
        print("[OK] Restored parameters to defaults")

    def browse_input(self):
        """Browse for input file or directory."""
        path = filedialog.askdirectory(title="Select Input Directory")
        if not path:
            path = filedialog.askopenfilename(
                title="Select Input Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
            )
        if path:
            self.input_var.set(path)

    def browse_output(self):
        """Browse for output directory."""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_var.set(path)

    def browse_config(self):
        """Browse for config file."""
        path = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if path:
            self.config_var.set(path)

    def browse_weights(self):
        """Browse for model weights."""
        path = filedialog.askopenfilename(
            title="Select Model Weights",
            filetypes=[("Model files", "*.pth *.pkl"), ("All files", "*.*")]
        )
        if path:
            self.weights_var.set(path)

    def browse_mask2former_path(self):
        """Browse for Mask2Former project directory."""
        path = filedialog.askdirectory(
            title="Select Mask2Former Project Directory"
        )
        if path:
            self.mask2former_path_var.set(path)

    # Execution methods

    def on_run_threaded(self):
        """Handle Run button click - runs segmentation in a thread."""
        if self.is_running:
            messagebox.showwarning("Already Running", "Segmentation is already in progress. Please wait.")
            return

        # Validate parameters
        is_valid, error_msg = self.validate_parameters()
        if not is_valid:
            messagebox.showerror("Invalid Parameters", error_msg)
            return

        # Save configuration
        self.save_config()

        # Clear console and start segmentation in thread
        self.clear_console()
        self.write_to_console("=== Starting Segmentation ===\n")
        self.write_to_console(f"Input: {self.params['input_path']}\n")
        self.write_to_console(f"Output: {self.params['output_dir']}\n\n")

        # Disable run button, enable stop button, reset stop flag
        self.is_running = True
        self.stop_requested = False
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Run in thread
        thread = threading.Thread(target=self.run_segmentation_in_gui)
        thread.daemon = True
        thread.start()

    def on_stop(self):
        """Handle Stop button click - requests segmentation to stop."""
        if self.is_running:
            self.stop_requested = True
            self.write_to_console("\n[WARNING]  Stop requested. Segmentation will halt after current image...\n")
            self.stop_button.config(state='disabled')

    def run_segmentation_in_gui(self):
        """Run segmentation and redirect output to console."""
        old_stdout = sys.stdout

        try:
            # Redirect stdout to capture print statements
            sys.stdout = GUIOutputStream(self)
            # Load Mask2Former modules
            print("[RUNNING] Loading Mask2Former and detectron2 modules...")
            ensure_mask2former_loaded(explicit_path=self.params.get('mask2former_path'))

            # Import after Mask2Former is loaded
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            from detectron2.data.detection_utils import read_image
            from mask2former import add_maskformer2_config
            print("[OK] Modules loaded successfully\n")

            # Import our core modules (these are now in the modular structure)
            # Note: We can't use relative imports here because this runs in a thread
            # We need to import from the monolithic file for now
            # This will be fixed when we update the main myotube_segmentation.py file
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from myotube_segmentation import MyotubeFijiIntegration, TiledMyotubeSegmentation

            # Build custom config from parameters
            max_image_size = 1024 if self.params.get('force_1024') else self.params.get('max_image_size')
            custom_config = {
                'confidence_threshold': self.params['confidence'],
                'min_area': self.params['min_area'],
                'max_area': self.params['max_area'],
                'final_min_area': self.params['final_min_area'],
                'max_image_size': max_image_size,
                'force_cpu': self.params['cpu'],
                'save_measurements': self.params['save_measurements']
            }

            # Initialize integration
            integration = MyotubeFijiIntegration(
                config_file=self.params['config'] if self.params['config'] else None,
                model_weights=self.params['weights'] if self.params['weights'] else None,
                skip_merged_masks=self.params['skip_merged_masks'],
                mask2former_path=self.params['mask2former_path'] if self.params['mask2former_path'] else None
            )

            # Initialize tiled segmentation if requested
            if self.params['use_tiling']:
                print(f" Tiled inference mode enabled (grid: {self.params['grid_size']}×{self.params['grid_size']}, overlap: {self.params['tile_overlap']*100:.0f}%)")
                tiled_segmenter = TiledMyotubeSegmentation(
                    segmentation_backend=integration,
                    target_overlap=self.params['tile_overlap'],
                    grid_size=self.params['grid_size']
                )
            else:
                tiled_segmenter = None

            # Process input
            input_path = self.params['input_path']
            output_dir = self.params['output_dir']

            if os.path.isfile(input_path):
                # Single image
                print(f"[IMAGE] Processing single image: {input_path}")
                try:
                    if tiled_segmenter:
                        tiled_segmenter.segment_image_tiled(input_path, output_dir, custom_config)
                    else:
                        integration.segment_image(input_path, output_dir, custom_config)
                except Exception as e:
                    print(f"[ERROR] Error processing image: {e}")
                    import traceback
                    print("\nFull error traceback:")
                    print(traceback.format_exc())
                    raise
            elif os.path.isdir(input_path):
                # Directory of images
                base_dir = Path(input_path)

                # Priority: search images/ subdirectory if it exists, otherwise search base directory
                images_subdir = base_dir / 'images'
                if images_subdir.exists() and images_subdir.is_dir():
                    search_dir = images_subdir
                    print(f"   [LOADED] Searching images/ subdirectory")
                else:
                    search_dir = base_dir

                # Collect image files
                image_files_set = []
                for ext in IMAGE_EXTENSIONS:
                    for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                        image_files_set.extend(search_dir.rglob(pattern))

                # De-duplicate using resolved absolute paths
                unique_files = {}
                for f in image_files_set:
                    try:
                        resolved = str(f.resolve())
                        if sys.platform == 'win32':
                            resolved = resolved.lower()
                        unique_files[resolved] = f
                    except (OSError, RuntimeError):
                        unique_files[str(f)] = f

                image_files = sorted(unique_files.values())

                print(f"[FOLDER] Found {len(image_files)} images in directory (including subfolders)")

                processed_count = 0
                for i, img_path_obj in enumerate(image_files, 1):
                    # Check if stop was requested
                    if self.stop_requested:
                        print(f"\n Segmentation stopped by user after {processed_count}/{len(image_files)} images")
                        break

                    print(f"\n{'='*60}")
                    print(f"Processing {i}/{len(image_files)}: {img_path_obj.name}")

                    # Get relative path from search directory to preserve folder structure
                    try:
                        rel_path = img_path_obj.relative_to(search_dir)
                    except ValueError:
                        # If not relative, just use the filename
                        rel_path = Path(img_path_obj.name)

                    print(f"   Relative path: {rel_path}")
                    print(f"{'='*60}")

                    try:
                        # Create subdirectory for each image's output, preserving folder structure
                        # Structure: output_dir/parent_folders/image_name/
                        base_name = img_path_obj.stem
                        parent_rel_path = rel_path.parent

                        # Create folder: output_dir/parent_folders/image_name/
                        image_output_dir = os.path.join(output_dir, str(parent_rel_path), base_name)

                        print(f"   [LOADED] Output folder: {image_output_dir}")

                        if tiled_segmenter:
                            tiled_segmenter.segment_image_tiled(str(img_path_obj), image_output_dir, custom_config)
                        else:
                            integration.segment_image(str(img_path_obj), image_output_dir, custom_config)
                        processed_count += 1
                    except Exception as e:
                        print(f"[ERROR] Error processing {img_path_obj.name}: {e}")
                        continue

                if not self.stop_requested:
                    print(f"\n Batch processing complete! Processed {processed_count} images.")
                else:
                    print(f"\n[OK] Partial results saved for {processed_count} processed images.")

            if not self.stop_requested:
                print(f"\n[OK] All segmentation complete!")
            print(f"[LOADED] Results saved to: {output_dir}")

        except Exception as e:
            try:
                print(f"\n{'='*80}")
                print(f"[ERROR] SEGMENTATION FAILED")
                print(f"{'='*80}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"\nFull traceback:")
                import traceback
                print(traceback.format_exc())
                print(f"{'='*80}")
                print(f"Please check the error above and verify:")
                print(f"  - Input path exists and is valid")
                print(f"  - Mask2Former model files are accessible")
                print(f"  - Sufficient GPU/CPU memory is available")
                print(f"  - All dependencies are properly installed")
                print(f"{'='*80}")
            except:
                # If even error printing fails, write directly to console
                import traceback
                error_text = f"\n{'='*80}\n[ERROR] CRITICAL ERROR\n{'='*80}\n{traceback.format_exc()}\n{'='*80}\n"
                try:
                    self.write_to_console(error_text)
                except:
                    # Last resort - print to original stdout
                    sys.stdout = old_stdout
                    print(error_text)
        finally:
            # Restore stdout (always, no matter what)
            try:
                sys.stdout = old_stdout
            except:
                pass

            # Re-enable run button, disable stop button (always, no matter what)
            try:
                self.is_running = False
                self.root.after(0, lambda: self.run_button.config(state='normal'))
                self.root.after(0, lambda: self.stop_button.config(state='disabled'))
            except:
                pass
