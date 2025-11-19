"""
CellPose segmentation tab for the Fiji integration GUI.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import numpy as np
from skimage import io
import zipfile

from fiji_integration.gui.base_tab import TabInterface
from fiji_integration.gui.output_stream import GUIOutputStream


__all__ = ['CellPoseTab']


class CellPoseTab(TabInterface):
    """Tab for CellPose instance segmentation."""

    def __init__(self, config_file=None):
        """
        Initialize the CellPose segmentation tab.

        Args:
            config_file: Path to config file (default: auto-detect)
        """
        super().__init__()

        # Config file location
        if config_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fiji_integration_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            config_file = os.path.join(fiji_integration_dir, '.cellpose_gui_config.json')
        self.config_file = config_file

        # Default parameters
        home_dir = os.path.expanduser('~')
        workflow_base = os.path.join(home_dir, 'fiji_workflow')

        self.default_params = {
            'input_dir': os.path.join(workflow_base, '1_max_projection', 'nuclei_channel'),
            'output_dir': os.path.join(workflow_base, '3_cellpose_segmentation'),
            'model_type': 'cyto3',
            'diameter': 0,  # 0 = auto-detect
            'flow_threshold': 0.4,
            'cellprob_threshold': 0.0,
            'use_gpu': True,
            'save_npy': True,
            'save_rois': False,
        }

        # Load saved parameters or use defaults
        self.params = self.load_config()

        # GUI widgets (will be created in build_ui)
        self.input_var = None
        self.output_var = None
        self.model_type_var = None
        self.diameter_var = None
        self.flow_threshold_var = None
        self.cellprob_threshold_var = None
        self.use_gpu_var = None
        self.save_npy_var = None
        self.save_rois_var = None
        self.run_button = None
        self.stop_button = None
        self.restore_button = None

    def get_tab_name(self) -> str:
        """Return the display name for this tab."""
        return "CellPose Segmentation"

    def build_ui(self, parent_frame: ttk.Frame, console_text: tk.Text) -> None:
        """Build the CellPose segmentation UI."""
        self.console_text = console_text

        # Create tkinter variables
        self.input_var = tk.StringVar(value=self.params['input_dir'])
        self.output_var = tk.StringVar(value=self.params['output_dir'])
        self.model_type_var = tk.StringVar(value=self.params['model_type'])
        self.diameter_var = tk.IntVar(value=self.params['diameter'])
        self.flow_threshold_var = tk.DoubleVar(value=self.params['flow_threshold'])
        self.cellprob_threshold_var = tk.DoubleVar(value=self.params['cellprob_threshold'])
        self.use_gpu_var = tk.BooleanVar(value=self.params['use_gpu'])
        self.save_npy_var = tk.BooleanVar(value=self.params['save_npy'])
        self.save_rois_var = tk.BooleanVar(value=self.params['save_rois'])

        row = 0

        # ===== Paths Section =====
        ttk.Label(parent_frame, text="Input/Output Paths", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Label(parent_frame, text="Input Directory:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.input_var, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_input).grid(row=row, column=2)
        row += 1

        ttk.Label(parent_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.output_var, width=50).grid(
            row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(parent_frame, text="Browse...", command=self.browse_output).grid(row=row, column=2)
        row += 1

        # Separator
        ttk.Separator(parent_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Model Configuration =====
        ttk.Label(parent_frame, text="Model Configuration", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Label(parent_frame, text="Model Type:").grid(row=row, column=0, sticky=tk.W)
        model_combo = ttk.Combobox(parent_frame, textvariable=self.model_type_var,
                                   values=['cyto', 'cyto2', 'cyto3', 'nuclei'],
                                   state='readonly', width=20)
        model_combo.grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent_frame, text="Diameter (0=auto):").grid(row=row, column=0, sticky=tk.W)
        ttk.Spinbox(parent_frame, from_=0, to=500, textvariable=self.diameter_var, width=20).grid(
            row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent_frame, text="Flow Threshold (0-1):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.flow_threshold_var, width=20).grid(
            row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(parent_frame, text="Cell Prob Threshold:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent_frame, textvariable=self.cellprob_threshold_var, width=20).grid(
            row=row, column=1, sticky=tk.W, padx=5)
        row += 1

        ttk.Checkbutton(parent_frame, text="Use GPU (faster)", variable=self.use_gpu_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        # Separator
        ttk.Separator(parent_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # ===== Output Options =====
        ttk.Label(parent_frame, text="Output Options", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Checkbutton(parent_frame, text="Save _seg.npy (NumPy format)", variable=self.save_npy_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1

        ttk.Checkbutton(parent_frame, text="Save ROIs (Fiji/ImageJ format) - WARNING: Takes very long", variable=self.save_rois_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
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
                print(f"[LOADED] Loaded saved CellPose configuration from: {self.config_file}")
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
            print(f"[SAVED] Saved CellPose configuration to: {self.config_file}")
        except Exception as e:
            print(f"[WARNING]  Could not save config file: {e}")

    def validate_parameters(self) -> tuple[bool, Optional[str]]:
        """Validate current parameters before running."""
        # Update parameters from GUI
        self.update_params_from_gui()

        # Validate required fields
        if not self.params['input_dir']:
            return False, "Please select an input directory"

        if not os.path.isdir(self.params['input_dir']):
            return False, "Input directory does not exist"

        if not self.params['output_dir']:
            return False, "Please select an output directory"

        # Validate numeric parameters
        try:
            if not (0 <= self.params['flow_threshold'] <= 1):
                return False, "Flow threshold must be between 0 and 1"
            if self.params['diameter'] < 0:
                return False, "Diameter must be non-negative (0 = auto-detect)"
        except ValueError as e:
            return False, str(e)

        # Validate at least one output format is selected
        if not self.params['save_npy'] and not self.params['save_rois']:
            return False, "Please select at least one output format (NPY or ROIs)"

        return True, None

    # GUI helper methods

    def update_params_from_gui(self):
        """Update parameters from GUI widgets."""
        self.params['input_dir'] = self.input_var.get()
        self.params['output_dir'] = self.output_var.get()
        self.params['model_type'] = self.model_type_var.get()
        self.params['diameter'] = int(self.diameter_var.get())
        self.params['flow_threshold'] = float(self.flow_threshold_var.get())
        self.params['cellprob_threshold'] = float(self.cellprob_threshold_var.get())
        self.params['use_gpu'] = self.use_gpu_var.get()
        self.params['save_npy'] = self.save_npy_var.get()
        self.params['save_rois'] = self.save_rois_var.get()

    def update_gui_from_params(self):
        """Update GUI widgets from current parameters."""
        self.input_var.set(self.params['input_dir'])
        self.output_var.set(self.params['output_dir'])
        self.model_type_var.set(self.params['model_type'])
        self.diameter_var.set(self.params['diameter'])
        self.flow_threshold_var.set(self.params['flow_threshold'])
        self.cellprob_threshold_var.set(self.params['cellprob_threshold'])
        self.use_gpu_var.set(self.params['use_gpu'])
        self.save_npy_var.set(self.params['save_npy'])
        self.save_rois_var.set(self.params['save_rois'])

    def restore_defaults(self):
        """Restore all parameters to default values."""
        self.params = self.default_params.copy()
        self.update_gui_from_params()
        print("[OK] Restored CellPose parameters to defaults")

    def browse_input(self):
        """Browse for input directory."""
        path = filedialog.askdirectory(title="Select Input Directory")
        if path:
            self.input_var.set(path)

    def browse_output(self):
        """Browse for output directory."""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_var.set(path)

    # Execution methods

    def on_run_threaded(self):
        """Handle Run button click - runs segmentation in a thread."""
        if self.is_running:
            messagebox.showwarning("Already Running", "CellPose segmentation is already in progress. Please wait.")
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
        self.write_to_console("=== Starting CellPose Segmentation ===\n")
        self.write_to_console(f"Input: {self.params['input_dir']}\n")
        self.write_to_console(f"Output: {self.params['output_dir']}\n")
        self.write_to_console(f"Model: {self.params['model_type']}\n\n")

        # Disable run button, enable stop button, reset stop flag
        self.is_running = True
        self.stop_requested = False
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Run in thread
        thread = threading.Thread(target=self.run_cellpose_segmentation)
        thread.daemon = True
        thread.start()

    def on_stop(self):
        """Handle Stop button click - requests segmentation to stop."""
        if self.is_running:
            self.stop_requested = True
            self.write_to_console("\n[WARNING]  Stop requested. Segmentation will halt after current image...\n")
            self.stop_button.config(state='disabled')

    def run_cellpose_segmentation(self):
        """Run CellPose segmentation and redirect output to console."""
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = GUIOutputStream(self)

        try:
            # Import CellPose and visualization
            print("[RUNNING] Loading CellPose...")
            from cellpose import models
            from cellpose.io import save_masks
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            print("[OK] CellPose loaded successfully\n")

            # Initialize model
            print(f"[RUNNING] Initializing CellPose model ({self.params['model_type']})...")
            model = models.Cellpose(gpu=self.params['use_gpu'], model_type=self.params['model_type'])
            print("[OK] Model initialized\n")

            # Get image files from input directory (recursively)
            input_dir = Path(self.params['input_dir'])
            image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

            # Use rglob to search recursively through all subfolders
            image_files_list = []
            for ext in image_extensions:
                image_files_list.extend(input_dir.rglob(f'*{ext}'))
                image_files_list.extend(input_dir.rglob(f'*{ext.upper()}'))

            # Remove duplicates and convert to Path objects
            unique_files = {}
            for f in image_files_list:
                resolved = str(f.resolve())
                unique_files[resolved] = f

            image_files = sorted(unique_files.values())

            print(f"[FOLDER] Found {len(image_files)} images in directory (including subfolders)\n")

            if len(image_files) == 0:
                print("[WARNING]  No images found in input directory")
                return

            # Process each image
            processed_count = 0
            output_base_dir = Path(self.params['output_dir'])

            for i, img_path_obj in enumerate(image_files, 1):
                # Check if stop was requested
                if self.stop_requested:
                    print(f"\n Segmentation stopped by user after {processed_count}/{len(image_files)} images")
                    break

                img_path = str(img_path_obj)
                print(f"\n{'='*60}")
                print(f"Processing {i}/{len(image_files)}: {img_path_obj.name}")

                # Get relative path from input directory to preserve folder structure
                try:
                    rel_path = img_path_obj.relative_to(input_dir)
                except ValueError:
                    # If not relative, just use the filename
                    rel_path = Path(img_path_obj.name)

                print(f"Relative path: {rel_path}")
                print(f"{'='*60}")

                try:
                    # Load image
                    img = io.imread(img_path)
                    print(f"[IMAGE] Image shape: {img.shape}, dtype: {img.dtype}")

                    # Run segmentation with progress updates
                    import time
                    diameter = self.params['diameter'] if self.params['diameter'] > 0 else None

                    # Print estimate
                    pixels = img.shape[0] * img.shape[1]
                    megapixels = pixels / 1_000_000
                    print(f"[RUNNING] Running CellPose segmentation on {megapixels:.1f} MP image...")
                    print(f"   Device: {'GPU' if self.params['use_gpu'] else 'CPU'}")
                    print(f"   [INFO] Large images take several minutes - progress updates every 30 seconds...")

                    # Start progress monitoring thread
                    start_time = time.time()
                    keep_running = [True]  # Use list to allow modification in nested function

                    def print_progress():
                        """Print progress updates every 30 seconds."""
                        while keep_running[0]:
                            time.sleep(30)
                            if keep_running[0]:
                                elapsed = time.time() - start_time
                                print(f"   [PROGRESS] Still running... {elapsed:.0f}s elapsed ({elapsed/60:.1f} min)")

                    progress_thread = threading.Thread(target=print_progress, daemon=True)
                    progress_thread.start()

                    try:
                        masks, flows, styles, diams = model.eval(
                            img,
                            diameter=diameter,
                            channels=[0, 0],
                            flow_threshold=self.params['flow_threshold'],
                            cellprob_threshold=self.params['cellprob_threshold']
                        )
                    finally:
                        # Stop progress updates
                        keep_running[0] = False

                    elapsed = time.time() - start_time
                    print(f"[OK] Segmentation completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

                    num_cells = len(np.unique(masks)) - 1  # Subtract background
                    print(f"[OK] Found {num_cells} cells")

                    # Create output folder for this image (preserving subfolder structure)
                    # Structure: output_dir/subfolder/image_name/files
                    base_name = img_path_obj.stem
                    parent_rel_path = rel_path.parent

                    # Create folder: output_dir/parent_folders/image_name/
                    image_output_dir = output_base_dir / parent_rel_path / base_name
                    os.makedirs(image_output_dir, exist_ok=True)
                    print(f"[LOADED] Output folder: {image_output_dir}")

                    # Save _seg.npy format
                    if self.params['save_npy']:
                        npy_path = image_output_dir / f"{base_name}_seg.npy"
                        np.save(npy_path, masks)
                        print(f"[SAVED] Saved masks: {npy_path.name}")

                    # Save ROIs (Fiji format)
                    if self.params['save_rois']:
                        roi_path = image_output_dir / f"{base_name}_RoiSet.zip"
                        self.save_rois_fiji(masks, str(roi_path))
                        print(f"[SAVED] Saved ROIs: {roi_path.name}")

                    # Save visualization (overlay)
                    print("[RUNNING] Creating visualization...")
                    viz_path = image_output_dir / f"{base_name}_overlay.png"
                    self.save_visualization(img, masks, str(viz_path))
                    print(f"[SAVED] Saved visualization: {viz_path.name}")

                    processed_count += 1

                except Exception as e:
                    print(f"[ERROR] Error processing {img_path_obj.name}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue

            if not self.stop_requested:
                print(f"\n Batch processing complete! Processed {processed_count} images.")
            else:
                print(f"\n[OK] Partial results saved for {processed_count} processed images.")

            print(f"[LOADED] Results saved to: {self.params['output_dir']}")

        except Exception as e:
            print(f"\n[ERROR] Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            # Restore stdout
            sys.stdout = old_stdout

            # Re-enable run button, disable stop button
            self.is_running = False
            self.root.after(0, lambda: self.run_button.config(state='normal'))
            self.root.after(0, lambda: self.stop_button.config(state='disabled'))

    def save_visualization(self, image, masks, output_path):
        """
        Save visualization with overlay of masks on original image.

        Args:
            image: Original image (2D numpy array)
            masks: Segmentation masks (2D numpy array with labeled regions)
            output_path: Path to save visualization PNG
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')

        # Segmentation masks
        axes[1].imshow(masks, cmap='nipy_spectral')
        axes[1].set_title(f'Segmentation ({len(np.unique(masks)) - 1} cells)', fontsize=14)
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(image, cmap='gray', alpha=0.7)
        axes[2].imshow(masks, cmap='nipy_spectral', alpha=0.3)
        axes[2].set_title('Overlay', fontsize=14)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def save_rois_fiji(self, masks, output_path):
        """
        Save segmentation masks as Fiji/ImageJ ROI set.

        Args:
            masks: 2D numpy array with labeled regions
            output_path: Path to save ROI zip file
        """
        from skimage import measure
        import struct

        # Find all unique labels (excluding background 0)
        labels = np.unique(masks)
        labels = labels[labels != 0]

        if len(labels) == 0:
            print("[WARNING]  No objects to save as ROIs")
            return

        # Create temporary directory for individual ROI files
        import tempfile
        temp_dir = tempfile.mkdtemp()

        try:
            roi_files = []

            for label_idx, label in enumerate(labels, 1):
                # Get contours for this label
                binary_mask = (masks == label).astype(np.uint8)
                contours = measure.find_contours(binary_mask, 0.5)

                if len(contours) == 0:
                    continue

                # Use the longest contour
                contour = max(contours, key=len)

                # Convert to ImageJ ROI format (polygon)
                # ROI file format: https://imagej.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
                roi_name = f"{label_idx:04d}.roi"
                roi_path = os.path.join(temp_dir, roi_name)

                # Write ROI file
                with open(roi_path, 'wb') as f:
                    # Header
                    f.write(b'Iout')  # Magic number
                    f.write(struct.pack('>h', 227))  # Version
                    f.write(struct.pack('>h', 0))  # ROI type: polygon

                    # Bounding box (we'll calculate from contour)
                    y_coords = contour[:, 0]
                    x_coords = contour[:, 1]
                    top = int(np.min(y_coords))
                    left = int(np.min(x_coords))
                    bottom = int(np.max(y_coords))
                    right = int(np.max(x_coords))
                    width = right - left
                    height = bottom - top

                    f.write(struct.pack('>h', top))
                    f.write(struct.pack('>h', left))
                    f.write(struct.pack('>h', bottom))
                    f.write(struct.pack('>h', right))

                    # Number of points
                    n_points = len(contour)
                    f.write(struct.pack('>h', n_points))

                    # Stroke width, shape ROI size (not used for polygon)
                    f.write(struct.pack('>f', 0))  # stroke width
                    f.write(struct.pack('>i', 0))  # shape roi size

                    # Stroke color, fill color (not used)
                    f.write(struct.pack('>i', 0))  # stroke color
                    f.write(struct.pack('>i', 0))  # fill color

                    # Subtype, options, arrow style/head size, rect rounded corner diameter
                    f.write(struct.pack('>h', 0))  # subtype
                    f.write(struct.pack('>h', 0))  # options
                    f.write(struct.pack('>B', 0))  # arrow style
                    f.write(struct.pack('>B', 0))  # arrow head size
                    f.write(struct.pack('>h', 0))  # rounded rect corner diameter

                    # Position (slice, position in stack)
                    f.write(struct.pack('>i', 0))  # position

                    # Header2 offset (64 bytes from start)
                    f.write(struct.pack('>i', 64))

                    # Padding to get to byte 64
                    current_pos = f.tell()
                    padding = 64 - current_pos
                    f.write(b'\x00' * padding)

                    # Write coordinates (x and y as shorts, relative to bounding box)
                    for y, x in contour:
                        x_rel = int(x) - left
                        y_rel = int(y) - top
                        f.write(struct.pack('>h', x_rel))

                    for y, x in contour:
                        x_rel = int(x) - left
                        y_rel = int(y) - top
                        f.write(struct.pack('>h', y_rel))

                    # Write name (header2)
                    name_bytes = roi_name.encode('utf-8')
                    f.write(struct.pack('>i', len(name_bytes)))
                    f.write(name_bytes)

                roi_files.append(roi_path)

            # Create ZIP file with all ROIs
            with zipfile.ZipFile(output_path, 'w') as zf:
                for roi_file in roi_files:
                    zf.write(roi_file, os.path.basename(roi_file))

        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
