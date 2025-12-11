#!/usr/bin/env python3
"""
Myotube Instance Segmentation for Fiji Integration

This script provides myotube instance segmentation with modular post-processing
pipeline and seamless Fiji integration via composite ROIs.

Usage:
    python myotube_segmentation.py input_image output_dir [--config CONFIG] [--weights WEIGHTS]

Features:
    - Modular post-processing pipeline (easy to extend)
    - Composite ROI generation for multi-segment instances
    - Automatic Fiji-compatible output formats
    - Configurable confidence thresholds and filtering
"""

import os
import sys
import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from datetime import datetime

# Import torch early to check CUDA availability
import torch

# Fix Windows console encoding for emoji/Unicode characters
# Only apply if stdout hasn't been replaced by GUI (check for .buffer attribute)
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path to allow fiji_integration imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Check if CUDA is available - if not, force CPU mode before importing detectron2
# This prevents detectron2 from trying to load CUDA libraries that don't exist
if not torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide all CUDA devices
    print("⚠️  CUDA not available - forcing CPU mode")

# Import path utilities from the new modular structure
from fiji_integration.utils.path_utils import find_mask2former_project, ensure_mask2former_loaded

# Global variable for project directory (used by imported modules)
project_dir = None

# Check if we're in GUI mode - GUI doesn't need detectron2/mask2former imports immediately
GUI_MODE = '--gui' in sys.argv

# Check for explicit Mask2Former path in command-line arguments (before parsing)
_explicit_m2f_path = None
if '--mask2former-path' in sys.argv:
    try:
        _idx = sys.argv.index('--mask2former-path')
        if _idx + 1 < len(sys.argv):
            _explicit_m2f_path = sys.argv[_idx + 1]
    except (IndexError, ValueError):
        pass

# Detectron2 imports - only load if NOT in GUI mode
# In GUI mode, these will be loaded later after user selects parameters
if not GUI_MODE:
    try:
        project_dir = ensure_mask2former_loaded(explicit_path=_explicit_m2f_path)
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.data.detection_utils import read_image
        from mask2former import add_maskformer2_config
    except Exception as _import_exc:
        # Write an ERROR status file early if imports fail (common in Fiji env)
        try:
            if len(sys.argv) >= 3:
                _out_dir = sys.argv[2]
                os.makedirs(_out_dir, exist_ok=True)
                with open(os.path.join(_out_dir, "ERROR"), 'w') as f:
                    f.write(f"Failed to import Mask2Former or Detectron2: {_import_exc}\n")
                    f.write("\nTroubleshooting:\n")
                    f.write("1. Ensure detectron2 is installed: pip install 'git+https://github.com/facebookresearch/detectron2.git'\n")
                    f.write("2. Ensure Mask2Former is available: git clone https://github.com/facebookresearch/Mask2Former.git\n")
                    f.write("3. Set MASK2FORMER_PATH environment variable or use --mask2former-path argument\n")
        except:
            pass
        raise _import_exc
else:
    # GUI mode - set these to None for now, will be imported later
    DefaultPredictor = None
    get_cfg = None
    add_deeplab_config = None
    read_image = None
    add_maskformer2_config = None

# Import core classes from the new modular structure
from fiji_integration.core import (
    PostProcessingPipeline,
    MyotubeFijiIntegration,
    TiledMyotubeSegmentation
)

class GUIOutputStream:
    """Redirects stdout to GUI console widget."""

    def __init__(self, gui):
        self.gui = gui

    def write(self, text):
        if text:  # Write all text including newlines
            self.gui.write_to_console(text)

    def flush(self):
        pass  # Required for file-like object


class ParameterGUI:
    """User-friendly GUI for parameter configuration."""

    def __init__(self, config_file=None, locked_output_dir=None):
        """Initialize the GUI with saved or default parameters.

        Args:
            config_file: Path to config file (default: auto-detect)
            locked_output_dir: If provided, locks the output directory (used by Fiji integration)
        """
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.locked_output_dir = locked_output_dir

        # Default parameters
        # Set default output directory to Desktop/myotube_results
        default_output = os.path.join(os.path.expanduser('~'), 'Desktop', 'myotube_results')

        self.defaults = {
            'input_path': '',
            'output_dir': default_output,
            'config': '',
            'weights': '',
            'mask2former_path': '',
            'confidence': 0.25,
            'min_area': 100,
            'max_area': 50000,
            'final_min_area': 1000,
            'cpu': False,
            'max_image_size': '',
            'force_1024': False,
            'use_tiling': True,
            'grid_size': 2,
            'tile_overlap': 0.20,
            'skip_merged_masks': True,
            'save_measurements': False,
        }

        # Config file location (in script directory or user home)
        if config_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(script_dir, '.myotube_gui_config.json')
        self.config_file = config_file

        # Load saved parameters
        self.params = self.load_config()

        # Note: locked_output_dir parameter kept for backward compatibility but not used
        # Users can now always choose their output directory in the GUI

        # GUI state
        self.result = None
        self.root = None
        self.console_text = None
        self.is_running = False
        self.stop_requested = False

    def write_to_console(self, text):
        """Write text to console widget."""
        if self.console_text:
            self.console_text.config(state='normal')
            self.console_text.insert(self.tk.END, text)
            self.console_text.see(self.tk.END)  # Auto-scroll to bottom
            self.console_text.config(state='disabled')
            self.root.update_idletasks()

    def clear_console(self):
        """Clear console widget."""
        if self.console_text:
            self.console_text.config(state='normal')
            self.console_text.delete('1.0', self.tk.END)
            self.console_text.config(state='disabled')

    def load_config(self):
        """Load saved configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                # Merge with defaults to handle new parameters
                params = self.defaults.copy()
                params.update(saved)
                print(f"📂 Loaded saved configuration from: {self.config_file}")
                return params
            except Exception as e:
                print(f"⚠️  Could not load config file: {e}")
                return self.defaults.copy()
        else:
            return self.defaults.copy()

    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.params, f, indent=2)
            print(f"💾 Saved configuration to: {self.config_file}")
        except Exception as e:
            print(f"⚠️  Could not save config file: {e}")

    def restore_defaults(self):
        """Restore all parameters to default values."""
        # Keep paths but restore processing parameters
        input_path = self.params.get('input_path', '')
        output_dir = self.params.get('output_dir', '')
        config = self.params.get('config', '')
        weights = self.params.get('weights', '')

        self.params = self.defaults.copy()

        # Restore the paths
        self.params['input_path'] = input_path
        self.params['output_dir'] = output_dir
        self.params['config'] = config
        self.params['weights'] = weights

        # Update GUI
        self.update_gui_from_params()

        # No popup - just restore silently
        print("✅ Restored parameters to defaults (paths preserved)")

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
        if hasattr(self, 'confidence_label'):
            self.confidence_label.configure(text=f"{self.params['confidence']:.2f}")
        if hasattr(self, 'tile_overlap_label'):
            self.tile_overlap_label.configure(text=f"{self.params['tile_overlap'] * 100:.1f}")

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

    def browse_input(self):
        """Browse for input file or directory."""
        path = self.filedialog.askdirectory(title="Select Input Directory")
        if not path:
            path = self.filedialog.askopenfilename(
                title="Select Input Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
            )
        if path:
            self.input_var.set(path)

    def browse_output(self):
        """Browse for output directory."""
        path = self.filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_var.set(path)

    def browse_config(self):
        """Browse for config file."""
        path = self.filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if path:
            self.config_var.set(path)

    def browse_weights(self):
        """Browse for model weights."""
        path = self.filedialog.askopenfilename(
            title="Select Model Weights",
            filetypes=[("Model files", "*.pth *.pkl"), ("All files", "*.*")]
        )
        if path:
            self.weights_var.set(path)

    def browse_mask2former_path(self):
        """Browse for Mask2Former project directory."""
        path = self.filedialog.askdirectory(
            title="Select Mask2Former Project Directory"
        )
        if path:
            self.mask2former_path_var.set(path)

    def on_run_threaded(self):
        """Handle Run button click - runs segmentation in a thread."""
        if self.is_running:
            self.messagebox.showwarning("Already Running", "Segmentation is already in progress. Please wait.")
            return

        # Update parameters from GUI
        self.update_params_from_gui()

        # Validate required fields
        if not self.params['input_path']:
            self.messagebox.showerror("Error", "Please select an input image or directory")
            return

        if not self.params['output_dir']:
            self.messagebox.showerror("Error", "Please select an output directory")
            return

        # Validate numeric parameters
        try:
            if not (0 <= self.params['confidence'] <= 1):
                raise ValueError("Confidence must be between 0 and 1")
            if self.params['min_area'] <= 0:
                raise ValueError("Minimum area must be positive")
            if self.params['max_area'] <= self.params['min_area']:
                raise ValueError("Maximum area must be greater than minimum area")
            if self.params['final_min_area'] < 0:
                raise ValueError("Final minimum area must be non-negative")
            if self.params['grid_size'] < 1:
                raise ValueError("Grid size must be at least 1")
            if not (0 < self.params['tile_overlap'] < 1):
                raise ValueError("Tile overlap must be between 0 and 1")
        except ValueError as e:
            self.messagebox.showerror("Invalid Parameter", str(e))
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
        import threading
        thread = threading.Thread(target=self.run_segmentation_in_gui)
        thread.daemon = True
        thread.start()

    def on_stop(self):
        """Handle Stop button click - requests segmentation to stop."""
        if self.is_running:
            self.stop_requested = True
            self.write_to_console("\n⚠️  Stop requested. Segmentation will halt after current image...\n")
            self.stop_button.config(state='disabled')

    def run_segmentation_in_gui(self):
        """Run segmentation and redirect output to console."""
        import sys

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = GUIOutputStream(self)

        try:
            # Load Mask2Former modules
            print("🔄 Loading Mask2Former and detectron2 modules...")
            ensure_mask2former_loaded(explicit_path=self.params.get('mask2former_path'))

            # Import after Mask2Former is loaded
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            from detectron2.data.detection_utils import read_image
            from mask2former import add_maskformer2_config
            print("✅ Modules loaded successfully\n")

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
                print(f"🔲 Tiled inference mode enabled (grid: {self.params['grid_size']}×{self.params['grid_size']}, overlap: {self.params['tile_overlap']*100:.0f}%)")
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
                print(f"📷 Processing single image: {input_path}")
                if tiled_segmenter:
                    tiled_segmenter.segment_image_tiled(input_path, output_dir, custom_config)
                else:
                    integration.segment_image(input_path, output_dir, custom_config)
            elif os.path.isdir(input_path):
                # Directory of images
                from pathlib import Path
                image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
                base_dir = Path(input_path)
                
                # Priority: search images/ subdirectory if it exists, otherwise search base directory
                images_subdir = base_dir / 'images'
                if images_subdir.exists() and images_subdir.is_dir():
                    search_dir = images_subdir
                    print(f"   📂 Searching images/ subdirectory")
                else:
                    search_dir = base_dir
                
                # Collect image files
                image_files_set = []
                for ext in image_extensions:
                    for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                        image_files_set.extend(search_dir.rglob(pattern))
                
                # De-duplicate using resolved absolute paths
                unique_files = {}
                for f in image_files_set:
                    try:
                        resolved = str(f.resolve())
                        if sys.platform == 'win32':
                            resolved = resolved.lower()
                        unique_files[resolved] = str(f)
                    except (OSError, RuntimeError):
                        unique_files[str(f)] = str(f)
                
                image_files = sorted(unique_files.values())

                print(f"📁 Found {len(image_files)} images in directory")

                processed_count = 0
                for i, img_path in enumerate(image_files, 1):
                    # Check if stop was requested
                    if self.stop_requested:
                        print(f"\n🛑 Segmentation stopped by user after {processed_count}/{len(image_files)} images")
                        break

                    print(f"\n{'='*60}")
                    print(f"Processing {i}/{len(image_files)}: {os.path.basename(img_path)}")
                    print(f"{'='*60}")

                    try:
                        if tiled_segmenter:
                            tiled_segmenter.segment_image_tiled(img_path, output_dir, custom_config)
                        else:
                            integration.segment_image(img_path, output_dir, custom_config)
                        processed_count += 1
                    except Exception as e:
                        print(f"❌ Error processing {os.path.basename(img_path)}: {e}")
                        continue

                if not self.stop_requested:
                    print(f"\n🎉 Batch processing complete! Processed {processed_count} images.")
                else:
                    print(f"\n✅ Partial results saved for {processed_count} processed images.")

            if not self.stop_requested:
                print(f"\n✅ All segmentation complete!")
            print(f"📂 Results saved to: {output_dir}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            # Restore stdout
            sys.stdout = old_stdout

            # Re-enable run button, disable stop button
            self.is_running = False
            self.root.after(0, lambda: self.run_button.config(state='normal'))
            self.root.after(0, lambda: self.stop_button.config(state='disabled'))

    def on_close(self):
        """Handle Close button click - returns parameters."""
        # Update parameters before closing
        self.update_params_from_gui()
        self.save_config()

        # Set result and close
        self.result = self.params
        self.root.quit()
        self.root.destroy()

    def on_cancel(self):
        """Handle Cancel button click - for backward compatibility."""
        self.result = None
        self.root.quit()
        self.root.destroy()

    def show(self):
        """Display the GUI and return selected parameters."""
        # Create main window
        self.root = self.tk.Tk()
        self.root.title("Myotube Segmentation Parameters")
        self.root.geometry("900x1000")

        # Create main container frame
        container = self.ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky=(self.tk.N, self.tk.W, self.tk.E, self.tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create canvas with scrollbar
        canvas = self.tk.Canvas(container)
        scrollbar = self.ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        # Create scrollable frame inside canvas
        main_frame = self.ttk.Frame(canvas, padding="10")

        # Configure canvas
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar and canvas
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        canvas.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

        # Create window in canvas
        canvas_frame = canvas.create_window((0, 0), window=main_frame, anchor="nw")

        # Configure scroll region when frame changes size
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        main_frame.bind("<Configure>", configure_scroll_region)

        # Bind mousewheel for scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        # Bind mousewheel to canvas and all child widgets
        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", on_mousewheel)  # Windows/MacOS
            widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux scroll up
            widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux scroll down
            for child in widget.winfo_children():
                bind_mousewheel(child)

        # Initial bind
        self.root.after(100, lambda: bind_mousewheel(main_frame))

        # Update canvas width when container is resized
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_frame, width=event.width)

        canvas.bind("<Configure>", on_canvas_configure)

        # Variables for GUI widgets
        self.input_var = self.tk.StringVar(value=self.params['input_path'])
        self.output_var = self.tk.StringVar(value=self.params['output_dir'])
        self.config_var = self.tk.StringVar(value=self.params['config'])
        self.weights_var = self.tk.StringVar(value=self.params['weights'])
        self.mask2former_path_var = self.tk.StringVar(value=self.params['mask2former_path'])
        self.confidence_var = self.tk.DoubleVar(value=self.params['confidence'])
        self.min_area_var = self.tk.IntVar(value=self.params['min_area'])
        self.max_area_var = self.tk.IntVar(value=self.params['max_area'])
        self.final_min_area_var = self.tk.IntVar(value=self.params['final_min_area'])
        self.cpu_var = self.tk.BooleanVar(value=self.params['cpu'])
        self.max_image_size_var = self.tk.StringVar(value=str(self.params['max_image_size']) if self.params['max_image_size'] else '')
        self.force_1024_var = self.tk.BooleanVar(value=self.params['force_1024'])
        self.use_tiling_var = self.tk.BooleanVar(value=self.params['use_tiling'])
        self.grid_size_var = self.tk.IntVar(value=self.params['grid_size'])
        self.tile_overlap_var = self.tk.DoubleVar(value=self.params['tile_overlap'] * 100)
        self.skip_merged_var = self.tk.BooleanVar(value=self.params['skip_merged_masks'])
        self.save_measurements_var = self.tk.BooleanVar(value=self.params['save_measurements'])

        row = 0

        # ===== Paths Section =====
        self.ttk.Label(main_frame, text="Input/Output Paths", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Label(main_frame, text="Input (Image/Directory):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.input_var, width=50).grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.ttk.Button(main_frame, text="Browse...", command=self.browse_input).grid(row=row, column=2)
        row += 1

        self.ttk.Label(main_frame, text="Output Directory:").grid(row=row, column=0, sticky=self.tk.W)
        output_entry = self.ttk.Entry(main_frame, textvariable=self.output_var, width=50)
        output_entry.grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        output_browse_btn = self.ttk.Button(main_frame, text="Browse...", command=self.browse_output)
        output_browse_btn.grid(row=row, column=2)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Model Configuration =====
        self.ttk.Label(main_frame, text="Model Configuration (Optional)", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Label(main_frame, text="Config File:").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.config_var, width=50).grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.ttk.Button(main_frame, text="Browse...", command=self.browse_config).grid(row=row, column=2)
        row += 1

        self.ttk.Label(main_frame, text="Model Weights:").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.weights_var, width=50).grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.ttk.Button(main_frame, text="Browse...", command=self.browse_weights).grid(row=row, column=2)
        row += 1

        self.ttk.Label(main_frame, text="Mask2Former Path:").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.mask2former_path_var, width=50).grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.ttk.Button(main_frame, text="Browse...", command=self.browse_mask2former_path).grid(row=row, column=2)
        row += 1

        self.ttk.Label(main_frame, text="(Leave empty for auto-detection)", font=('Arial', 9, 'italic')).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Detection Parameters =====
        self.ttk.Label(main_frame, text="Detection Parameters", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Label(main_frame, text="Confidence Threshold (0-1):").grid(row=row, column=0, sticky=self.tk.W)
        confidence_scale = self.ttk.Scale(main_frame, from_=0.0, to=1.0, variable=self.confidence_var, orient='horizontal', length=300)
        confidence_scale.grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.confidence_label = self.ttk.Label(main_frame, text=f"{self.confidence_var.get():.2f}")
        self.confidence_label.grid(row=row, column=2)
        confidence_scale.configure(command=lambda v: self.confidence_label.configure(text=f"{float(v):.2f}"))
        row += 1

        self.ttk.Label(main_frame, text="Minimum Area (pixels):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.min_area_var, width=20).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        self.ttk.Label(main_frame, text="Maximum Area (pixels):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.max_area_var, width=20).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        self.ttk.Label(main_frame, text="Final Min Area (pixels):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.final_min_area_var, width=20).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Performance Options =====
        self.ttk.Label(main_frame, text="Performance Options", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Checkbutton(main_frame, text="Use CPU (slower, less memory)", variable=self.cpu_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        self.ttk.Checkbutton(main_frame, text="Force 1024px input (memory optimization)", variable=self.force_1024_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        self.ttk.Label(main_frame, text="Max Image Size (optional):").grid(row=row, column=0, sticky=self.tk.W)
        self.ttk.Entry(main_frame, textvariable=self.max_image_size_var, width=20).grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Tiling Options =====
        self.ttk.Label(main_frame, text="Tiling Options", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Checkbutton(main_frame, text="Use tiled inference (for images with many myotubes)", variable=self.use_tiling_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        self.ttk.Label(main_frame, text="Grid Size (1=no split, 2=2×2, etc.):").grid(row=row, column=0, sticky=self.tk.W)
        grid_size_spinbox = self.ttk.Spinbox(main_frame, from_=1, to=10, textvariable=self.grid_size_var, width=10)
        grid_size_spinbox.grid(row=row, column=1, sticky=self.tk.W, padx=5)
        row += 1

        self.ttk.Label(main_frame, text="Tile Overlap (%):").grid(row=row, column=0, sticky=self.tk.W)
        tile_overlap_scale = self.ttk.Scale(main_frame, from_=10, to=50, variable=self.tile_overlap_var, orient='horizontal', length=300)
        tile_overlap_scale.grid(row=row, column=1, sticky=(self.tk.W, self.tk.E), padx=5)
        self.tile_overlap_label = self.ttk.Label(main_frame, text=f"{self.tile_overlap_var.get():.1f}")
        self.tile_overlap_label.grid(row=row, column=2)
        tile_overlap_scale.configure(command=lambda v: self.tile_overlap_label.configure(text=f"{float(v):.1f}"))
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Output Options =====
        self.ttk.Label(main_frame, text="Output Options", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        self.ttk.Checkbutton(main_frame, text="Skip merged masks (skip imaginary boundary generation)", variable=self.skip_merged_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        self.ttk.Checkbutton(main_frame, text="Save measurements CSV (includes area, length, width, etc.)", variable=self.save_measurements_var).grid(row=row, column=0, columnspan=2, sticky=self.tk.W, pady=2)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Output Console =====
        self.ttk.Label(main_frame, text="Console Output", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, sticky=self.tk.W, pady=(0, 5))
        row += 1

        # Create text widget with scrollbar
        console_frame = self.ttk.Frame(main_frame)
        console_frame.grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E, self.tk.N, self.tk.S), pady=5)

        scrollbar = self.ttk.Scrollbar(console_frame)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        self.console_text = self.tk.Text(console_frame, height=15, width=80,
                                          yscrollcommand=scrollbar.set,
                                          bg='#1e1e1e', fg='#d4d4d4',
                                          font=('Consolas', 9))
        self.console_text.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        scrollbar.config(command=self.console_text.yview)

        # Make console read-only
        self.console_text.config(state='disabled')

        # Configure row weight for console
        main_frame.rowconfigure(row, weight=1)
        row += 1

        # Separator
        self.ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        row += 1

        # ===== Buttons =====
        button_frame = self.ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.ttk.Button(button_frame, text="Restore Defaults", command=self.restore_defaults).pack(side=self.tk.LEFT, padx=5)
        self.run_button = self.ttk.Button(button_frame, text="Run Segmentation", command=self.on_run_threaded)
        self.run_button.pack(side=self.tk.LEFT, padx=5)
        self.stop_button = self.ttk.Button(button_frame, text="Stop", command=self.on_stop, state='disabled')
        self.stop_button.pack(side=self.tk.LEFT, padx=5)
        self.ttk.Button(button_frame, text="Close", command=self.on_close).pack(side=self.tk.LEFT, padx=5)

        # Configure column weights
        main_frame.columnconfigure(1, weight=1)

        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

        # Run the GUI
        self.root.mainloop()

        return self.result


def main():
    """Main function for command-line usage."""
    
    # Register datasets like demo.py does (if register_two_stage_datasets exists)
    try:
        from register_two_stage_datasets import register_two_stage_datasets
        register_two_stage_datasets(
            dataset_root="./myotube_batch_output", 
            register_instance=True, 
            register_panoptic=False
        )
    except ImportError:
        # Dataset registration not available, continue without it
        pass
    
    parser = argparse.ArgumentParser(description="Myotube Instance Segmentation for Fiji")

    # GUI mode
    parser.add_argument("--gui", action="store_true",
                       help="Launch graphical user interface for parameter configuration")
    parser.add_argument("--gui-output", type=str, default=None,
                       help="Lock output directory when using GUI (used by Fiji integration)")

    # Positional arguments (optional when using --gui)
    parser.add_argument("input_path", nargs='?', help="Path to input image or directory containing images")
    parser.add_argument("output_dir", nargs='?', help="Output directory for results")

    # Optional arguments
    parser.add_argument("--config", help="Path to model config file")
    parser.add_argument("--weights", help="Path to model weights")
    parser.add_argument("--mask2former-path", type=str, default=None,
                       help="Path to Mask2Former project directory (auto-detected if not specified)")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold for detection (default: 0.25)")
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum myotube area in pixels")
    parser.add_argument("--max-area", type=int, default=50000,
                       help="Maximum myotube area in pixels")
    parser.add_argument("--final-min-area", type=int, default=1000,
                       help="Final minimum area filter (applied after post-processing)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU inference (slower but uses less memory)")
    parser.add_argument("--max-image-size", type=int, default=None,
                       help="Maximum image dimension (larger images will be resized). Default: respect training resolution")
    parser.add_argument("--force-1024", action="store_true",
                       help="Force 1024px input resolution for memory optimization (may reduce accuracy)")

    # Tiling parameters
    parser.add_argument("--use-tiling", action="store_true", default=True,
                       help="Enable tiled inference for large images with many myotubes (enabled by default)")
    parser.add_argument("--no-tiling", dest="use_tiling", action="store_false",
                       help="Disable tiled inference and process entire image at once")
    parser.add_argument("--grid-size", type=int, default=2,
                       help="Grid size for tiling (1=no split, 2=2×2, 3=3×3, etc.). Default: 2")
    parser.add_argument("--tile-overlap", type=float, default=0.20,
                       help="Overlap ratio between tiles (default: 0.20 = 20%%). Only used with --use-tiling")

    # Merged mask generation parameter
    parser.add_argument("--skip-merged-masks", action="store_true", default=True,
                       help="Skip generation of merged visualization masks (imaginary boundaries connecting disconnected components, skipped by default)")
    parser.add_argument("--generate-merged-masks", dest="skip_merged_masks", action="store_false",
                       help="Generate merged visualization masks with imaginary boundaries")

    # Measurements CSV generation parameter
    parser.add_argument("--save-measurements", action="store_true", default=False,
                       help="Save comprehensive measurements CSV (area, length, width, etc. - disabled by default)")

    args = parser.parse_args()

    # Check if GUI mode is requested
    if args.gui:
        print("🖥️  Launching GUI...")
        # If gui-output is specified, lock the output directory
        gui = ParameterGUI(locked_output_dir=args.gui_output)
        params = gui.show()

        if params is None:
            print("❌ User cancelled")
            return

        print("✅ Parameters selected via GUI")

        # Override args with GUI parameters
        args.input_path = params['input_path']
        # Use locked output if provided, otherwise use GUI selection
        args.output_dir = args.gui_output if args.gui_output else params['output_dir']
        args.config = params['config'] if params['config'] else None
        args.weights = params['weights'] if params['weights'] else None
        args.mask2former_path = params['mask2former_path'] if params['mask2former_path'] else None
        args.confidence = params['confidence']
        args.min_area = params['min_area']
        args.max_area = params['max_area']
        args.final_min_area = params['final_min_area']
        args.cpu = params['cpu']
        args.force_1024 = params['force_1024']
        args.use_tiling = params['use_tiling']
        args.grid_size = params['grid_size']
        args.tile_overlap = params['tile_overlap']
        args.skip_merged_masks = params['skip_merged_masks']
        args.save_measurements = params['save_measurements']
        args.max_image_size = params['max_image_size'] if params['max_image_size'] else None

        # Now that GUI is done, load the imports we skipped earlier
        print("🔄 Loading Mask2Former and detectron2 modules...")
        ensure_mask2former_loaded(explicit_path=args.mask2former_path)
        # Re-import the modules that were set to None in GUI mode
        global DefaultPredictor, get_cfg, add_deeplab_config, read_image, add_maskformer2_config
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.data.detection_utils import read_image
        from mask2former import add_maskformer2_config
        print("✅ Modules loaded successfully")
    else:
        # Command-line mode - validate required arguments
        if not args.input_path or not args.output_dir:
            parser.error("input_path and output_dir are required unless using --gui mode")

    # Print parameter summary
    print("\n" + "="*60)
    print("MYOTUBE SEGMENTATION PARAMETERS")
    print("="*60)
    print(f"Input:           {args.input_path}")
    print(f"Output:          {args.output_dir}")
    print(f"Config:          {args.config or 'Auto-detect'}")
    print(f"Weights:         {args.weights or 'Auto-detect'}")
    print(f"Confidence:      {args.confidence}")
    print(f"Min Area:        {args.min_area} px")
    print(f"Max Area:        {args.max_area} px")
    print(f"Final Min Area:  {args.final_min_area} px")
    print(f"CPU Mode:        {args.cpu}")
    print(f"Force 1024px:    {args.force_1024}")
    print(f"Max Image Size:  {args.max_image_size or 'Auto'}")
    print(f"Use Tiling:      {args.use_tiling}")
    print(f"Tile Overlap:    {args.tile_overlap*100:.0f}%")
    print(f"Skip Merged:     {args.skip_merged_masks}")
    print("="*60 + "\n")
    
    # Custom post-processing config
    max_image_size = 1024 if args.force_1024 else args.max_image_size
    custom_config = {
        'confidence_threshold': args.confidence,
        'min_area': args.min_area,
        'max_area': args.max_area,
        'final_min_area': args.final_min_area,
        'max_image_size': max_image_size,
        'force_cpu': args.cpu,
        'save_measurements': args.save_measurements
    }
    
    # Initialize integration
    integration = MyotubeFijiIntegration(
        config_file=args.config,
        model_weights=args.weights,
        skip_merged_masks=args.skip_merged_masks,
        mask2former_path=args.mask2former_path
    )

    # Initialize tiled segmentation if requested
    if args.use_tiling:
        print(f"🔲 Tiled inference mode enabled (grid: {args.grid_size}×{args.grid_size}, overlap: {args.tile_overlap*100:.0f}%)")
        tiled_segmenter = TiledMyotubeSegmentation(
            segmentation_backend=integration,
            target_overlap=args.tile_overlap,
            grid_size=args.grid_size
        )
    else:
        tiled_segmenter = None

    # Apply memory optimization settings
    if args.cpu:
        print("🖥️  CPU inference mode enabled")
    if args.force_1024:
        print("📏 Forced 1024px input resolution (memory optimization - may reduce accuracy)")
    elif args.max_image_size:
        print(f"📏 Max image size set to: {args.max_image_size}px")
    else:
        print("📏 Using training resolution (1500px) for best accuracy")
    
    # Ensure output directory exists for status files
    os.makedirs(args.output_dir, exist_ok=True)

    # Clean up old status files to avoid confusion from previous runs
    old_success_file = os.path.join(args.output_dir, "BATCH_SUCCESS")
    old_error_file = os.path.join(args.output_dir, "ERROR")
    if os.path.exists(old_success_file):
        os.remove(old_success_file)
        print("🗑️  Deleted old BATCH_SUCCESS file")
    if os.path.exists(old_error_file):
        os.remove(old_error_file)
        print("🗑️  Deleted old ERROR file")

    # Process image(s)
    try:
        # Check if input is a directory or single image
        if os.path.isdir(args.input_path):
            # Batch processing mode
            print(f"📁 Batch processing mode: {args.input_path}")
            
            # Find all image files in directory
            image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
            base_dir = Path(args.input_path)
            
            # Priority: search images/ subdirectory if it exists, otherwise search base directory
            images_subdir = base_dir / 'images'
            if images_subdir.exists() and images_subdir.is_dir():
                # Use images/ subdirectory (COCO format)
                search_dir = images_subdir
                print(f"   📂 Searching images/ subdirectory: {images_subdir}")
            else:
                # Use base directory
                search_dir = base_dir
            
            # Collect image files (recursive search)
            image_files = []
            for ext in image_extensions:
                # Search case-insensitively by checking both lowercase and uppercase
                # Use set to avoid case-insensitive duplicates on Windows
                for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                    image_files.extend(search_dir.rglob(pattern))
            
            # De-duplicate using resolved absolute paths (handles symlinks, UNC paths, etc.)
            unique_files = {}
            for f in image_files:
                # Resolve to canonical absolute path
                try:
                    resolved = str(f.resolve())
                    # On Windows, normalize to handle case-insensitive filesystem
                    if sys.platform == 'win32':
                        resolved = resolved.lower()
                    unique_files[resolved] = f
                except (OSError, RuntimeError):
                    # Handle edge cases with inaccessible files
                    unique_files[str(f)] = f
            
            image_files = sorted(unique_files.values(), key=lambda p: str(p))
            
            if not image_files:
                msg = (f"No image files found in directory: {args.input_path}\n"
                       f"Searched: {search_dir}\n"
                       f"Supported: {', '.join(image_extensions)}")
                print(f"❌ {msg}")
                # Write error status for Fiji macro
                with open(os.path.join(args.output_dir, "ERROR"), 'w') as f:
                    f.write(msg + "\n")
                return 1
            
            print(f"📊 Found {len(image_files)} image files to process")

            # Process each image
            total_myotubes = 0
            successful_images = 0
            failed_images = 0

            for i, image_path in enumerate(image_files, 1):
                print(f"\n{'='*60}")
                print(f"🖼️  Processing image {i}/{len(image_files)}: {Path(image_path).name}")

                # Get relative path from search directory to preserve folder structure
                try:
                    rel_path = Path(image_path).relative_to(search_dir)
                except ValueError:
                    # If not relative, just use the filename
                    rel_path = Path(image_path).name

                print(f"   Relative path: {rel_path}")
                print(f"{'='*60}")

                try:
                    # Create subdirectory for each image's output, preserving folder structure
                    # Structure: output_dir/parent_folders/image_name/
                    base_name = Path(image_path).stem
                    parent_rel_path = Path(rel_path).parent

                    # Create folder: output_dir/parent_folders/image_name/
                    image_output_dir = os.path.join(args.output_dir, str(parent_rel_path), base_name)

                    print(f"   📂 Output folder: {image_output_dir}")

                    # Use tiled or standard segmentation based on flag
                    if tiled_segmenter:
                        output_files = tiled_segmenter.segment_image_tiled(
                            str(image_path),
                            image_output_dir,
                            custom_config
                        )
                    else:
                        output_files = integration.segment_image(
                            str(image_path),
                            image_output_dir,
                            custom_config
                        )

                    myotube_count = output_files['count']
                    total_myotubes += myotube_count
                    successful_images += 1

                    print(f"✅ {Path(image_path).name}: {myotube_count} myotubes detected")

                except Exception as e:
                    print(f"❌ Failed to process {Path(image_path).name}: {e}")
                    failed_images += 1
                    continue
            
            # Create batch summary
            print(f"\n{'='*60}")
            print(f"🎉 BATCH PROCESSING COMPLETED!")
            print(f"{'='*60}")
            print(f"📊 Total images processed: {successful_images}/{len(image_files)}")
            print(f"📊 Failed images: {failed_images}")
            print(f"📊 Total myotubes detected: {total_myotubes}")
            print(f"📁 Output directory: {args.output_dir}")
            print(f"{'='*60}")
            
            # Write batch summary file
            summary_file = os.path.join(args.output_dir, "BATCH_SUCCESS")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"{successful_images}/{len(image_files)} images processed\n")
                f.write(f"{total_myotubes} total myotubes detected\n")
                f.write(f"{failed_images} failed images\n")
            
            output_files = {
                'count': total_myotubes,
                'processed_images': successful_images,
                'failed_images': failed_images,
                'total_images': len(image_files)
            }
            
        else:
            msg = f"Input path must be a directory containing images: {args.input_path}"
            print(f"❌ {msg}")
            # Write error status for Fiji macro
            with open(os.path.join(args.output_dir, "ERROR"), 'w') as f:
                f.write(msg + "\n")
            return 1
        
        # Signal success to ImageJ macro
        success_file = os.path.join(args.output_dir, "BATCH_SUCCESS")
        print(f"📝 Batch processing completed. Summary written to: {success_file}")
        
    except BaseException as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Signal failure to ImageJ macro
        error_file = os.path.join(args.output_dir, "ERROR")
        with open(error_file, 'w') as f:
            f.write(f"ERROR: {str(e)}\n")
        
        return 1


if __name__ == "__main__":
    main()
