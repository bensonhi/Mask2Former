"""
Max Projection tab for processing separate myotube and nucleus image folders.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import threading
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import tifffile
from skimage import io

from fiji_integration.gui.base_tab import TabInterface


class MaxProjectionProcessor:
    """Processor for max projection of separate myotube and nucleus folders."""

    def __init__(self, myotube_input_dir: str, nucleus_input_dir: str,
                 output_dir: str, progress_callback=None):
        """
        Initialize the processor.

        Args:
            myotube_input_dir: Input directory containing myotube images
            nucleus_input_dir: Input directory containing nucleus images
            output_dir: Output directory for processed images
            progress_callback: Callback function to report progress
        """
        self.myotube_input_dir = Path(myotube_input_dir) if myotube_input_dir else None
        self.nucleus_input_dir = Path(nucleus_input_dir) if nucleus_input_dir else None
        self.output_dir = Path(output_dir)
        self.progress_callback = progress_callback

        # Output subdirectories
        self.myotube_output_dir = self.output_dir / "myotube_max_projection"
        self.nucleus_output_dir = self.output_dir / "nucleus_max_projection"

        # Statistics
        self.stats = {
            'myotube_processed': 0,
            'myotube_skipped': 0,
            'myotube_errors': 0,
            'nucleus_processed': 0,
            'nucleus_skipped': 0,
            'nucleus_errors': 0,
        }

    def log(self, message: str):
        """Log a message via callback or print."""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message, flush=True)

    def find_images(self, input_dir: Path) -> List[Path]:
        """
        Find all TIFF images in input directory recursively.

        Args:
            input_dir: Directory to search

        Returns:
            List of image paths
        """
        image_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        image_files = []

        for ext in image_extensions:
            image_files.extend(input_dir.rglob(f'*{ext}'))

        return sorted(image_files)

    def apply_max_projection(self, image: np.ndarray) -> np.ndarray:
        """
        Apply max intensity projection along Z-axis.

        Args:
            image: Image with shape (Z, Y, X) or (Y, X)

        Returns:
            Max projected image with shape (Y, X)
        """
        if image.ndim == 2:
            # Already 2D
            return image
        elif image.ndim == 3:
            # 3D - apply max projection along Z-axis
            return np.max(image, axis=0)
        else:
            raise ValueError(f"Unexpected image dimensions: {image.ndim}")

    def process_image(self, image_path: Path, input_dir: Path, output_dir: Path,
                      channel_name: str) -> bool:
        """
        Process a single image: apply max projection and save.

        Args:
            image_path: Path to input image
            input_dir: Base input directory (for preserving structure)
            output_dir: Base output directory
            channel_name: Name of channel for logging ('myotube' or 'nucleus')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get relative path to preserve structure
            rel_path = image_path.relative_to(input_dir)
            base_name = image_path.stem
            extension = image_path.suffix

            # Construct output path with MAX_ prefix
            output_rel_path = rel_path.parent / f"MAX_{base_name}{extension}"
            output_path = output_dir / output_rel_path

            # Check if output already exists
            if output_path.exists():
                self.log(f"  Skipping (exists): {rel_path}")
                return True  # Return True but mark as skipped

            self.log(f"  Processing: {rel_path}")

            # Load image
            image = tifffile.imread(str(image_path))
            self.log(f"    Input shape: {image.shape}")

            # Apply max projection if needed
            if image.ndim == 2:
                # Already 2D
                max_proj = image
                self.log(f"    Already 2D, no projection needed")
            elif image.ndim == 3:
                # Z-stack - apply max projection
                max_proj = self.apply_max_projection(image)
                self.log(f"    Applied max projection: {image.shape} → {max_proj.shape}")
            else:
                self.log(f"    Warning: Unsupported image dimensions: {image.ndim}")
                return False

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(output_path), max_proj)
            self.log(f"    ✓ Saved: {output_path.name}")

            return True

        except Exception as e:
            self.log(f"    ✗ Error processing {image_path.name}: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False

    def process_folder(self, input_dir: Path, output_dir: Path, channel_name: str):
        """
        Process all images in a folder.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            channel_name: Name of channel ('myotube' or 'nucleus')
        """
        if not input_dir or not input_dir.exists():
            self.log(f"\n[WARNING] Skipping {channel_name}: Input directory not specified or doesn't exist")
            return

        self.log(f"\n{'=' * 80}")
        self.log(f"PROCESSING {channel_name.upper()} IMAGES")
        self.log(f"{'=' * 80}")
        self.log(f"Input directory:  {input_dir}")
        self.log(f"Output directory: {output_dir}")
        self.log(f"{'=' * 80}")

        # Find all images
        self.log("\nSearching for images...")
        image_files = self.find_images(input_dir)
        self.log(f"Found {len(image_files)} TIFF images\n")

        if len(image_files) == 0:
            self.log(f"No images found in {channel_name} folder.")
            return

        # Process each image
        processed = 0
        skipped = 0
        errors = 0

        for i, image_path in enumerate(image_files, 1):
            self.log(f"[{i}/{len(image_files)}]")

            # Check if already exists before processing
            rel_path = image_path.relative_to(input_dir)
            output_path = output_dir / rel_path.parent / f"MAX_{image_path.stem}{image_path.suffix}"

            if output_path.exists():
                skipped += 1
            else:
                success = self.process_image(image_path, input_dir, output_dir, channel_name)
                if success:
                    processed += 1
                else:
                    errors += 1

        # Update stats
        if channel_name == 'myotube':
            self.stats['myotube_processed'] = processed
            self.stats['myotube_skipped'] = skipped
            self.stats['myotube_errors'] = errors
        else:
            self.stats['nucleus_processed'] = processed
            self.stats['nucleus_skipped'] = skipped
            self.stats['nucleus_errors'] = errors

        # Print folder summary
        self.log(f"\n{channel_name.upper()} FOLDER SUMMARY:")
        self.log(f"  Processed: {processed}")
        self.log(f"  Skipped:   {skipped}")
        self.log(f"  Errors:    {errors}")

    def process_all(self):
        """Process all images in both folders."""
        self.log("=" * 80)
        self.log("MAX PROJECTION PROCESSING")
        self.log("=" * 80)

        # Process myotube folder
        if self.myotube_input_dir:
            self.process_folder(self.myotube_input_dir, self.myotube_output_dir, 'myotube')
        else:
            self.log("\n[WARNING] Myotube input directory not specified, skipping")

        # Process nucleus folder
        if self.nucleus_input_dir:
            self.process_folder(self.nucleus_input_dir, self.nucleus_output_dir, 'nucleus')
        else:
            self.log("\n[WARNING] Nucleus input directory not specified, skipping")

        # Print overall summary
        self.log("\n" + "=" * 80)
        self.log("OVERALL SUMMARY")
        self.log("=" * 80)
        self.log(f"MYOTUBE:")
        self.log(f"  Processed: {self.stats['myotube_processed']}")
        self.log(f"  Skipped:   {self.stats['myotube_skipped']}")
        self.log(f"  Errors:    {self.stats['myotube_errors']}")
        self.log(f"\nNUCLEUS:")
        self.log(f"  Processed: {self.stats['nucleus_processed']}")
        self.log(f"  Skipped:   {self.stats['nucleus_skipped']}")
        self.log(f"  Errors:    {self.stats['nucleus_errors']}")
        self.log("=" * 80)


class MaxProjectionTab(TabInterface):
    """Tab for max projection of separate myotube and nucleus folders."""

    def __init__(self, config_file=None):
        """
        Initialize the max projection tab.

        Args:
            config_file: Path to config file (default: auto-detect)
        """
        super().__init__()

        # Config file location
        if config_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fiji_integration_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            config_file = os.path.join(fiji_integration_dir, '.max_projection_gui_config.json')
        self.config_file = config_file

        # Default parameters
        home_dir = os.path.expanduser('~')
        workflow_base = os.path.join(home_dir, 'fiji_workflow')

        self.default_params = {
            'myotube_input_dir': os.path.join(workflow_base, 'raw_myotube_images'),
            'nucleus_input_dir': os.path.join(workflow_base, 'raw_nucleus_images'),
            'output_dir': os.path.join(workflow_base, '1_max_projection'),
        }

        # Load saved parameters or use defaults
        self.params = self.load_config()

        # GUI widgets (will be created in build_ui)
        self.myotube_input_var = None
        self.nucleus_input_var = None
        self.output_dir_var = None
        self.run_button = None
        self.stop_button = None

    def get_tab_name(self) -> str:
        return "Max Projection"

    def load_config(self):
        """Load saved configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                # Merge with defaults to handle new parameters
                params = self.default_params.copy()
                params.update(saved)
                print(f"[LOADED] Loaded saved Max Projection configuration from: {self.config_file}")
                return params
            except Exception as e:
                print(f"[WARNING]  Could not load config file: {e}")
                return self.default_params.copy()
        else:
            return self.default_params.copy()

    def save_config(self, config=None):
        """Save configuration to file."""
        if config is None:
            config = self.params
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[SAVED] Saved configuration to: {self.config_file}")
        except Exception as e:
            print(f"[WARNING]  Could not save config file: {e}")

    def validate_parameters(self):
        """Validate current parameters."""
        myotube_input = self.params['myotube_input_dir'].strip()
        nucleus_input = self.params['nucleus_input_dir'].strip()
        output_dir = self.params['output_dir'].strip()

        self.log(f"[VALIDATING] Validating parameters...")
        self.log(f"   Myotube input: {myotube_input}")
        self.log(f"   Nucleus input: {nucleus_input}")
        self.log(f"   Output dir: {output_dir}")

        # At least one input must be specified
        if not myotube_input and not nucleus_input:
            return False, "Please specify at least one input directory (myotube or nucleus)"

        if not output_dir:
            return False, "Please select an output directory"

        # Check that specified directories exist
        if myotube_input and not os.path.exists(myotube_input):
            return False, f"Myotube input directory does not exist: {myotube_input}"

        if nucleus_input and not os.path.exists(nucleus_input):
            return False, f"Nucleus input directory does not exist: {nucleus_input}"

        self.log(f"[OK] Parameters validated")
        return True, None

    def build_ui(self, parent_frame, console_text):
        """Build the UI for this tab."""
        self.console_text = console_text

        # Create tkinter variables from saved params
        self.myotube_input_var = tk.StringVar(value=self.params['myotube_input_dir'])
        self.nucleus_input_var = tk.StringVar(value=self.params['nucleus_input_dir'])
        self.output_dir_var = tk.StringVar(value=self.params['output_dir'])

        # Input/Output Section
        io_frame = ttk.LabelFrame(parent_frame, text="Input/Output Folders", padding=10)
        io_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Myotube input directory
        ttk.Label(io_frame, text="Myotube Input:").grid(row=0, column=0, sticky=tk.W, pady=2)
        myotube_entry = ttk.Entry(io_frame, textvariable=self.myotube_input_var, width=50)
        myotube_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_myotube_input).grid(row=0, column=2, padx=5, pady=2)

        # Nucleus input directory
        ttk.Label(io_frame, text="Nucleus Input:").grid(row=1, column=0, sticky=tk.W, pady=2)
        nucleus_entry = ttk.Entry(io_frame, textvariable=self.nucleus_input_var, width=50)
        nucleus_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_nucleus_input).grid(row=1, column=2, padx=5, pady=2)

        # Output directory
        ttk.Label(io_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=2)
        output_entry = ttk.Entry(io_frame, textvariable=self.output_dir_var, width=50)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_output_dir).grid(row=2, column=2, padx=5, pady=2)

        io_frame.columnconfigure(1, weight=1)

        # Info Section
        info_frame = ttk.LabelFrame(parent_frame, text="Processing Information", padding=10)
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        info_text = """This tool applies max intensity projection to Z-stack images:

1. Provide two separate input folders:
   - Myotube folder: Contains myotube/cytoplasm images
   - Nucleus folder: Contains nucleus images

2. Processing:
   - Scans all subdirectories for TIFF files
   - Applies max intensity projection to Z-stacks
   - Copies 2D images as-is (no projection needed)

3. Output structure:
   - myotube_max_projection/ (processed myotube images)
   - nucleus_max_projection/ (processed nucleus images)
   - Original folder structure preserved

4. Output files: MAX_{original_name}.tif

Note: You can specify just one folder if you only need to process one channel.
"""
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.grid(row=0, column=0, sticky=tk.W)

        # Create buttons in button frame
        self.run_button = ttk.Button(self.button_frame, text="Run Max Projection", command=self.on_run_threaded)
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.on_stop, state='disabled')
        self.restore_button = ttk.Button(self.button_frame, text="Restore Defaults", command=self.restore_defaults)

    def get_button_frame_widgets(self):
        """Return list of (button, side) tuples for button frame."""
        return [
            (self.restore_button, tk.LEFT),
            (self.run_button, tk.LEFT),
            (self.stop_button, tk.LEFT)
        ]

    def browse_myotube_input(self):
        """Browse for myotube input directory."""
        folder = filedialog.askdirectory(title="Select Myotube Input Directory")
        if folder:
            self.myotube_input_var.set(folder)

    def browse_nucleus_input(self):
        """Browse for nucleus input directory."""
        folder = filedialog.askdirectory(title="Select Nucleus Input Directory")
        if folder:
            self.nucleus_input_var.set(folder)

    def browse_output_dir(self):
        """Browse for output directory."""
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_dir_var.set(folder)

    def update_params_from_gui(self):
        """Update parameters from GUI widgets."""
        self.params['myotube_input_dir'] = self.myotube_input_var.get()
        self.params['nucleus_input_dir'] = self.nucleus_input_var.get()
        self.params['output_dir'] = self.output_dir_var.get()

    def update_gui_from_params(self):
        """Update GUI widgets from current parameters."""
        self.myotube_input_var.set(self.params['myotube_input_dir'])
        self.nucleus_input_var.set(self.params['nucleus_input_dir'])
        self.output_dir_var.set(self.params['output_dir'])

    def restore_defaults(self):
        """Restore all parameters to default values."""
        self.params = self.default_params.copy()
        self.update_gui_from_params()
        self.log("[OK] Restored parameters to defaults")

    def on_run_threaded(self):
        """Run processing in a separate thread."""
        if self.is_running:
            return

        # Update params from GUI
        self.update_params_from_gui()

        # Validate inputs
        valid, error_msg = self.validate_parameters()
        if not valid:
            self.log(f"Error: {error_msg}")
            return

        # Save configuration
        self.save_config()

        myotube_input = self.params['myotube_input_dir']
        nucleus_input = self.params['nucleus_input_dir']
        output_dir = self.params['output_dir']

        # Update UI
        self.is_running = True
        self.stop_requested = False
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Start processing thread
        thread = threading.Thread(
            target=self.run_processing,
            args=(myotube_input, nucleus_input, output_dir),
            daemon=True
        )
        thread.start()

    def run_processing(self, myotube_input, nucleus_input, output_dir):
        """Run the max projection processing."""
        try:
            self.log(f"\n[STARTING] Starting max projection processing...")
            self.log(f"   Myotube input: {myotube_input}")
            self.log(f"   Nucleus input: {nucleus_input}")
            self.log(f"   Output: {output_dir}")

            # Create processor
            processor = MaxProjectionProcessor(
                myotube_input_dir=myotube_input,
                nucleus_input_dir=nucleus_input,
                output_dir=output_dir,
                progress_callback=self.log
            )

            # Run processing
            processor.process_all()

            self.log("\n[OK] Max projection processing complete!")

        except Exception as e:
            self.log(f"\n[ERROR] Error during processing: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            # Reset UI state
            self.is_running = False
            self.run_button.config(state='normal')
            self.stop_button.config(state='disabled')

    def on_stop(self):
        """Stop the processing."""
        if self.is_running:
            self.stop_requested = True
            self.log("\nStop requested...")

    def log(self, message: str):
        """Log a message to the console."""
        self.write_to_console(message + "\n")
