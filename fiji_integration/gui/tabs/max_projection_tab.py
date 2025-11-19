"""
Max Projection tab for channel splitting and max projection.
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
    """Processor for channel splitting and max projection."""

    def __init__(self, input_dir: str, output_dir: str, progress_callback=None):
        """
        Initialize the processor.

        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for processed images
            progress_callback: Callback function to report progress
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.progress_callback = progress_callback

        # Output subdirectories
        self.myotube_dir = self.output_dir / "myotube_channel"
        self.nuclei_dir = self.output_dir / "nuclei_channel"

        # Statistics
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0
        }

    def log(self, message: str):
        """Log a message via callback or print."""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message, flush=True)

    def find_images(self) -> List[Path]:
        """
        Find all TIFF images in input directory recursively.

        Returns:
            List of image paths
        """
        image_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        image_files = []

        for ext in image_extensions:
            image_files.extend(self.input_dir.rglob(f'*{ext}'))

        return sorted(image_files)

    def identify_channels(self, image_path: Path, image: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """
        Identify grey and blue/nuclei channels using ImageJ LUT metadata.

        Reads the LUT names from ImageJ TIFF metadata to determine which channel
        is grey (myotubes) and which is blue (nuclei).

        Args:
            image_path: Path to the image file (for reading metadata)
            image: Multi-channel image with shape (C, Z, Y, X) or (C, Y, X)

        Returns:
            Tuple of (grey_channel_idx, blue_channel_idx)
            Returns None for channels that aren't found
        """
        num_channels = image.shape[0]

        if num_channels == 1:
            # Single channel - ImageJ macro saves as both grey and blue
            return 0, 0

        # Try to read ImageJ metadata to get LUT names
        grey_idx = None
        blue_idx = None

        try:
            with tifffile.TiffFile(str(image_path)) as tif:
                if tif.is_imagej and hasattr(tif, 'imagej_metadata'):
                    metadata = tif.imagej_metadata
                    if 'Info' in metadata:
                        info_str = metadata['Info']

                        # Parse the Info string to find LUTName for each channel
                        for ch_idx in range(num_channels):
                            # Look for pattern: ChannelDescription #N|LUTName = XXX
                            pattern = f'ChannelDescription #{ch_idx}\\|LUTName = (\\w+)'
                            match = re.search(pattern, info_str)

                            if match:
                                lut_name = match.group(1)
                                self.log(f"    Channel {ch_idx}: LUTName = {lut_name}")

                                # Identify based on LUT name
                                if lut_name.lower() in ['gray', 'grey', 'grays']:
                                    grey_idx = ch_idx
                                    self.log(f"    → Identified as grey channel (LUT: {lut_name})")
                                elif lut_name.lower() in ['blue', 'cyan']:
                                    blue_idx = ch_idx
                                    self.log(f"    → Identified as blue channel (LUT: {lut_name})")
                                elif lut_name.lower() in ['green', 'lime']:
                                    # Some microscopes use green for nuclei
                                    if blue_idx is None:
                                        blue_idx = ch_idx
                                        self.log(f"    → Identified as blue/nuclei channel (LUT: {lut_name})")
        except Exception as e:
            self.log(f"    Could not read ImageJ metadata: {e}")

        # If we found channels via metadata, return them
        if grey_idx is not None or blue_idx is not None:
            if grey_idx is None:
                self.log(f"    [WARNING]  No grey channel found in metadata - will skip grey output")
            if blue_idx is None:
                self.log(f"    [WARNING]  No blue channel found in metadata - will skip blue output")
            return grey_idx, blue_idx

        # Fallback: Use intensity-based heuristics if metadata not available
        self.log(f"    No LUT metadata found, using intensity-based heuristics...")

        for ch_idx in range(num_channels):
            if image.ndim == 3:  # (C, Y, X)
                sample = image[ch_idx]
            else:  # (C, Z, Y, X) - already max projected
                sample = image[ch_idx]

            # Calculate statistics on non-zero pixels
            nonzero_pixels = sample[sample > 0]
            if len(nonzero_pixels) == 0:
                continue

            mean_val = np.mean(nonzero_pixels)
            std_val = np.std(nonzero_pixels)
            cv = std_val / (mean_val + 1e-6)  # Coefficient of variation

            self.log(f"    Channel {ch_idx}: mean={mean_val:.1f}, std={std_val:.1f}, CV={cv:.3f}")

            # Grey channels have lower CV (more uniform)
            # Nuclei channels have higher CV (more punctate)
            if grey_idx is None and cv < 1.0:
                grey_idx = ch_idx
                self.log(f"    → Identified as grey channel (uniform)")
            elif blue_idx is None and ch_idx != grey_idx and cv > 0.5:
                blue_idx = ch_idx
                self.log(f"    → Identified as blue channel (punctate)")

        # Final fallback: use intensity
        if grey_idx is None and blue_idx is None and num_channels >= 2:
            mean_intensities = [(ch_idx, np.mean(image[ch_idx])) for ch_idx in range(num_channels)]
            sorted_by_intensity = sorted(mean_intensities, key=lambda x: x[1], reverse=True)
            grey_idx = sorted_by_intensity[0][0]
            blue_idx = sorted_by_intensity[-1][0] if num_channels > 2 else sorted_by_intensity[1][0]
            self.log(f"    Fallback: grey={grey_idx}, blue={blue_idx} (by intensity)")

        if grey_idx is None:
            self.log(f"    [WARNING]  No grey channel identified - will skip grey output")
        if blue_idx is None:
            self.log(f"    [WARNING]  No blue channel identified - will skip blue output")

        return grey_idx, blue_idx

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

    def process_image(self, image_path: Path) -> bool:
        """
        Process a single image: split channels, apply max projection, and save.

        Args:
            image_path: Path to input image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get relative path to preserve structure
            rel_path = image_path.relative_to(self.input_dir)
            base_name = image_path.stem
            extension = image_path.suffix

            # Construct output paths
            myotube_rel_path = rel_path.parent / f"MAX_{base_name}_grey{extension}"
            nuclei_rel_path = rel_path.parent / f"MAX_{base_name}_blue{extension}"

            myotube_output = self.myotube_dir / myotube_rel_path
            nuclei_output = self.nuclei_dir / nuclei_rel_path

            # Check if outputs already exist
            both_exist = myotube_output.exists() and nuclei_output.exists()
            if both_exist:
                self.log(f"  Skipping (both outputs exist): {rel_path}")
                self.stats['skipped'] += 1
                return True

            self.log(f"  Processing: {rel_path}")

            # Load image
            image = tifffile.imread(str(image_path))
            self.log(f"    Shape: {image.shape}")

            # Determine image format
            # Possible formats: (Y, X), (Z, Y, X), (C, Y, X), (C, Z, Y, X)

            if image.ndim == 2:
                # Single 2D image - save as both grey and blue (matching ImageJ behavior)
                self.log(f"    Single channel image - saving as both grey and blue")

                if not myotube_output.exists():
                    myotube_output.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(str(myotube_output), image)
                    self.log(f"    Saved grey: {myotube_output}")

                if not nuclei_output.exists():
                    nuclei_output.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(str(nuclei_output), image)
                    self.log(f"    Saved blue: {nuclei_output}")

            elif image.ndim == 3:
                # Could be (Z, Y, X) or (C, Y, X)
                # Heuristic: if first dimension is small (<=4), likely channels
                if image.shape[0] <= 4:
                    # Multi-channel 2D image (C, Y, X)
                    self.log(f"    Multi-channel 2D image: {image.shape[0]} channels")
                    grey_idx, blue_idx = self.identify_channels(image_path, image)

                    # Save grey channel if identified
                    if grey_idx is not None and not myotube_output.exists():
                        grey_channel = image[grey_idx]
                        myotube_output.parent.mkdir(parents=True, exist_ok=True)
                        tifffile.imwrite(str(myotube_output), grey_channel)
                        self.log(f"    Saved grey channel {grey_idx}: {myotube_output}")
                    elif grey_idx is None:
                        self.log(f"    Skipping grey output (no grey channel found)")

                    # Save blue channel if identified
                    if blue_idx is not None and not nuclei_output.exists():
                        blue_channel = image[blue_idx]
                        nuclei_output.parent.mkdir(parents=True, exist_ok=True)
                        tifffile.imwrite(str(nuclei_output), blue_channel)
                        self.log(f"    Saved blue channel {blue_idx}: {nuclei_output}")
                    elif blue_idx is None:
                        self.log(f"    Skipping blue output (no blue channel found)")
                else:
                    # Z-stack (Z, Y, X) - single channel
                    self.log(f"    Single channel Z-stack: {image.shape[0]} slices")
                    max_proj = self.apply_max_projection(image)

                    # Save as both grey and blue (matching ImageJ behavior for single channel)
                    if not myotube_output.exists():
                        myotube_output.parent.mkdir(parents=True, exist_ok=True)
                        tifffile.imwrite(str(myotube_output), max_proj)
                        self.log(f"    Saved grey (max proj): {myotube_output}")

                    if not nuclei_output.exists():
                        nuclei_output.parent.mkdir(parents=True, exist_ok=True)
                        tifffile.imwrite(str(nuclei_output), max_proj)
                        self.log(f"    Saved blue (max proj): {nuclei_output}")

            elif image.ndim == 4:
                # 4D image: (Z, C, Y, X) - Z-stack with multiple color channels
                num_slices = image.shape[0]
                num_channels = image.shape[1]
                self.log(f"    Z-stack with color channels: {num_slices} slices, {num_channels} channels")

                # First, apply max projection along Z-axis (axis 0)
                self.log(f"    Applying max projection along Z-axis...")
                max_projected = np.max(image, axis=0)  # Result: (C, Y, X)
                self.log(f"    Max projected shape: {max_projected.shape}")

                # Now identify channels in the max-projected image
                grey_idx, blue_idx = self.identify_channels(image_path, max_projected)

                # Save grey channel if identified
                if grey_idx is not None and not myotube_output.exists():
                    grey_channel = max_projected[grey_idx]
                    myotube_output.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(str(myotube_output), grey_channel)
                    self.log(f"    Saved grey channel {grey_idx}: {myotube_output}")
                elif grey_idx is None:
                    self.log(f"    Skipping grey output (no grey channel found)")

                # Save blue channel if identified
                if blue_idx is not None and not nuclei_output.exists():
                    blue_channel = max_projected[blue_idx]
                    nuclei_output.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(str(nuclei_output), blue_channel)
                    self.log(f"    Saved blue channel {blue_idx}: {nuclei_output}")
                elif blue_idx is None:
                    self.log(f"    Skipping blue output (no blue channel found)")
            else:
                self.log(f"    Warning: Unsupported image dimensions: {image.ndim}")
                self.stats['errors'] += 1
                return False

            self.stats['processed'] += 1
            return True

        except Exception as e:
            self.log(f"    Error processing {image_path.name}: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.stats['errors'] += 1
            return False

    def process_all(self):
        """Process all images in input directory."""
        self.log("=" * 80)
        self.log("CHANNEL SPLITTING AND MAX PROJECTION")
        self.log("=" * 80)
        self.log(f"Input directory: {self.input_dir}")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"  Myotube channel: {self.myotube_dir}")
        self.log(f"  Nuclei channel: {self.nuclei_dir}")
        self.log("=" * 80)

        # Find all images
        self.log("\nSearching for images...")
        image_files = self.find_images()
        self.stats['total_images'] = len(image_files)
        self.log(f"Found {len(image_files)} TIFF images\n")

        if len(image_files) == 0:
            self.log("No images found to process.")
            return

        # Process each image
        for i, image_path in enumerate(image_files, 1):
            self.log(f"[{i}/{len(image_files)}]")
            self.process_image(image_path)

        # Print summary
        self.log("\n" + "=" * 80)
        self.log("PROCESSING SUMMARY")
        self.log("=" * 80)
        self.log(f"Total images found:    {self.stats['total_images']}")
        self.log(f"Successfully processed: {self.stats['processed']}")
        self.log(f"Skipped (exist):       {self.stats['skipped']}")
        self.log(f"Errors:                {self.stats['errors']}")
        self.log("=" * 80)


class MaxProjectionTab(TabInterface):
    """Tab for channel splitting and max projection."""

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
            'input_dir': os.path.join(workflow_base, 'raw_images'),
            'output_dir': os.path.join(workflow_base, '1_max_projection'),
        }

        # Load saved parameters or use defaults
        self.params = self.load_config()

        # GUI widgets (will be created in build_ui)
        self.input_dir_var = None
        self.output_dir_var = None
        self.run_button = None
        self.stop_button = None

    def get_tab_name(self) -> str:
        return "Channel Splitting & Max Projection"

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
        input_dir = self.params['input_dir'].strip()
        output_dir = self.params['output_dir'].strip()

        self.log(f"[VALIDATING] Validating parameters...")
        self.log(f"   Input dir: {input_dir}")
        self.log(f"   Output dir: {output_dir}")

        if not input_dir:
            return False, "Please select an input directory"

        if not output_dir:
            return False, "Please select an output directory"

        if not os.path.exists(input_dir):
            return False, f"Input directory does not exist: {input_dir}"

        self.log(f"[OK] Parameters validated")
        return True, None

    def build_ui(self, parent_frame, console_text):
        """Build the UI for this tab."""
        self.console_text = console_text

        # Create tkinter variables from saved params
        self.input_dir_var = tk.StringVar(value=self.params['input_dir'])
        self.output_dir_var = tk.StringVar(value=self.params['output_dir'])

        # Input/Output Section
        io_frame = ttk.LabelFrame(parent_frame, text="Input/Output", padding=10)
        io_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Input directory
        ttk.Label(io_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        input_entry = ttk.Entry(io_frame, textvariable=self.input_dir_var, width=50)
        input_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=2)

        # Output directory
        ttk.Label(io_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        output_entry = ttk.Entry(io_frame, textvariable=self.output_dir_var, width=50)
        output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=2)

        io_frame.columnconfigure(1, weight=1)

        # Info Section
        info_frame = ttk.LabelFrame(parent_frame, text="Processing Information", padding=10)
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        info_text = """This tool processes multi-channel TIFF images by:

1. Scanning all subdirectories for TIFF files
2. Splitting channels and identifying:
   - Grey channel (myotube/cytoplasm)
   - Blue channel (nuclei)
3. Applying max intensity projection to Z-stacks
4. Saving results to two output folders:
   - myotube_channel/ (grey channel)
   - nuclei_channel/ (blue channel)
5. Preserving the original folder structure

Output files are named: MAX_{original_name}_grey.tif and MAX_{original_name}_blue.tif
"""
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.grid(row=0, column=0, sticky=tk.W)

        # Create buttons in button frame
        self.run_button = ttk.Button(self.button_frame, text="Run Processing", command=self.on_run_threaded)
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.on_stop, state='disabled')

    def get_button_frame_widgets(self):
        """Return list of (button, side) tuples for button frame."""
        return [
            (self.run_button, tk.LEFT),
            (self.stop_button, tk.LEFT)
        ]

    def browse_input_dir(self):
        """Browse for input directory."""
        folder = filedialog.askdirectory(title="Select Input Directory")
        if folder:
            self.input_dir_var.set(folder)

    def browse_output_dir(self):
        """Browse for output directory."""
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_dir_var.set(folder)

    def update_params_from_gui(self):
        """Update parameters from GUI widgets."""
        self.params['input_dir'] = self.input_dir_var.get()
        self.params['output_dir'] = self.output_dir_var.get()

    def update_gui_from_params(self):
        """Update GUI widgets from current parameters."""
        self.input_dir_var.set(self.params['input_dir'])
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

        input_dir = self.params['input_dir']
        output_dir = self.params['output_dir']

        # Update UI
        self.is_running = True
        self.stop_requested = False
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Start processing thread
        thread = threading.Thread(target=self.run_processing, args=(input_dir, output_dir), daemon=True)
        thread.start()

    def run_processing(self, input_dir, output_dir):
        """Run the channel splitting and max projection processing."""
        try:
            self.log(f"\n[STARTING] Starting processing...")
            self.log(f"   Input: {input_dir}")
            self.log(f"   Output: {output_dir}")

            # Create processor
            processor = MaxProjectionProcessor(
                input_dir=input_dir,
                output_dir=output_dir,
                progress_callback=self.log
            )

            # Run processing
            processor.process_all()

            self.log("\n[OK] Processing complete!")

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
