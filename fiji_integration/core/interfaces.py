"""
Abstract interfaces for segmentation components.

This module defines the interfaces that allow components to interact
without tight coupling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


__all__ = ['SegmentationInterface']


class SegmentationInterface(ABC):
    """
    Abstract interface for myotube segmentation implementations.

    This interface defines the contract that segmentation backends must implement
    to work with tiled inference and other wrappers.
    """

    @abstractmethod
    def initialize_predictor(self, force_cpu: bool = False) -> None:
        """
        Initialize the segmentation model predictor.

        Args:
            force_cpu: If True, force CPU mode even if GPU is available
        """
        pass

    @property
    @abstractmethod
    def predictor(self):
        """
        Return the initialized predictor instance.

        Returns:
            Predictor instance (e.g., Detectron2 DefaultPredictor)
        """
        pass

    @property
    @abstractmethod
    def post_processor(self):
        """
        Return the post-processing pipeline.

        Returns:
            PostProcessingPipeline instance
        """
        pass

    @abstractmethod
    def segment_image(self, image_path: str, output_dir: str,
                     custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Segment a single image and generate outputs.

        Args:
            image_path: Path to input image
            output_dir: Directory for output files
            custom_config: Optional custom configuration parameters

        Returns:
            dict: Processing results and output file paths
        """
        pass

    @abstractmethod
    def _generate_fiji_outputs(self, raw_instances, processed_instances: Dict[str, Any],
                               original_image: np.ndarray, output_dir: str,
                               skip_merged_masks: bool = True) -> Dict[str, str]:
        """
        Generate Fiji-compatible output files.

        Args:
            raw_instances: Raw model predictions
            processed_instances: Post-processed instances
            original_image: Original input image
            output_dir: Directory for output files
            skip_merged_masks: If True, skip merged mask generation

        Returns:
            dict: Paths to generated output files
        """
        pass

    # Optional: Methods for tiled inference to set internal state
    # These are accessed as attributes by TiledMyotubeSegmentation
    # Implementations should support these as properties or attributes:
    # - _original_size: tuple[int, int] or None
    # - _scale_factor: float
    # - _processing_size: tuple[int, int] or None
