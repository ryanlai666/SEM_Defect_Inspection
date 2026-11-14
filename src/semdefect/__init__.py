"""SEM defect inspection benchmark (Carinthia-S) + CUDA/TensorRT inference engine."""

__version__ = "0.1.0"

CLASS_NAMES = {
    1: "class_1",
    2: "class_2",
    3: "class_3",
    4: "class_4",
    5: "class_5",
    6: "class_6_no_visible_defect",
}
"""Carinthia-S label ids.  The dataset publishes numeric ids only (no physical defect names);
class 6 is documented as 'no visible defect' (SEM mis-alignment, adjacent region imaged)."""
