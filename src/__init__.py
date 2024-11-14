"""
Student Adaptability Prediction Module
"""

from .train_model import train_model
from .predict import predict_student_adaptability
from .utils import prepare_data, load_data

__version__ = '0.1.0'
__all__ = ['train_model', 'predict_student_adaptability', 'prepare_data', 'load_data']