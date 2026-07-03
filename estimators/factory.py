"""
Estimator Factory
=================

Factory function for creating estimators.
"""

from typing import Dict, Type

from ..config import SimulationConfig
from .base import BaseEstimator
from .psm import PSMEstimator
from .ipw import IPWEstimator
from .ow import OWEstimator
from .aipw import AIPWEstimator
from .fm import FMEstimator
from .eb import EBEstimator


def get_estimator(name: str, config: SimulationConfig) -> BaseEstimator:
    """
    Factory function to create an estimator by name.
    
    Parameters
    ----------
    name : str
        Estimator name ('PSM', 'IPW', 'OW', 'AIPW_correct', 
                 'AIPW_restricted', 'FM', 'EB')
    config : SimulationConfig
        Simulation configuration
    
    Returns
    -------
    BaseEstimator
        Estimator instance
    
    Raises
    ------
    ValueError
        If estimator name is unknown
    """
    estimators: Dict[str, Type[BaseEstimator]] = {
        "PSM": PSMEstimator,
        "IPW": IPWEstimator,
        "OW": OWEstimator,
        "FM": FMEstimator,
        "EB": EBEstimator,
    }
    
    if name == "AIPW_correct":
        return AIPWEstimator(config, method_name="AIPW_correct", use_extended_outcome=True)
    elif name == "AIPW_restricted":
        return AIPWEstimator(config, method_name="AIPW_restricted", use_extended_outcome=False)
    elif name in estimators:
        return estimators[name](config)
    else:
        raise ValueError(f"Unknown estimator: {name}")