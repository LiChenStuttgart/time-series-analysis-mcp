"""Optional foundation-model forecasting providers."""

from .automl import forecast_automl
from .chronos2 import forecast_chronos2
from .toto2 import forecast_toto2

__all__ = ["forecast_automl", "forecast_chronos2", "forecast_toto2"]
