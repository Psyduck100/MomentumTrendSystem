from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

from Config import EngineConfig
from Data_model import compute_returns, validate_prices
from Selector import Selector
from UniverseProvider import UniverseProvider
from Rules import EntryRule, ExitRule
from Defensive import DefensiveAsset
