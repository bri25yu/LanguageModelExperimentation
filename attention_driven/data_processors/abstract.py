from typing import Dict, List, Union

from abc import ABC, abstractmethod

import pandas as pd


class AbstractDataProcessor(ABC):
    @abstractmethod
    def __call__(self) -> Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]]:
        pass
