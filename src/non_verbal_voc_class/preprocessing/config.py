import json

from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class PreprocessorConfig:
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, 'r') as file:
            data = json.load(file)
        return cls(config=data)

    def __getattr__(self, item):
        return self.config.get(item)