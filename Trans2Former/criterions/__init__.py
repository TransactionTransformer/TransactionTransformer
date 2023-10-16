import os
from pathlib import Path
import importlib

# automatically import any Python files in the criterions/ directory
for file in sorted(Path(__file__).parent.glob('*.py')):
    if not file.name.startswith("_"):
        # TODO lifan: automatic detection the main folder name later
        versionname = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
        importlib.import_module("{}.criterions.".format(versionname) + file.name[:-3])