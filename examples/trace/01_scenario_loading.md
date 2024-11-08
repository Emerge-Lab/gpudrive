## Code Snippet
```python
import gpudrive

from pygpudrive.env.config import SceneConfig
from pygpudrive.env.scene_selector import select_scenes

scene_config = SceneConfig(path="data/processed/examples", num_scenes=1)

sim = gpudrive.SimManager( # Specify the execution mode: CUDA or CPU
    exec_mode=gpudrive.madrona.ExecMode.CPU, #gpudrive.madrona.ExecMode.CUDA
    gpu_id=0,
    scenes=select_scenes(scene_config),
    params=gpudrive.Parameters(),  # Environment parameters
)
```
## Trace
- The `SceneConfig` class is found in [./pygpudrive/env/config.py](../../pygpudrive/env/config.py).
```python
@dataclass
class SceneConfig:
    """Configuration for selecting scenes from a dataset.

    Attributes:
        path (str): Path to the dataset.
        num_scenes (int): Number of scenes to select.
        discipline (SelectionDiscipline): Method for selecting scenes.
        k_unique_scenes (Optional[int]): Number of unique scenes if using
            K_UNIQUE_N discipline.
        seed (Optional[int]): Seed for random scene selection.
    """

    path: str
    num_scenes: int
    discipline: SelectionDiscipline = SelectionDiscipline.PAD_N
    k_unique_scenes: Optional[int] = None
    seed: Optional[int] = None
```
- The `SimManager` Python class is bound to the `Manager` class in [./src/mgr.hpp](../../src/mgr.hpp) (bindings in [./src/bindings.cpp](../../src/bindings.cpp)).
