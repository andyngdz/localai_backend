from app.services.model_manager.schedulers import (
    SCHEDULER_DESCRIPTIONS,
    SCHEDULER_NAMES,
    SamplerType,
)
from app.services.model_manager.schemas import SamplerItem

default_negative_prompt: str = 'blurry, low quality, bad anatomy, deformed, ugly, distorted, noise, grainy, watermark, text, signature, worst quality, low resolution, out of focus, jpeg artifacts, bad composition, bad perspective, unrealistic, render, cartoon, anime, 3d, illustration'

samplers = [
    SamplerItem(
        value=stype.value,
        name=SCHEDULER_NAMES[stype],
        description=SCHEDULER_DESCRIPTIONS[stype],
    )
    for stype in SamplerType
]
