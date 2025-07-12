from app.model_manager import (
    SCHEDULER_DESCRIPTIONS,
    SCHEDULER_NAMES,
    SamplerType,
)

from .schemas import SamplerItem

default_negative_prompt = '(worst quality, low quality, lowres, blurry, jpeg artifacts, watermark, signature, text, logo), (bad hands, bad anatomy, mutated, deformed, disfigured, extra limbs, cropped, out of frame), (cartoon, anime, cgi, render, 3d, doll, toy, painting, sketch)'

samplers = [
    SamplerItem(
        value=stype.value,
        name=SCHEDULER_NAMES[stype],
        description=SCHEDULER_DESCRIPTIONS[stype],
    )
    for stype in SamplerType
]
