from app.services.model_manager.schedulers import (
    SCHEDULER_DESCRIPTIONS,
    SCHEDULER_NAMES,
    SamplerType,
)

from .schemas import SamplerItem

samplers = [
    SamplerItem(
        value=stype.value,
        name=SCHEDULER_NAMES[stype],
        description=SCHEDULER_DESCRIPTIONS[stype],
    )
    for stype in SamplerType
]
