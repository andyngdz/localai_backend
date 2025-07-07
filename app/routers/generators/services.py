from app.services.model_manager.schedulers import (
    SCHEDULER_DESCRIPTIONS,
    SCHEDULER_NAMES,
    SamplerType,
)
from app.services.model_manager.schemas import AvailableSampler


def get_available_samplers() -> list[AvailableSampler]:
    """Returns list of supported samplers."""
    return [
        AvailableSampler(
            value=stype.value,
            name=SCHEDULER_NAMES[stype],
            description=SCHEDULER_DESCRIPTIONS[stype],
        )
        for stype in SamplerType
    ]
