from .schedulers import SCHEDULER_DESCRIPTIONS, SCHEDULER_NAMES, SamplerType
from .schemas import SamplerItem


class ConstantService:
    """
    A service class to manage constants used across the application.
    """

    @property
    def samplers(self):
        """
        Returns a list of all available sampler types.
        """
        return [
            SamplerItem(
                value=stype.value,
                name=SCHEDULER_NAMES[stype],
                description=SCHEDULER_DESCRIPTIONS[stype],
            )
            for stype in SamplerType
        ]


constant_service = ConstantService()
