from enum import Enum
from typing import Any, Dict

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)


class SamplerType(str, Enum):
    """Enum for supported sampler types."""

    EULER_A = 'EULER_A'
    KDPM2_A = 'KDPM2_A'
    EULER = 'EULER'
    DDIM = 'DDIM'
    LMS = 'LMS'
    PNDM = 'PNDM'
    DPM_SOLVER_MULTISTEP = 'DPM_SOLVER_MULTISTEP'
    DPM_SOLVER_MULTISTEP_KARRAS = 'DPM_SOLVER_MULTISTEP_KARRAS'
    DPM_SOLVER_SINGLES = 'DPM_SOLVER_SINGLES'
    DPM_SOLVER_SDE = 'DPM_SOLVER_SDE'
    DPM_SOLVER_SDE_KARRAS = 'DPM_SOLVER_SDE_KARRAS'
    DDPMS = 'DDPM'
    UNIPC = 'UniPC'
    DEIS = 'DEIS'
    KDPM2 = 'KDPM2'


SCHEDULER_MAPPING: Dict[SamplerType, Any] = {
    SamplerType.EULER_A: EulerAncestralDiscreteScheduler,
    SamplerType.KDPM2_A: KDPM2AncestralDiscreteScheduler,
    SamplerType.EULER: EulerDiscreteScheduler,
    SamplerType.DDIM: DDIMScheduler,
    SamplerType.LMS: LMSDiscreteScheduler,
    SamplerType.PNDM: PNDMScheduler,
    SamplerType.DDPMS: DDPMScheduler,
    SamplerType.UNIPC: UniPCMultistepScheduler,
    SamplerType.DEIS: DEISMultistepScheduler,
    SamplerType.KDPM2: KDPM2DiscreteScheduler,
    SamplerType.DPM_SOLVER_MULTISTEP: DPMSolverMultistepScheduler,
    SamplerType.DPM_SOLVER_MULTISTEP_KARRAS: DPMSolverMultistepScheduler,
    SamplerType.DPM_SOLVER_SINGLES: DPMSolverSinglestepScheduler,
    SamplerType.DPM_SOLVER_SDE: DPMSolverSDEScheduler,
    SamplerType.DPM_SOLVER_SDE_KARRAS: DPMSolverSDEScheduler,
}

SCHEDULER_DESCRIPTIONS: Dict[SamplerType, str] = {
    SamplerType.EULER_A: 'Fast, exploratory, slightly non-deterministic. Good for quick iterations.',
    SamplerType.KDPM2_A: 'Ancestral K-Diffusion sampler, good for exploration with a different feel.',
    SamplerType.EULER: 'Fast, deterministic. Simple and effective.',
    SamplerType.DDIM: 'Deterministic, stable, and widely used.',
    SamplerType.LMS: 'Deterministic, often produces smooth results.',
    SamplerType.PNDM: 'Deterministic, good balance of speed and quality, often default for older models.',
    SamplerType.DDPMS: 'The original DDPM scheduler, very stable but typically requires many steps.',
    SamplerType.UNIPC: 'Unified Pseudo-numerical ODE solver, known for good quality at fewer steps.',
    SamplerType.DEIS: 'Deterministic, known for efficiency and quality.',
    SamplerType.KDPM2: 'Deterministic K-Diffusion sampler.',
    SamplerType.DPM_SOLVER_MULTISTEP: 'High quality, deterministic, good general-purpose sampler (e.g., DPM++ 2M).',
    SamplerType.DPM_SOLVER_MULTISTEP_KARRAS: 'Similar to DPM++ 2M, but with Karras noise schedule for potentially better quality and detail.',
    SamplerType.DPM_SOLVER_SINGLES: 'Single-step deterministic DPM-Solver. Fast but might require more steps for quality.',
    SamplerType.DPM_SOLVER_SDE: 'DPM Solver++ Stochastic Differential Equation (SDE) solver. Can produce very high-quality and detailed results, good for creative variations.',
    SamplerType.DPM_SOLVER_SDE_KARRAS: 'DPM++ SDE with Karras noise schedule. A stochastic sampler for high-quality, detailed results.',
}

SCHEDULER_NAMES: Dict[SamplerType, str] = {
    SamplerType.EULER_A: 'Euler A',
    SamplerType.KDPM2_A: 'KDPM2 A',
    SamplerType.EULER: 'Euler',
    SamplerType.DDIM: 'DDIM',
    SamplerType.LMS: 'LMS',
    SamplerType.PNDM: 'PNDM',
    SamplerType.DDPMS: 'DDPM',
    SamplerType.UNIPC: 'UniPC',
    SamplerType.DEIS: 'DEIS',
    SamplerType.KDPM2: 'KDPM2',
    SamplerType.DPM_SOLVER_MULTISTEP: 'DPM++ 2M',
    SamplerType.DPM_SOLVER_MULTISTEP_KARRAS: 'DPM++ 2M Karras',
    SamplerType.DPM_SOLVER_SINGLES: 'DPM++ 1M',
    SamplerType.DPM_SOLVER_SDE: 'DPM++ SDE',
    SamplerType.DPM_SOLVER_SDE_KARRAS: 'DPM++ SDE Karras',
}
