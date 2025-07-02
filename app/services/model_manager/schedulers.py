# python_backend/schemas.py (Updated SamplerType)

from enum import Enum
from typing import Any, Dict, Type  # Ensure all types are imported

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,  # Correct class for DPM++ SDE
    DPMSolverSinglestepScheduler,  # Correct class for DPM++ 1M / DPM++ 1S
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

    # Ancestral (stochastic) samplers - good for exploration
    EULER_A = 'Euler A'
    KDPM2_A = 'KDPM2 A'  # K-Diffusion PM2 Ancestral
    # Deterministic samplers - good for reproducibility
    EULER = 'Euler'
    DDIM = 'DDIM'
    LMS = 'LMS'
    PNDM = 'PNDM'
    DPM_SOLVER_MULTISTEP = 'DPM++ 2M'  # DPM Solver++ (2M)
    DPM_SOLVER_MULTISTEP_KARRAS = (
        'DPM++ 2M Karras'  # DPM Solver++ (2M) with Karras sigmas
    )
    DPM_SOLVER_SINGLES = 'DPM++ 1M'  # DPM Solver++ Single-step (deterministic)
    DPM_SOLVER_SDE = 'DPM++ SDE'  # Added for the non-Karras SDE variant
    DPM_SOLVER_SDE_KARRAS = 'DPM++ SDE Karras'  # DPM Solver++ SDE with Karras sigmas
    DDPMS = 'DDPM'  # Denoising Diffusion Probabilistic Models Scheduler
    UNIPC = 'UniPC'  # Unified Pseudo-numerical ODE Solver
    DEIS = 'DEIS'  # Denoising Diffusion Implicit Models Scheduler
    KDPM2 = 'KDPM2'  # K-Diffusion PM2 Discrete (deterministic)


# --- Sampler Mapping ---
SCHEDULER_MAPPING: Dict[SamplerType, Type[Any]] = {
    # Ancestral (stochastic) samplers
    SamplerType.EULER_A: EulerAncestralDiscreteScheduler,
    SamplerType.KDPM2_A: KDPM2AncestralDiscreteScheduler,
    # Deterministic samplers
    SamplerType.EULER: EulerDiscreteScheduler,
    SamplerType.DDIM: DDIMScheduler,
    SamplerType.LMS: LMSDiscreteScheduler,
    SamplerType.PNDM: PNDMScheduler,
    SamplerType.DDPMS: DDPMScheduler,
    SamplerType.UNIPC: UniPCMultistepScheduler,
    SamplerType.DEIS: DEISMultistepScheduler,
    SamplerType.KDPM2: KDPM2DiscreteScheduler,
    # DPM-Solver variations
    SamplerType.DPM_SOLVER_MULTISTEP: DPMSolverMultistepScheduler,
    SamplerType.DPM_SOLVER_MULTISTEP_KARRAS: DPMSolverMultistepScheduler,
    SamplerType.DPM_SOLVER_SINGLES: DPMSolverSinglestepScheduler,
    SamplerType.DPM_SOLVER_SDE: DPMSolverSDEScheduler,
    SamplerType.DPM_SOLVER_SDE_KARRAS: DPMSolverSDEScheduler,
}

# Sampler Descriptions
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
