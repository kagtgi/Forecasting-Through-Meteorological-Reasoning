"""ASG-WM: an Atmospheric Scene Graph world model for faithful precipitation nowcasting.

Three readable stages and a faithful information bottleneck:
    Stage A (perception) -> ASG_t  ->  Stage B (transition) -> ASG_{t+h}
    -> [faithful bottleneck: Z = ASG_{t+h} (+) advect_blind(X_t)] -> Stage C (renderer) -> field

See specs/{idea,architecture,datasource,training_method,eval}.md for the design.
"""
__version__ = "0.1.0"

from .asg.schema import ASG, StormObject, ASGSequence, REGIMES, N_MAX  # noqa: F401
