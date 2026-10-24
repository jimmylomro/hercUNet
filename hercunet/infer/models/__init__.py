"""Built-in inference models. Importing this package registers them with :class:`hercunet.infer.model.InferenceModel`."""
from .nnunet_jacobi import NnunetJacobiModel  # noqa: F401 (registers "nnunet-jacobi")

__all__ = ["NnunetJacobiModel"]
