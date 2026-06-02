from collections.abc import Iterable
from typing import Any, override

from modosaic.core.modalities import Modalities
from modosaic.core.modality import Modality
from modosaic.core.modality_generator import ModalityGenerator
from modosaic.core.postprocessor import ModalityPostprocessor
from modosaic.core.validator import ModalityValidator
from modosaic.core.validator_step import ValidatorStep


class ConfiguredModality[T](Modality[T]):
    """Concrete `Modality` implementation backed by a runtime enum value.

    Use this class when a modality can be described through composition instead
    of a custom subclass.
    """

    _modality: Modalities

    def __init__(
            self,
            modality: Modalities,
            model: ModalityGenerator[T],
            postprocessor: ModalityPostprocessor[T],
            validators: Iterable[
                            ModalityValidator[T, Any] | ValidatorStep[T, Any]
                            ] | None = None,
    ) -> None:
        """Create a configured modality.

        Args:
            modality: Modality enum value represented by this object.
            model: Generator used to produce the modality output.
            postprocessor: Postprocessor used to create artifacts. Its modality
                must match `modality`.
            validators: Optional validators or validator steps.

        Raises:
            ValueError: If the postprocessor writes a different modality than
                the configured modality.
        """
        if modality != postprocessor.modality:
            raise ValueError(
                f"Postprocessor for {postprocessor.modality} cannot save {modality}"
            )

        self._modality = modality
        super().__init__(
            model=model,
            postprocessor=postprocessor,
            validators=validators,
        )

    @property
    @override
    def modality(self) -> Modalities:
        """Return the configured modality enum value."""
        return self._modality
