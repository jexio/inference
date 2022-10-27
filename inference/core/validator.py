class Validator:
    """Validator for dataclass fields."""

    def __post_init__(self) -> None:
        """Run validation methods if declared.

        The validation method can be a simple check
        that raises ValueError or a transformation to the field value.
        The validation is performed by calling a function named:
            `validate_<field_name>(self, value) -> None`
        Finally, calls (if defined) `validate(self)` for validations that depend on other fields
        """
        for name in self.__dataclass_fields__.keys():
            validator_name = f"validate_{name}"
            if method := getattr(self, validator_name, None):
                method(getattr(self, name))

        if (validate := getattr(self, "validate", None)) and callable(validate):
            validate()


__all__ = ("Validator",)
