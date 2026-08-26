"""Structure resolution helpers."""

from aiida import orm

from .contracts import validate_source_specification


def resolve_structure(value: orm.StructureData | orm.Dict) -> orm.StructureData:
    """Resolve an explicit structure; recognized external sources are future extensions."""
    if isinstance(value, orm.StructureData):
        return value
    specification = validate_source_specification(value)
    if specification["source"] == "materials_project":
        raise NotImplementedError("the materials_project structure source is not implemented")
    raise AssertionError("unreachable")
