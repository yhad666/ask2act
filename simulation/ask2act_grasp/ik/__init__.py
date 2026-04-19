__all__ = ["SimpleIK"]


def __getattr__(name: str):
    if name == "SimpleIK":
        from .simple_ik import SimpleIK

        return SimpleIK
    raise AttributeError(name)
