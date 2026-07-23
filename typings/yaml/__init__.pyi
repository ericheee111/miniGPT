from collections.abc import Mapping

class YAMLError(Exception): ...

def safe_dump(
    data: Mapping[str, object],
    *,
    sort_keys: bool = ...,
    allow_unicode: bool = ...,
) -> str: ...
def safe_load(stream: str) -> object: ...
