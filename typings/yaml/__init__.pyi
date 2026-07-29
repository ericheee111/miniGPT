from collections.abc import Callable, Mapping

class Node: ...
class MappingNode(Node):
    value: list[tuple[Node, Node]]

class SafeLoader:
    @classmethod
    def add_constructor(cls, tag: str, constructor: Callable[[SafeLoader, Node], object]) -> None: ...
    def construct_object(self, node: Node, deep: bool = ...) -> object: ...

class YAMLError(Exception): ...

def safe_dump(
    data: Mapping[str, object],
    *,
    sort_keys: bool = ...,
    allow_unicode: bool = ...,
) -> str: ...
def safe_load(stream: str) -> object: ...
def load(stream: str, Loader: type[SafeLoader]) -> object: ...

class resolver:
    class BaseResolver:
        DEFAULT_MAPPING_TAG: str
