class _MemoryInfo:
    rss: int

class Process:
    def memory_info(self) -> _MemoryInfo: ...

def cpu_count(*, logical: bool = ...) -> int | None: ...
