from typing import Protocol

from atlasfin.contracts import Chunk, ParsedObject


class Chunker(Protocol):
    def chunk(self, parsed: ParsedObject) -> list[Chunk]: ...
