from typing import Protocol

from atlasfin.contracts import ParsedObject, RawObject


class Parser(Protocol):
    def parse(self, raw: RawObject) -> ParsedObject: ...
