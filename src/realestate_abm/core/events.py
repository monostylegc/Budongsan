"""이벤트 버스 - 생애이벤트, 정책변경 등 이벤트 전파"""

from dataclasses import dataclass, field
from typing import Callable, Any
from collections import defaultdict


@dataclass
class Event:
    """기본 이벤트"""
    type: str
    month: int
    data: dict = field(default_factory=dict)


class EventBus:
    """간단한 이벤트 버스"""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._pending: list[Event] = []

    def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        self._handlers[event_type].append(handler)

    def publish(self, event: Event):
        self._pending.append(event)

    def process(self):
        """대기중인 이벤트 처리"""
        events = self._pending.copy()
        self._pending.clear()
        for event in events:
            for handler in self._handlers.get(event.type, []):
                handler(event)
            # 와일드카드 핸들러
            for handler in self._handlers.get("*", []):
                handler(event)

    def clear(self):
        self._pending.clear()
        self._handlers.clear()
