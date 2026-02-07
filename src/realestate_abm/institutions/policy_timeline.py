"""정책 타임라인 - 시간에 따른 정책 변경 스케줄"""

from ..core.events import EventBus, Event


class PolicyTimeline:
    """시간에 따른 정책 변경"""

    def __init__(self, events_config: list, event_bus: EventBus):
        self.scheduled = sorted(events_config, key=lambda e: e.month)
        self.event_bus = event_bus
        self._next_idx = 0

    def check(self, current_month: int):
        """현재 월에 예정된 정책 변경 발행"""
        while self._next_idx < len(self.scheduled):
            event_cfg = self.scheduled[self._next_idx]
            if event_cfg.month <= current_month:
                self.event_bus.publish(Event(
                    type=f"policy.{event_cfg.type}",
                    month=current_month,
                    data=event_cfg.params,
                ))
                self._next_idx += 1
            else:
                break

    def reset(self):
        self._next_idx = 0
