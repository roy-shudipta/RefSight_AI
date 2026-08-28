from dataclasses import dataclass


@dataclass(frozen=True)
class Columns:
    match_id: str = "MergeID"
    frame: str = "frameIdx"
    time: str = "matchSeconds"
    event_time: str = "Time"

    event_flag: str = "event"
    incorrect: str = "Incorrect_Decision"
    referee_name: str = "Referee Name"

    # Event names and broad event types are mapped.
    event_name: str = "EventName"
    event_type: str = "Type"
