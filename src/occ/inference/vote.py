from collections import Counter, deque
from typing import List, Optional

class MajorityVoter:
    """
    Sliding window majority voter for temporal flicker suppression.
    Filters out spurious transient state predictions.
    """
    def __init__(self, window_size: int = 3, default_label: str = "Empty"):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.default_label = default_label

    def add(self, label: str) -> str:
        """Add prediction to buffer and return consensus voted label."""
        self.buffer.append(label)
        counts = Counter(self.buffer)
        most_common = counts.most_common()
        if not most_common:
            return self.default_label
        # In case of tie, prioritize current state or most common
        return most_common[0][0]

    def reset(self):
        self.buffer.clear()
