import time

class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        self.elapsed = None
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._t0
        return False  # don't suppress exceptions