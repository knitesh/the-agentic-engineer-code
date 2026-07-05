## agent/sinks.py
class ConsoleMetricsSink:
    def record(self, **metrics):
        print(f"[metrics] {metrics}")

class ConsoleTraceSink:
    def write(self, trace):
        print(f"[trace] {len(trace) if hasattr(trace, '__len__') else 1} events")

class ConsoleErrorSink:
    def log(self, msg):
        print(f"[error] {msg}")

class ConsoleStreamSink:
    def push(self, session_id, event):   # used by streaming in 3.19
        print(f"[stream:{session_id}] {event}")

class Sinks:
    def __init__(self):
        self.metrics = ConsoleMetricsSink()
        self.trace = ConsoleTraceSink()
        self.errors = ConsoleErrorSink()
        self.stream = ConsoleStreamSink()
