import time

class Timer :
    def __init__(self):
        self.start_time = {}
        self.durations = {}
        self.total_start = time.time()

    def start(self, label: str) :
        self.start_time[label] = time.time()

    def end(self, label: str) :
        if label not in self.start_time:
            raise ValueError(f"Timer for '{label}' was not started.")
        elapsed = time.time() - self.start_time[label]
        self.durations[label] = elapsed

    def total(self):
        return time.time() - self.total_start
    
    def summary(self):
        print("\n[TIMER SUMMARY]")
        for label, duration in self.durations.items():
            print(f"{label}: {duration:.2f}s")

if __name__ == "__main__":
    timer = Timer()    