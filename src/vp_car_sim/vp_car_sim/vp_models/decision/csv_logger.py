"""CSV logger for per-frame compute_command() output (bin number, velocities, brake state)."""
import csv
import os


class CommandLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        out_dir = os.path.dirname(csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        is_new = not os.path.exists(csv_path)
        self.file = open(csv_path, 'a', newline='')
        self.writer = csv.writer(self.file)
        self._flush_counter = 0
        if is_new:
            self.writer.writerow(['bin_number', 'linear_velocity', 'angular_velocity', 'command'])
            self.file.flush()

    def log(self, bin_number, linear_velocity, angular_velocity, blocked):
        command = 'BRAKE' if blocked else 'CLEAR'
        self.writer.writerow([bin_number, linear_velocity, angular_velocity, command])
        self._flush_counter += 1
        if self._flush_counter >= 10:
            self.file.flush()
            self._flush_counter = 0

    def close(self):
        self.file.flush()
        self.file.close()
