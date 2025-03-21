import os
import csv

class DataWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scalar_logs = {}
        self.current_step = {}

        # Create the log directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)

        # Check if the CSV file exists and load data if available
        csv_file_path = os.path.join(self.log_dir, 'scalars.csv')
        if os.path.exists(csv_file_path):
            self._load_scalar_logs(csv_file_path)

    def add_scalar(self, tag, value):
        if tag not in self.scalar_logs:
            self.scalar_logs[tag] = {}
            if tag in self.current_step:
                self.current_step[tag] = max(self.current_step[tag], 0)
            else:
                self.current_step[tag] = 0

        self.scalar_logs[tag][self.current_step[tag]] = value
        self.current_step[tag] += 1

    def save(self):
        # Save the scalar logs to a CSV file
        self._save_scalar_logs()

    def close(self):
        # Perform any cleanup or finalization tasks here
        pass

    def _load_scalar_logs(self, csv_file_path):
        with open(csv_file_path, 'r', newline='') as file:
            reader = csv.reader(file, delimiter = ';')
            header = next(reader)  # Read the header row
            tags = header[1:]  # Extract tags from the header

            # Initialize scalar_logs dictionary with tags
            for tag in tags:
                self.scalar_logs[tag] = {}

            # Load data into scalar_logs dictionary
            for row in reader:
                step = int(row[0])
                for i, tag in enumerate(tags):
                    if row[i + 1]:
                        self.scalar_logs[tag][step] = float(row[i + 1])  # Convert to float
                    self.current_step[tag] = step + 1

    def _save_scalar_logs(self):
        with open(os.path.join(self.log_dir, 'scalars.csv'), 'w', newline='') as file:
            writer = csv.writer(file, delimiter = ';')
            writer.writerow(['step'] + list(self.scalar_logs.keys()))

            max_steps = max(self.current_step.values())
            for step in range(max_steps):
                writer.writerow([step] + [self.scalar_logs[tag].get(step, '') for tag in self.scalar_logs])