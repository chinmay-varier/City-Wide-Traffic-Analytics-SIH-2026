import csv
from datetime import datetime


class DataManager:
    def __init__(self, nodes_file="camera_nodes.csv", logs_file="anpr_logs.csv"):
        self.nodes_file = nodes_file
        self.logs_file = logs_file

    def load_nodes(self):
        """Loads camera node spatial data indexed by camera_id."""
        nodes = {}
        with open(self.nodes_file, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                nodes[row["camera_id"]] = {
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "sector_name": row["sector_name"],
                }
        return nodes

    def load_logs(self):
        """Loads detection logs with parsed timestamps."""
        logs = []
        with open(self.logs_file, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                row["speed_kmh"] = float(row["speed_kmh"])
                row["timestamp"] = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                logs.append(row)
        return logs