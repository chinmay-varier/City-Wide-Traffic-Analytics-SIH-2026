class TrajectoryEngine:
    def __init__(self, data_manager):
        self.nodes = data_manager.load_nodes()
        self.logs = data_manager.load_logs()

    def get_vehicle_trajectory(self, target_plate):
        """Reconstructs node-to-node chronological trajectory for a plate number."""
        trajectory = []
        
        for log in self.logs:
            if log["plate_number"].upper() == target_plate.upper():
                cam_id = log["camera_id"]
                node_info = self.nodes.get(cam_id, {})
                
                trajectory.append({
                    "timestamp": log["timestamp"],
                    "camera_id": cam_id,
                    "latitude": node_info.get("latitude"),
                    "longitude": node_info.get("longitude"),
                    "sector_name": node_info.get("sector_name", "Unknown"),
                    "speed_kmh": log["speed_kmh"],
                })

        # Sort chronologically node-to-node
        trajectory.sort(key=lambda x: x["timestamp"])
        return trajectory