from data_manager import DataManager
from trajectory_engine import TrajectoryEngine
from heatmap_generator import GISMapVisualizer


def run_pipeline():
    # Load camera node configurations and telemetry
    data_mgr = DataManager(nodes_file="camera_nodes.csv", logs_file="anpr_logs.csv")
    nodes = data_mgr.load_nodes()
    logs = data_mgr.load_logs()

    # Track specific vehicle across camera nodes
    target_plate = "KA-01-AB-1234"
    engine = TrajectoryEngine(data_mgr)
    trajectory = engine.get_vehicle_trajectory(target_plate)

    print(f"--- Node-to-Node Trajectory for {target_plate} ---")
    for step, item in enumerate(trajectory, start=1):
        print(f"{step}. [{item['timestamp']}] Node: {item['camera_id']} -> Speed: {item['speed_kmh']} km/h")

    # Render interactive map output
    visualizer = GISMapVisualizer(nodes, logs)
    visualizer.generate_map(trajectory=trajectory, target_plate=target_plate, output_filename="trajectory_heatmap.html")


if __name__ == "__main__":
    run_pipeline()