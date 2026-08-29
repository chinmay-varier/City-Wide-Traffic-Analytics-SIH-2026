import folium
from folium.plugins import HeatMap


class GISMapVisualizer:
    def __init__(self, nodes, logs):
        self.nodes = nodes
        self.logs = logs

    def generate_map(self, trajectory=None, target_plate=None, output_filename="city_map.html"):
        """Generates map with traffic density heatmaps and vehicle polylines."""
        # Center map using average coordinates of camera nodes
        avg_lat = sum(node["latitude"] for node in self.nodes.values()) / len(self.nodes)
        avg_lng = sum(node["longitude"] for node in self.nodes.values()) / len(self.nodes)
        
        city_map = folium.Map(location=[avg_lat, avg_lng], zoom_start=13, tiles="OpenStreetMap")

        # 1. Macro Density Heatmap Layer
        heat_data = [
            [self.nodes[log["camera_id"]]["latitude"], self.nodes[log["camera_id"]]["longitude"], 1.0]
            for log in self.logs if log["camera_id"] in self.nodes
        ]
        HeatMap(heat_data, name="Traffic Density", radius=25, blur=15).add_to(city_map)

        # 2. Node-to-Node Trajectory Layer
        if trajectory:
            route_coords = []
            for step, point in enumerate(trajectory):
                coords = (point["latitude"], point["longitude"])
                route_coords.append(coords)

                color = "green" if step == 0 else ("red" if step == len(trajectory) - 1 else "blue")
                popup_text = (
                    f"<b>Step {step + 1}</b><br>"
                    f"<b>Node:</b> {point['camera_id']} ({point['sector_name']})<br>"
                    f"<b>Time:</b> {point['timestamp']}<br>"
                    f"<b>Speed:</b> {point['speed_kmh']} km/h"
                )

                folium.Marker(
                    location=coords,
                    popup=folium.Popup(popup_text, max_width=250),
                    tooltip=f"Node {point['camera_id']}",
                    icon=folium.Icon(color=color, icon="camera"),
                ).add_to(city_map)

            # Draw trajectory path line across camera nodes
            folium.PolyLine(
                locations=route_coords,
                color="red",
                weight=4,
                opacity=0.8,
                tooltip=f"Trajectory for {target_plate}"
            ).add_to(city_map)

        folium.LayerControl().add_to(city_map)
        city_map.save(output_filename)
        print(f"Map generated successfully: {output_filename}")