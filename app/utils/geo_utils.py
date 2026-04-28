from __future__ import annotations

import folium
from branca.element import MacroElement, Template

from utils.domain import DUBAI_CENTER, ZONE_NAMES, ZONES, get_location_context as domain_get_location_context, get_nearest_zone as domain_get_nearest_zone


class _DraggableRouteBridge(MacroElement):
    _template = Template(
        """
        {% macro script(this, kwargs) %}
        const map = {{ this._parent.get_name() }};
        const pickupMarker = {{ this.pickup_marker_name }};
        const dropoffMarker = {{ this.dropoff_marker_name }};

        pickupMarker.dragging.enable();
        dropoffMarker.dragging.enable();

        function boundsToDict(bounds) {
            return {
                "_southWest": {
                    "lat": bounds.getSouthWest().lat,
                    "lng": bounds.getSouthWest().lng,
                },
                "_northEast": {
                    "lat": bounds.getNorthEast().lat,
                    "lng": bounds.getNorthEast().lng,
                },
            };
        }

        function pointFeature(marker, role, label, color) {
            const latLng = marker.getLatLng();
            return {
                type: "Feature",
                properties: {
                    role: role,
                    label: label,
                    color: color,
                },
                geometry: {
                    type: "Point",
                    coordinates: [latLng.lng, latLng.lat],
                },
            };
        }

        function publish(role) {
            const activeMarker = role === "pickup" ? pickupMarker : dropoffMarker;
            const activeLatLng = activeMarker.getLatLng();
            const payload = {
                route_editor_event: {
                    role: role,
                    lat: activeLatLng.lat,
                    lng: activeLatLng.lng,
                    signature: role + ":" + activeLatLng.lat.toFixed(6) + ":" + activeLatLng.lng.toFixed(6),
                },
                last_clicked: null,
                last_object_clicked: activeLatLng,
                last_object_clicked_tooltip: role,
                last_object_clicked_popup: null,
                all_drawings: [
                    pointFeature(pickupMarker, "pickup", "Pickup", "#22c55e"),
                    pointFeature(dropoffMarker, "dropoff", "Dropoff", "#ef4444"),
                ],
                last_active_drawing: pointFeature(
                    activeMarker,
                    role,
                    role === "pickup" ? "Pickup" : "Dropoff",
                    role === "pickup" ? "#22c55e" : "#ef4444"
                ),
                bounds: boundsToDict(map.getBounds()),
                zoom: map.getZoom(),
                center: map.getCenter(),
                last_circle_radius: null,
                last_circle_polygon: null,
                selected_layers: window.__GLOBAL_DATA__ && window.__GLOBAL_DATA__.selected_layers
                    ? Object.values(window.__GLOBAL_DATA__.selected_layers)
                    : [],
                selected_tags: window.__GLOBAL_DATA__ && window.__GLOBAL_DATA__.selected_tags
                    ? window.__GLOBAL_DATA__.selected_tags
                    : [],
            };

            window.__GLOBAL_DATA__ = window.__GLOBAL_DATA__ || {};
            window.__GLOBAL_DATA__.last_object_clicked = payload.last_object_clicked;
            window.__GLOBAL_DATA__.last_object_clicked_tooltip = payload.last_object_clicked_tooltip;
            window.__GLOBAL_DATA__.all_drawings = payload.all_drawings;
            window.__GLOBAL_DATA__.last_active_drawing = payload.last_active_drawing;
            window.__GLOBAL_DATA__.zoom = payload.zoom;
            window.__GLOBAL_DATA__.center = payload.center;
            window.__GLOBAL_DATA__.previous_data = payload;

            if (window.Streamlit && window.Streamlit.setComponentValue) {
                window.Streamlit.setComponentValue(payload);
            }
        }

        pickupMarker.on("dragend", function() { publish("pickup"); });
        dropoffMarker.on("dragend", function() { publish("dropoff"); });
        {% endmacro %}
        """
    )

    def __init__(self, pickup_marker_name: str, dropoff_marker_name: str):
        super().__init__()
        self._name = "DraggableRouteBridge"
        self.pickup_marker_name = pickup_marker_name
        self.dropoff_marker_name = dropoff_marker_name


def _build_route_pin(label: str, pill_background: str, pill_border: str, dot_fill: str) -> folium.DivIcon:
    return folium.DivIcon(
        html=f"""
        <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:18px;height:18px;border-radius:999px;background:{dot_fill};border:3px solid #ffffff;box-shadow:0 8px 18px rgba(15,23,42,0.18);"></div>
            <div style="padding:6px 12px;border-radius:999px;background:{pill_background};border:1px solid {pill_border};color:#0f172a;font-size:12px;font-weight:800;letter-spacing:0.02em;white-space:nowrap;box-shadow:0 10px 22px rgba(15,23,42,0.12);cursor:grab;">
                {label}
            </div>
        </div>
        """,
        icon_size=(150, 38),
        icon_anchor=(18, 18),
    )


def get_nearest_zone(lat: float, lon: float) -> str:
    return domain_get_nearest_zone(lat, lon)


def _map_location_label(lat: float, lon: float) -> str:
    location = domain_get_location_context(lat, lon)
    if location["neighborhood"] == location["zone"]:
        return location["zone"]
    return f"{location['neighborhood']} / {location['zone']}"


def build_picker_map(
    pickup_point: tuple[float, float] | None,
    dropoff_point: tuple[float, float] | None,
    route_geometry: list[tuple[float, float]] | None = None,
    center: tuple[float, float] | None = None,
    zoom: int | None = None,
):
    map_center = center or pickup_point or dropoff_point or DUBAI_CENTER
    map_object = folium.Map(location=map_center, zoom_start=zoom or 11, tiles="CartoDB positron", control_scale=True)

    for zone_name in ZONE_NAMES:
        zone = ZONES[zone_name]
        folium.CircleMarker(
            location=(zone["lat"], zone["lon"]),
            radius=4,
            color="#134e4a",
            weight=1,
            fill=True,
            fill_opacity=0.65,
            fill_color="#0d9488",
            tooltip=zone_name,
        ).add_to(map_object)

    if route_geometry:
        folium.PolyLine(route_geometry, color="#f59e0b", weight=5, opacity=0.85).add_to(map_object)

    pickup_marker = None
    if pickup_point:
        pickup_marker = folium.Marker(
            location=pickup_point,
            draggable=True,
            icon=_build_route_pin("Pickup", "#ecfdf5", "#86efac", "#22c55e"),
            tooltip=f"Pickup | {_map_location_label(*pickup_point)}",
            z_index_offset=1000,
        )
        pickup_marker.add_to(map_object)

    dropoff_marker = None
    if dropoff_point:
        dropoff_marker = folium.Marker(
            location=dropoff_point,
            draggable=True,
            icon=_build_route_pin("Dropoff", "#fef2f2", "#fca5a5", "#ef4444"),
            tooltip=f"Dropoff | {_map_location_label(*dropoff_point)}",
            z_index_offset=1000,
        )
        dropoff_marker.add_to(map_object)

    if pickup_marker and dropoff_marker:
        _DraggableRouteBridge(pickup_marker.get_name(), dropoff_marker.get_name()).add_to(map_object)

    if center is None and zoom is None:
        if route_geometry:
            map_object.fit_bounds(route_geometry)
        elif pickup_point and dropoff_point:
            map_object.fit_bounds([pickup_point, dropoff_point])

    return map_object