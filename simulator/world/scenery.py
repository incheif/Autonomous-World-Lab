"""
Procedural 3D City Scenery Generator for Autonomous-World-Lab.
Generates realistic roadside buildings, sidewalks/footpaths, curbs, trees, and streetlights
along straight and curved road alignments.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from simulator.world.road import RoadMap


BUILDING_PALETTE = [
    {"color": "#1e293b", "accent": "#38bdf8", "style": "office"},       # Slate glass office
    {"color": "#334155", "accent": "#f59e0b", "style": "commercial"},   # Dark commercial
    {"color": "#475569", "accent": "#22c55e", "style": "residential"},  # Mid-rise apartment
    {"color": "#374151", "accent": "#a855f7", "style": "modern"},       # Modern mixed-use
    {"color": "#1f2937", "accent": "#3b82f6", "style": "tower"},        # High-tech tower
    {"color": "#52525b", "accent": "#f43f5e", "style": "loft"},         # Urban loft
]

TREE_PALETTE = [
    {"foliage": "#166534", "trunk": "#78350f"},  # Deep Forest
    {"foliage": "#15803d", "trunk": "#854d0e"},  # Lush Green
    {"foliage": "#10b981", "trunk": "#713f12"},  # Emerald Pine
    {"foliage": "#047857", "trunk": "#5c2d0c"},  # Dark Spruce
]


@dataclass
class Building:
    id: str
    x: float
    y: float
    width: float           # Along road tangent [m]
    depth: float           # Perpendicular to road [m]
    height: float          # Vertical height [m]
    yaw: float             # Oriented with road tangent
    color: str
    accent_color: str
    stories: int
    has_rooftop_hvac: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "width": round(self.width, 2),
            "depth": round(self.depth, 2),
            "height": round(self.height, 2),
            "yaw": round(self.yaw, 3),
            "color": self.color,
            "accent": self.accent_color,
            "stories": self.stories,
            "hvac": self.has_rooftop_hvac,
        }


@dataclass
class Tree:
    id: str
    x: float
    y: float
    trunk_height: float
    canopy_radius: float
    canopy_height: float
    foliage_color: str
    trunk_color: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "trunk_h": round(self.trunk_height, 2),
            "radius": round(self.canopy_radius, 2),
            "canopy_h": round(self.canopy_height, 2),
            "foliage": self.foliage_color,
            "trunk": self.trunk_color,
        }


@dataclass
class Streetlight:
    id: str
    x: float
    y: float
    yaw: float
    height: float = 6.5
    arm_length: float = 2.2
    light_color: str = "#fef08a"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "yaw": round(self.yaw, 3),
            "height": self.height,
            "arm": self.arm_length,
            "color": self.light_color,
        }


class CitySceneryGenerator:
    """
    Generates procedural roadside urban architecture, greenery, and infrastructure
    aligned with road segments.
    """

    def __init__(
        self,
        sidewalk_width_m: float = 2.8,
        curb_height_m: float = 0.18,
        building_setback_m: float = 2.0,
    ):
        self.sidewalk_width: float = sidewalk_width_m
        self.curb_height: float = curb_height_m
        self.building_setback: float = building_setback_m

    def generate(self, road: "RoadMap") -> Dict[str, Any]:
        """
        Populates buildings, trees, and streetlights along the road geometry.
        """
        if not road.segments or len(road.segments) < 4:
            return {"buildings": [], "trees": [], "streetlights": [], "sidewalks": {}}

        half_road = road.total_width_m / 2.0
        n_segs = len(road.segments)
        road_len = road.road_length_m

        buildings: List[Building] = []
        trees: List[Tree] = []
        streetlights: List[Streetlight] = []

        # 1. Procedural Buildings along left (+1) and right (-1) road borders
        rng = np.random.RandomState(42)  # Deterministic seed for reproducible cityscape

        for side in [1.0, -1.0]:
            side_str = "left" if side > 0 else "right"
            s = 8.0  # Start building after road entrance

            b_idx = 0
            while s < road_len - 12.0:
                width = float(rng.uniform(12.0, 22.0))
                depth = float(rng.uniform(14.0, 24.0))
                stories = int(rng.choice([3, 4, 6, 8, 12, 16], p=[0.2, 0.3, 0.25, 0.15, 0.07, 0.03]))
                height = float(stories * 3.2 + rng.uniform(0.5, 2.0))

                s_center = s + width / 2.0
                if s_center >= road_len - 5.0:
                    break

                # Interpolate road segment at s_center
                seg_idx = min(int(round(s_center)), n_segs - 1)
                seg = road.segments[seg_idx]

                # Position building center setback from sidewalk
                dist_from_center = half_road + self.sidewalk_width + self.building_setback + depth / 2.0
                nx = -math.sin(seg.yaw)
                ny = math.cos(seg.yaw)

                bx = seg.center_x + (side * dist_from_center) * nx
                by = seg.center_y + (side * dist_from_center) * ny

                palette = BUILDING_PALETTE[rng.randint(0, len(BUILDING_PALETTE))]

                buildings.append(
                    Building(
                        id=f"bld_{side_str}_{b_idx+1}",
                        x=bx,
                        y=by,
                        width=width,
                        depth=depth,
                        height=height,
                        yaw=seg.yaw,
                        color=palette["color"],
                        accent_color=palette["accent"],
                        stories=stories,
                        has_rooftop_hvac=bool(stories >= 4),
                    )
                )

                b_idx += 1
                # Small gap between adjacent buildings (alleys/pedestrian walkways)
                s += width + float(rng.uniform(3.5, 7.0))

        # 2. Roadside Trees along sidewalk planting verge
        tree_idx = 0
        tree_spacing = 10.0
        num_tree_slots = int(road_len / tree_spacing)

        for slot in range(1, num_tree_slots):
            s_tree = slot * tree_spacing
            seg_idx = min(int(round(s_tree)), n_segs - 1)
            seg = road.segments[seg_idx]

            nx = -math.sin(seg.yaw)
            ny = math.cos(seg.yaw)

            for side in [1.0, -1.0]:
                side_str = "L" if side > 0 else "R"
                # Plant in outer half of sidewalk
                dist_tree = half_road + 1.2
                tx = seg.center_x + (side * dist_tree) * nx
                ty = seg.center_y + (side * dist_tree) * ny

                tree_pal = TREE_PALETTE[(slot + (1 if side > 0 else 0)) % len(TREE_PALETTE)]
                trunk_h = 2.2 + float(rng.uniform(0.0, 0.6))
                c_rad = 1.6 + float(rng.uniform(0.0, 0.7))
                c_h = 2.8 + float(rng.uniform(0.0, 0.8))

                trees.append(
                    Tree(
                        id=f"tree_{side_str}_{tree_idx+1}",
                        x=tx,
                        y=ty,
                        trunk_height=trunk_h,
                        canopy_radius=c_rad,
                        canopy_height=c_h,
                        foliage_color=tree_pal["foliage"],
                        trunk_color=tree_pal["trunk"],
                    )
                )
                tree_idx += 1

        # 3. Streetlights along curbs
        light_idx = 0
        light_spacing = 22.0
        num_lights = int(road_len / light_spacing)

        for slot in range(1, num_lights):
            s_light = slot * light_spacing
            seg_idx = min(int(round(s_light)), n_segs - 1)
            seg = road.segments[seg_idx]

            nx = -math.sin(seg.yaw)
            ny = math.cos(seg.yaw)

            # Alternate left and right curbs
            side = 1.0 if (slot % 2 == 1) else -1.0
            dist_light = half_road + 0.35
            lx = seg.center_x + (side * dist_light) * nx
            ly = seg.center_y + (side * dist_light) * ny

            # Light arm points in toward road center
            arm_yaw = seg.yaw + (math.pi / 2.0 if side < 0 else -math.pi / 2.0)

            streetlights.append(
                Streetlight(
                    id=f"light_{light_idx+1}",
                    x=lx,
                    y=ly,
                    yaw=arm_yaw,
                    height=6.5,
                    arm_length=2.2,
                )
            )
            light_idx += 1

        # 4. Sidewalk boundary polylines for Three.js geometry extrusions
        left_sidewalk_outer = []
        right_sidewalk_outer = []
        for seg in road.segments:
            nx = -math.sin(seg.yaw)
            ny = math.cos(seg.yaw)
            total_dist = half_road + self.sidewalk_width
            left_sidewalk_outer.append([round(seg.center_x + total_dist * nx, 3), round(seg.center_y + total_dist * ny, 3)])
            right_sidewalk_outer.append([round(seg.center_x - total_dist * nx, 3), round(seg.center_y - total_dist * ny, 3)])

        return {
            "buildings": [b.to_dict() for b in buildings],
            "trees": [t.to_dict() for t in trees],
            "streetlights": [sl.to_dict() for sl in streetlights],
            "sidewalk_width_m": self.sidewalk_width,
            "curb_height_m": self.curb_height,
            "left_outer": left_sidewalk_outer,
            "right_outer": right_sidewalk_outer,
        }
