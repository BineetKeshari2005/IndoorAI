"""Natural language navigation instruction generator compiling human-readable directives (Task 28)."""

from __future__ import annotations

from typing import List, Optional

from src.core.logging import setup_logger
from src.navigation.models import InstructionType, NavigationInstruction, NavigationWaypoint
from src.navigation.turn_detection import TurnDetector

logger = setup_logger("InstructionGenerator")


class InstructionGenerator:
    """Generates ordered, human-readable turn-by-turn guidance directives from route waypoints."""

    def __init__(self, detector: Optional[TurnDetector] = None):
        self.detector = detector or TurnDetector()

    def generate_instructions(
        self,
        waypoints: List[NavigationWaypoint],
        destination_name: str = "Destination",
    ) -> List[NavigationInstruction]:
        """Compiles human-readable turn-by-turn navigation instructions."""
        if not waypoints:
            return []

        if len(waypoints) == 1:
            wp = waypoints[0]
            return [
                NavigationInstruction(
                    instruction_id="INST-000",
                    sequence_number=1,
                    instruction_type=InstructionType.ARRIVE_DESTINATION,
                    text=f"You are already at your destination: {destination_name}.",
                    waypoint=wp,
                    floor_id=wp.floor_id,
                    target_heading_deg=None,
                    distance_to_next_m=0.0,
                    cumulative_distance_m=0.0,
                )
            ]

        # 1. Run turn detector
        transitions = self.detector.detect_transitions(waypoints)
        instructions: List[NavigationInstruction] = []
        cumulative_dist = 0.0

        for seq_idx, (wp_idx, inst_type, bearing_deg, dist_next) in enumerate(transitions):
            curr_wp = waypoints[wp_idx]
            next_wp = waypoints[wp_idx + 1] if wp_idx + 1 < len(waypoints) else None

            # Generate natural language text
            text = self._format_instruction_text(
                inst_type=inst_type,
                curr_wp=curr_wp,
                next_wp=next_wp,
                dist_m=dist_next,
                destination_name=destination_name,
            )

            inst = NavigationInstruction(
                instruction_id=f"INST-{seq_idx:03d}",
                sequence_number=seq_idx + 1,
                instruction_type=inst_type,
                text=text,
                waypoint=curr_wp,
                floor_id=curr_wp.floor_id,
                target_heading_deg=bearing_deg,
                distance_to_next_m=round(dist_next, 2),
                cumulative_distance_m=round(cumulative_dist, 2),
            )
            instructions.append(inst)
            cumulative_dist += dist_next

        return instructions

    @staticmethod
    def _format_instruction_text(
        inst_type: InstructionType,
        curr_wp: NavigationWaypoint,
        next_wp: Optional[NavigationWaypoint],
        dist_m: float,
        destination_name: str,
    ) -> str:
        next_label = next_wp.label if next_wp else "path"

        if inst_type == InstructionType.START:
            return f"Start walking along {curr_wp.label} towards {next_label} ({dist_m:.1f} m)."

        elif inst_type == InstructionType.CONTINUE_STRAIGHT:
            return f"Continue straight along {curr_wp.label} for {dist_m:.1f} m."

        elif inst_type == InstructionType.TURN_LEFT:
            return f"Turn left onto {next_label} and proceed for {dist_m:.1f} m."

        elif inst_type == InstructionType.TURN_RIGHT:
            return f"Turn right onto {next_label} and proceed for {dist_m:.1f} m."

        elif inst_type == InstructionType.SLIGHT_LEFT:
            return f"Bear slightly left onto {next_label} ({dist_m:.1f} m)."

        elif inst_type == InstructionType.SLIGHT_RIGHT:
            return f"Bear slightly right onto {next_label} ({dist_m:.1f} m)."

        elif inst_type == InstructionType.SHARP_LEFT:
            return f"Make a sharp left onto {next_label} ({dist_m:.1f} m)."

        elif inst_type == InstructionType.SHARP_RIGHT:
            return f"Make a sharp right onto {next_label} ({dist_m:.1f} m)."

        elif inst_type == InstructionType.UTURN:
            return f"Make a U-turn and continue along {next_label} ({dist_m:.1f} m)."

        elif inst_type == InstructionType.PASS_DOORWAY:
            return f"Pass through {curr_wp.label} into {next_label} ({dist_m:.1f} m)."

        elif inst_type == InstructionType.ENTER_ROOM:
            return f"Enter {next_label}."

        elif inst_type == InstructionType.TAKE_STAIRS:
            target_floor = next_wp.floor_id if next_wp else curr_wp.floor_id
            return f"Take the stairs ({curr_wp.label}) to Floor {target_floor}."

        elif inst_type == InstructionType.USE_ELEVATOR:
            target_floor = next_wp.floor_id if next_wp else curr_wp.floor_id
            return f"Take the elevator ({curr_wp.label}) to Floor {target_floor}."

        elif inst_type == InstructionType.USE_RAMP:
            return f"Use the accessibility ramp at {curr_wp.label}."

        elif inst_type == InstructionType.ARRIVE_DESTINATION:
            return f"You have arrived at your destination: {destination_name}."

        return f"Proceed towards {next_label} ({dist_m:.1f} m)."
