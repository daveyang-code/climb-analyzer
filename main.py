import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Deque, Optional


class Analyzer:
    def __init__(self, hold_model_path: str, video_source="input.mp4"):

        self.hold_model = YOLO(hold_model_path)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.video = cv2.VideoCapture(video_source)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1

        self.max_occlusion_frames = 10000
        self.min_confidence = 0.3
        self.interaction_tolerance = 20
        self.min_contact_frames = 10

        self.min_confirmation_frames = (
            10  # Number of frames a hold needs to be detected with good confidence
        )
        self.hold_confidence_history = {}  # {id: list of confidence values}
        self.confirmed_holds = set()  # Set of hold IDs that are confirmed

        # IOU tracking parameters
        self.iou_threshold = 0.3
        self.next_hold_id = 0
        self.tracked_holds = {}  # {id: hold_info}
        self.hold_absent_count = {}  # {id: frames_since_last_seen}
        self.max_absent_frames = 30  # Maximum frames to keep tracking a missing hold

        self.landmarks = {
            key: {
                "history": deque(maxlen=10),
                "visible": False if key != "center_of_mass" else True,
                "last_seen": {"position": None, "frame": 0},
            }
            for key in [
                "left_hand",
                "right_hand",
                "left_toe",
                "left_heel",
                "right_toe",
                "right_heel",
                "center_of_mass",
            ]
        }

        self.track_history = defaultdict(
            lambda: {
                "positions": deque(maxlen=300),
                "last_seen": 0,
                "size": (0, 0),
                "confidence": 0.0,
                "touched_by": set(),
                "first_touch_frame": None,
                "contact_counters": {
                    "left_hand": 0,
                    "right_hand": 0,
                    "left_foot": 0,
                    "right_foot": 0,
                },
                "confirmed_contacts": set(),
            }
        )

    def calculate_iou(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union between two bounding boxes (x1, y1, x2, y2)"""
        # Calculate intersection coordinates
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # Check if boxes intersect
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        # Calculate area of intersection
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate areas of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate union area
        union_area = box1_area + box2_area - intersection_area

        # Calculate and return IOU
        return intersection_area / union_area if union_area > 0 else 0.0

    def assign_hold_ids(self, current_holds: List[Dict]) -> List[Dict]:
        """Match current frame holds with tracked holds using IOU"""
        # Create a copy of the current holds to add tracking IDs
        tracked_current_holds = current_holds.copy()

        # Track used IDs to prevent multiple assignments
        matched_hold_ids = set()
        matched_current_indices = set()

        # Calculate IOU between each tracked hold and each current hold
        for hold_id, tracked_hold in self.tracked_holds.items():
            # Reset the absent counter to check which holds aren't seen in this frame
            self.hold_absent_count[hold_id] = self.hold_absent_count.get(hold_id, 0) + 1

            best_iou = self.iou_threshold
            best_match_idx = None

            for idx, current_hold in enumerate(current_holds):
                if idx in matched_current_indices:
                    continue  # Skip if this hold already matched

                iou = self.calculate_iou(tracked_hold["bbox"], current_hold["bbox"])

                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx

            # If we found a match
            if best_match_idx is not None:
                tracked_current_holds[best_match_idx]["id"] = hold_id
                self.tracked_holds[hold_id] = tracked_current_holds[best_match_idx]
                self.hold_absent_count[hold_id] = 0  # Reset absence counter
                matched_hold_ids.add(hold_id)
                matched_current_indices.add(best_match_idx)

        # Assign new IDs to unmatched holds
        for idx, hold in enumerate(tracked_current_holds):
            if idx not in matched_current_indices:
                new_id = self.next_hold_id
                self.next_hold_id += 1
                hold["id"] = new_id
                self.tracked_holds[new_id] = hold
                self.hold_absent_count[new_id] = 0

                # Initialize confidence history for new holds
                self.hold_confidence_history[new_id] = []

        # Update confidence history for all current holds
        for hold in tracked_current_holds:
            hold_id = hold["id"]

            # Add current confidence to history
            if hold_id not in self.hold_confidence_history:
                self.hold_confidence_history[hold_id] = []

            self.hold_confidence_history[hold_id].append(hold["confidence"])

            # Keep only recent history (e.g., last 10 frames)
            if len(self.hold_confidence_history[hold_id]) > 10:
                self.hold_confidence_history[hold_id].pop(0)

            # Check if hold meets confirmation criteria
            if len(
                self.hold_confidence_history[hold_id]
            ) >= self.min_confirmation_frames and all(
                conf > self.min_confidence
                for conf in self.hold_confidence_history[hold_id]
            ):
                self.confirmed_holds.add(hold_id)
                hold["confirmed"] = True
            else:
                hold["confirmed"] = hold_id in self.confirmed_holds

        # Remove holds that haven't been seen for too long
        hold_ids_to_remove = []
        for hold_id, absent_count in self.hold_absent_count.items():
            if absent_count > self.max_absent_frames:
                hold_ids_to_remove.append(hold_id)

        for hold_id in hold_ids_to_remove:
            if hold_id in self.tracked_holds:
                del self.tracked_holds[hold_id]
            if hold_id in self.hold_absent_count:
                del self.hold_absent_count[hold_id]
            if hold_id in self.hold_confidence_history:
                del self.hold_confidence_history[hold_id]
            if hold_id in self.confirmed_holds:
                self.confirmed_holds.remove(hold_id)

        return tracked_current_holds

    def detect_holds(self, frame: np.ndarray) -> List[Dict]:
        """Detect climbing holds using the YOLO model"""
        results = self.hold_model(frame)
        holds = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                if confidence > self.min_confidence:
                    hold_info = {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        "size": (int(x2 - x1), int(y2 - y1)),
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                    holds.append(hold_info)

        # Apply IOU tracking to maintain consistent hold IDs across frames
        tracked_holds = self.assign_hold_ids(holds)
        return tracked_holds

    def calculate_distance(
        self, point1: Tuple[int, int], point2: Tuple[int, int]
    ) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def find_nearest_hold(
        self,
        body_part_position: Tuple[int, int],
        holds: List[Dict],
        max_distance: int = 50,
    ) -> Optional[Dict]:
        """Find the nearest hold to a body part if within max_distance"""
        if not body_part_position or not holds:
            return None

        nearest_hold = None
        min_distance = float("inf")

        for hold in holds:
            distance = self.calculate_distance(body_part_position, hold["center"])
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                nearest_hold = hold

        return nearest_hold

    def snap_to_hold(
        self, body_part_position: Tuple[int, int], hold: Dict
    ) -> Tuple[int, int]:
        """Return the adjusted position that's snapped to the hold center"""
        return hold["center"]

    def calculate_weighted_com(
        self, landmarks, image_width: int, image_height: int
    ) -> Tuple[float, float]:

        masses = {
            "head": 0.07,
            "torso": 0.50,
            "arm_left": 0.03,
            "arm_right": 0.03,
            "leg_left": 0.16,
            "leg_right": 0.16,
        }

        com_x, com_y = 0.0, 0.0
        total_mass = 0.0

        for i, landmark in enumerate(landmarks.landmark):

            x_px = landmark.x * image_width
            y_px = landmark.y * image_height

            if i <= 10:
                mass = masses["head"] / 11
            elif i in [11, 12, 23, 24]:
                mass = masses["torso"] / 4
            elif 13 <= i <= 21:
                mass = masses["arm_left"] / 5 if i % 2 else masses["arm_right"] / 5
            else:
                mass = masses["leg_left"] / 4 if i % 2 else masses["leg_right"] / 4

            com_x += mass * x_px
            com_y += mass * y_px
            total_mass += mass

        return com_x / total_mass, com_y / total_mass

    def smooth_position(self, part_name, new_position):
        """Apply smoothing to landmark positions"""
        history = self.landmarks[part_name]["history"]

        if not history:
            for _ in range(history.maxlen):
                history.append(new_position)
            return new_position

        history.append(new_position)

        weights = list(range(1, len(history) + 1))
        total_weight = sum(weights)

        smoothed_x = sum(x * w for (x, y), w in zip(history, weights))
        smoothed_y = sum(y * w for (x, y), w in zip(history, weights))

        return (int(smoothed_x / total_weight), int(smoothed_y / total_weight))

    def detect_body_landmarks(self, frame: np.ndarray, frame_count: int) -> Dict:
        def is_visible(landmark_enum):
            return landmarks[landmark_enum].visibility > min_visibility

        def get_average_position(points):
            avg_x = int(sum(p.x for p in points) / len(points) * self.width)
            avg_y = int(sum(p.y for p in points) / len(points) * self.height)
            return avg_x, avg_y

        def get_position(landmark_enum):
            lm = landmarks[landmark_enum]
            return int(lm.x * self.width), int(lm.y * self.height)

        def update_landmark(name, raw_position):
            smoothed = self.smooth_position(name, raw_position)
            landmarks_detected[name] = smoothed
            self.landmarks[name]["last_seen"]["position"] = smoothed
            self.landmarks[name]["last_seen"]["frame"] = frame_count

        def occluded_position(name):
            last = self.landmarks[name]["last_seen"]["position"]
            if last is not None:
                landmarks_detected[name] = last

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        landmarks_detected = {}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            com_x, com_y = self.calculate_weighted_com(
                results.pose_landmarks, self.width, self.height
            )

            min_visibility = 0.5

            # Hands
            for side in ["left", "right"]:
                wrist = getattr(self.mp_pose.PoseLandmark, f"{side.upper()}_WRIST")
                thumb = getattr(self.mp_pose.PoseLandmark, f"{side.upper()}_THUMB")
                index = getattr(self.mp_pose.PoseLandmark, f"{side.upper()}_INDEX")
                pinky = getattr(self.mp_pose.PoseLandmark, f"{side.upper()}_PINKY")
                name = f"{side}_hand"
                visible = is_visible(wrist)
                self.landmarks[name]["visible"] = visible

                if visible:
                    raw = get_average_position(
                        [
                            landmarks[wrist],
                            landmarks[thumb],
                            landmarks[index],
                            landmarks[pinky],
                        ]
                    )
                    update_landmark(name, raw)
                else:
                    occluded_position(name)

            # Heels and toes
            for side in ["left", "right"]:
                heel_enum = getattr(self.mp_pose.PoseLandmark, f"{side.upper()}_HEEL")
                toe_enum = getattr(
                    self.mp_pose.PoseLandmark, f"{side.upper()}_FOOT_INDEX"
                )
                heel_name = f"{side}_heel"
                toe_name = f"{side}_toe"

                heel_visible = is_visible(heel_enum)
                toe_visible = is_visible(toe_enum)
                self.landmarks[heel_name]["visible"] = heel_visible
                self.landmarks[toe_name]["visible"] = toe_visible

                if heel_visible:
                    update_landmark(heel_name, get_position(heel_enum))
                else:
                    occluded_position(heel_name)

                if toe_visible:
                    update_landmark(toe_name, get_position(toe_enum))
                else:
                    occluded_position(toe_name)

            # Center of mass
            update_landmark("center_of_mass", (int(com_x), int(com_y)))

            return landmarks_detected, results

        # No pose detected: fallback to last known positions
        result_dict = {
            part: self.landmarks[part]["last_seen"]["position"]
            for part in self.landmarks
            if self.landmarks[part]["last_seen"]["position"] is not None
        }

        return (result_dict, None) if result_dict else ({}, None)

    def update_track_history(self, hold_id, hold_info, frame_count, body_part=None):
        """Update the tracking history for a hold"""
        track = self.track_history[hold_id]
        track["positions"].append(hold_info["center"])
        track["last_seen"] = frame_count
        track["size"] = hold_info["size"]
        track["confidence"] = hold_info["confidence"]

        # If a body part is interacting with this hold
        if body_part:
            if body_part not in track["touched_by"]:
                track["touched_by"].add(body_part)
                track["first_touch_frame"] = frame_count

            # Increment contact counter for specific body part
            if "hand" in body_part:
                counter_key = body_part
            else:  # toe or heel
                side = body_part.split("_")[0]
                counter_key = f"{side}_foot"

            track["contact_counters"][counter_key] += 1

            # Check if we've reached the threshold for confirming contact
            if track["contact_counters"][counter_key] >= self.min_contact_frames:
                track["confirmed_contacts"].add(counter_key)

    def analyze_stability(self, body_parts: Dict) -> Dict:
        """Analyze climber's stability based on body position"""
        stability_info = {}

        # Check if we have center of mass and at least one point of contact
        if "center_of_mass" not in body_parts:
            return {"status": "unknown", "message": "Center of mass not detected"}

        com = body_parts["center_of_mass"]
        supports = []

        # Collect all potential support points (hands and feet)
        hand_foot_parts = ["left_hand", "right_hand", "left_toe", "right_toe"]
        for part in hand_foot_parts:
            if part in body_parts:
                supports.append(body_parts[part])

        if len(supports) < 2:
            return {
                "status": "unstable",
                "message": "Insufficient contact points detected",
            }

        # Calculate the support polygon
        if len(supports) >= 3:
            # With 3+ points, can use ConvexHull to get the support polygon
            try:
                hull = cv2.convexHull(np.array(supports))
                # Check if COM is inside the support polygon
                com_inside = cv2.pointPolygonTest(hull, com, False) >= 0

                if com_inside:
                    stability_info["status"] = "stable"
                    stability_info["message"] = "Center of mass within support polygon"
                else:
                    # Calculate the distance to the support polygon
                    distance = cv2.pointPolygonTest(hull, com, True)
                    stability_info["status"] = "unstable"
                    stability_info["message"] = (
                        f"Center of mass outside support polygon by {abs(distance):.1f}px"
                    )
                    stability_info["distance"] = abs(distance)

            except Exception:
                # Fall back to simple checks if ConvexHull fails
                stability_info["status"] = "unknown"
                stability_info["message"] = (
                    "Could not calculate stability (hull creation failed)"
                )
        else:
            # With only 2 points, check if COM is close to the line between them
            p1, p2 = supports
            # Calculate perpendicular distance from COM to line
            line_length = self.calculate_distance(p1, p2)
            if line_length > 0:
                # Normalized perpendicular distance
                cross_product = abs(
                    (p2[1] - p1[1]) * com[0]
                    - (p2[0] - p1[0]) * com[1]
                    + p2[0] * p1[1]
                    - p2[1] * p1[0]
                )
                distance = cross_product / line_length

                if distance < 30:  # Define a reasonable threshold
                    stability_info["status"] = "stable"
                    stability_info["message"] = (
                        f"Center of mass close to support line ({distance:.1f}px)"
                    )
                else:
                    stability_info["status"] = "unstable"
                    stability_info["message"] = (
                        f"Center of mass far from support line ({distance:.1f}px)"
                    )

                stability_info["distance"] = distance
            else:
                stability_info["status"] = "unknown"
                stability_info["message"] = "Support points are too close together"

        return stability_info

    def draw_stability(
        self, frame: np.ndarray, body_parts: Dict, stability_info: Dict
    ) -> np.ndarray:
        """Draw stability visualization on the frame"""
        annotated_frame = frame.copy()

        if "center_of_mass" not in body_parts:
            return annotated_frame

        com = body_parts["center_of_mass"]
        supports = []

        # Collect all potential support points
        hand_foot_parts = ["left_hand", "right_hand", "left_toe", "right_toe"]
        for part in hand_foot_parts:
            if part in body_parts:
                supports.append(body_parts[part])

        # Draw support polygon if we have enough points
        if len(supports) >= 3:
            try:
                # Convert to numpy array for convex hull
                points = np.array(supports)
                hull = cv2.convexHull(points)

                # Draw the support polygon
                status = stability_info.get("status", "unknown")
                if status == "stable":
                    color = (0, 255, 0)  # Green
                elif status == "unstable":
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 255, 0)  # Yellow for unknown

                cv2.polylines(annotated_frame, [hull], True, color, 2)

                # Fill with semi-transparent color
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [hull], color)  # Removed the alpha channel
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

            except Exception as e:
                # If hull creation fails, fall back to connecting the dots
                for i in range(len(supports)):
                    cv2.line(
                        annotated_frame,
                        supports[i],
                        supports[(i + 1) % len(supports)],
                        (255, 255, 0),
                        1,
                    )
        elif len(supports) == 2:
            # Draw line between the two support points
            cv2.line(annotated_frame, supports[0], supports[1], (255, 255, 0), 2)

        # Display stability status
        status = stability_info.get("status", "unknown")
        message = stability_info.get("message", "Stability unknown")

        # Choose color based on status
        if status == "stable":
            status_color = (0, 255, 0)  # Green
        elif status == "unstable":
            status_color = (0, 0, 255)  # Red
        else:
            status_color = (255, 255, 0)  # Yellow

        # Draw status text at the top of the frame
        cv2.putText(
            annotated_frame,
            f"Stability: {status} - {message}",
            (10, 30),
            self.font,
            self.font_scale,
            status_color,
            self.font_thickness,
        )

        return annotated_frame

    def process_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        # Get the existing body landmarks detection
        body_landmarks, pose_results = self.detect_body_landmarks(frame, frame_count)

        # Detect holds in the current frame
        holds = self.detect_holds(frame)

        # Create a list that includes both currently visible holds and recently occluded ones
        visible_hold_ids = {hold.get("id") for hold in holds}
        all_holds = holds.copy()

        # Add recently occluded holds (but only if they're confirmed)
        for hold_id, absent_count in self.hold_absent_count.items():
            if (
                hold_id not in visible_hold_ids
                and absent_count <= self.max_absent_frames
                and hold_id in self.confirmed_holds
            ):
                # Get the hold from tracked_holds
                occluded_hold = self.tracked_holds.get(hold_id)
                if occluded_hold:
                    # Mark as occluded
                    occluded_hold = (
                        occluded_hold.copy()
                    )  # Create a copy to avoid modifying the original
                    occluded_hold["occluded"] = True
                    all_holds.append(occluded_hold)

        # Map to store snapped positions
        snapped_positions = {}

        # Associate body parts with holds
        body_parts_to_snap = ["left_hand", "right_hand", "left_toe", "right_toe"]
        snap_connections = {}

        for part in body_parts_to_snap:
            if part in body_landmarks and body_landmarks[part] is not None:
                part_position = body_landmarks[part]

                # For hands, we want a larger snap distance
                max_snap_distance = 70 if "hand" in part else 40

                nearest_hold = self.find_nearest_hold(
                    part_position, all_holds, max_snap_distance
                )

                if nearest_hold:
                    # Store the original and snapped positions
                    snapped_positions[part] = self.snap_to_hold(
                        part_position, nearest_hold
                    )
                    snap_connections[part] = {
                        "original": part_position,
                        "snapped": snapped_positions[part],
                        "hold": nearest_hold,
                    }

                    # Update tracking history with this interaction
                    self.update_track_history(
                        nearest_hold["id"], nearest_hold, frame_count, part
                    )

        # Create the annotated frame using the original method
        annotated_frame = frame.copy()

        # if pose_results:
        #     self.mp_drawing.draw_landmarks(
        #         annotated_frame,
        #         pose_results.pose_landmarks,
        #         self.mp_pose.POSE_CONNECTIONS,
        #         landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        #     )

        # Draw the detected holds with their tracking IDs
        for hold in all_holds:
            x1, y1, x2, y2 = hold["bbox"]
            hold_id = hold.get("id", -1)
            is_occluded = hold.get("occluded", False)
            is_confirmed = hold_id in self.confirmed_holds

            # Use consistent colors for hold IDs
            color_hash = hold_id % 6
            colors = [
                (0, 255, 255),
                (255, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 0, 255),
                (128, 128, 0),
            ]
            hold_color = colors[color_hash]

            # Adjust appearance for occluded holds
            line_thickness = 1 if is_occluded else 2
            if is_occluded:
                # Draw dashed line for occluded holds
                for i in range(x1, x2, 5):
                    cv2.line(
                        annotated_frame,
                        (i, y1),
                        (min(i + 3, x2), y1),
                        hold_color,
                        line_thickness,
                    )
                    cv2.line(
                        annotated_frame,
                        (i, y2),
                        (min(i + 3, x2), y2),
                        hold_color,
                        line_thickness,
                    )
                for i in range(y1, y2, 5):
                    cv2.line(
                        annotated_frame,
                        (x1, i),
                        (x1, min(i + 3, y2)),
                        hold_color,
                        line_thickness,
                    )
                    cv2.line(
                        annotated_frame,
                        (x2, i),
                        (x2, min(i + 3, y2)),
                        hold_color,
                        line_thickness,
                    )
            else:
                cv2.rectangle(
                    annotated_frame, (x1, y1), (x2, y2), hold_color, line_thickness
                )

            # Add hold ID and confidence with indication of confirmation status
            if is_confirmed:
                cv2.putText(
                    annotated_frame,
                    f"ID:{hold_id} ({hold['confidence']:.2f})",
                    (x1, y1 - 5),
                    self.font,
                    self.font_scale,
                    hold_color,
                    self.font_thickness,
                )

            # If this hold has confirmed contacts, show them
            if hold_id in self.track_history:
                track = self.track_history[hold_id]
                if track["confirmed_contacts"]:
                    contact_text = ", ".join(track["confirmed_contacts"])
                    cv2.putText(
                        annotated_frame,
                        f"Used by: {contact_text}",
                        (x1, y2 + 15),
                        self.font,
                        self.font_scale,
                        hold_color,
                        self.font_thickness,
                    )

        # Draw body landmarks
        for landmark, position in body_landmarks.items():
            if position is None:
                continue

            if landmark == "center_of_mass":
                cv2.circle(annotated_frame, position, 5, (255, 255, 255), -1)
                cv2.circle(annotated_frame, position, 5, (0, 0, 255), 2)
                cv2.putText(
                    annotated_frame,
                    "CoM",
                    (position[0] + 10, position[1]),
                    self.font,
                    self.font_scale,
                    (0, 0, 255),
                    self.font_thickness,
                )
            else:
                is_visible = (
                    landmark in self.landmarks and self.landmarks[landmark]["visible"]
                )
                color = (0, 255, 0) if is_visible else (0, 0, 255)

                cv2.circle(annotated_frame, position, 5, color, -1)

                label = landmark if is_visible else f"{landmark} (occluded)"
                cv2.putText(
                    annotated_frame,
                    label,
                    (position[0] + 10, position[1]),
                    self.font,
                    self.font_scale,
                    color,
                    self.font_thickness,
                )

        # Draw snapping connections
        for part, connection in snap_connections.items():
            original = connection["original"]
            snapped = connection["snapped"]
            hold = connection["hold"]

            # Draw a line connecting the original position to the snapped position
            cv2.line(annotated_frame, original, snapped, (255, 0, 255), 2)

            # Draw a larger circle at the snapped position
            cv2.circle(annotated_frame, snapped, 8, (255, 0, 255), -1)

            # Add a label indicating this is a snapped position with hold ID
            cv2.putText(
                annotated_frame,
                f"{part} → Hold #{hold.get('id', -1)}",
                (snapped[0] + 10, snapped[1]),
                self.font,
                self.font_scale,
                (255, 0, 255),
                self.font_thickness,
            )

        # Analyze stability based on the detected body parts
        stability_info = self.analyze_stability(body_landmarks)

        # Apply stability visualization to the frame
        annotated_frame = self.draw_stability(
            annotated_frame, body_landmarks, stability_info
        )

        return annotated_frame

    def analyze_video(self):
        frame_count = 0

        # For saving the output video
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_video = cv2.VideoWriter(
            "output_tracked.avi", fourcc, self.fps, (self.width, self.height)
        )

        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame, frame_count)

            # Write to output video
            output_video.write(processed_frame)

            cv2.imshow("Climbing Analysis", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

        self.video.release()
        output_video.release()
        cv2.destroyAllWindows()

        # Return statistics about the tracking
        return {
            "total_frames": frame_count,
            "unique_holds": self.next_hold_id,
            "hold_contacts": {
                hold_id: data["confirmed_contacts"]
                for hold_id, data in self.track_history.items()
                if data["confirmed_contacts"]
            },
        }


if __name__ == "__main__":
    analyzer = Analyzer(
        hold_model_path="./models/holds.pt",
        video_source="./videos/1.mp4",
    )

    results = analyzer.analyze_video()
