class DirectionNode:
    def __init__(self):
        self.model_path = None  # Initialize to None
        self.model = None


    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force recalculation on every change
        return float("NaN")
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_kps": ("POSE_KEYPOINT",),
            }
        }
        
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("direction", "direction_code")
    FUNCTION = "main"

    CATEGORY = "OpenPose"
    
    def main(self, pose_kps):
        # Get the first person's keypoints (assuming there's at least one person)
        if not pose_kps or len(pose_kps) == 0:
            return ("missing keypoints", -1)
            
        person = pose_kps[0]
        # print(person)
        # Extract pose keypoints from the actual structure
        if 'people' not in person or not person['people'] or 'pose_keypoints_2d' not in person['people'][0]:
            return ("missing keypoints", -1)
        pose = person['people'][0]['pose_keypoints_2d']
        if len(pose) < 21:
            return ("missing keypoints", -1)
        # COCO indices
        nose = (pose[0], pose[1], pose[2])
        left_shoulder = (pose[15], pose[16], pose[17])
        right_shoulder = (pose[18], pose[19], pose[20])
        
        # Check if we have enough keypoints to make a determination
        if not all([right_shoulder, left_shoulder, nose]):
            return ("missing keypoints", -1)
            
        # Calculate the center point between shoulders
        shoulder_center_x = (right_shoulder[0] + left_shoulder[0]) / 2
        nose_offset_x = nose[0] - shoulder_center_x
        shoulder_distance = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_y_diff = abs(right_shoulder[1] - left_shoulder[1])
        # Robust thresholds
        forward_x_threshold = 0.3 * shoulder_distance
        forward_y_margin = 0.4 * abs(right_shoulder[1] - left_shoulder[1] + 1)  # avoid div by zero
        # Shoulders are roughly level if y-diff is less than 20% of shoulder width
        shoulders_level = shoulder_y_diff < 0.2 * shoulder_distance
        # Nose y is between shoulders (with margin)
        nose_between_shoulders = (min(left_shoulder[1], right_shoulder[1]) - forward_y_margin <= nose[1] <= max(left_shoulder[1], right_shoulder[1]) + forward_y_margin)
        if abs(nose_offset_x) <= forward_x_threshold and shoulders_level and nose_between_shoulders:
            return ("forward", 0)
        
        # Try to use face keypoints for more robust forward detection
        face_kps = person['people'][0].get('face_keypoints_2d', None)
        face_forward = False
        if face_kps and len(face_kps) >= 68 * 3:  # 68-point face model
            # Indices for left/right eye and mouth corners in 68-point model
            # left_eye: 36-41, right_eye: 42-47, left_mouth: 48, right_mouth: 54
            left_eye_x = sum(face_kps[i*3] for i in range(36, 42)) / 6
            right_eye_x = sum(face_kps[i*3] for i in range(42, 48)) / 6
            left_mouth_x = face_kps[48*3]
            right_mouth_x = face_kps[54*3]
            # Midpoints
            eye_mid_x = (left_eye_x + right_eye_x) / 2
            mouth_mid_x = (left_mouth_x + right_mouth_x) / 2
            # Symmetry checks
            nose_x = nose[0]
            eye_sym = abs(nose_x - eye_mid_x)
            mouth_sym = abs(nose_x - mouth_mid_x)
            eye_dist = abs(left_eye_x - right_eye_x)
            mouth_dist = abs(left_mouth_x - right_mouth_x)
            # If nose is close to midpoints and eyes/mouth are not too far apart, likely forward
            if eye_sym < 0.15 * eye_dist and mouth_sym < 0.15 * mouth_dist:
                face_forward = True
        elif face_kps and len(face_kps) >= 12:  # fallback for smaller face models
            # Use first two pairs as eyes, next two as mouth corners
            left_eye_x = face_kps[0]
            right_eye_x = face_kps[3]
            left_mouth_x = face_kps[6]
            right_mouth_x = face_kps[9]
            eye_mid_x = (left_eye_x + right_eye_x) / 2
            mouth_mid_x = (left_mouth_x + right_mouth_x) / 2
            nose_x = nose[0]
            eye_sym = abs(nose_x - eye_mid_x)
            mouth_sym = abs(nose_x - mouth_mid_x)
            eye_dist = abs(left_eye_x - right_eye_x)
            mouth_dist = abs(left_mouth_x - right_mouth_x)
            if eye_sym < 0.15 * eye_dist and mouth_sym < 0.15 * mouth_dist:
                face_forward = True
        # If face symmetry says forward, return forward
        if face_forward:
            return ("forward", 0)
        
        # Calculate the angle between shoulders
        shoulder_angle = abs(right_shoulder[0] - left_shoulder[0])
        if shoulder_angle < 20:
            return ("angle too small", -1)
        right_visibility = abs(right_shoulder[0] - nose[0])
        left_visibility = abs(left_shoulder[0] - nose[0])
        if nose_offset_x > 0 and right_visibility > left_visibility:
            return ("right", 2)
        elif nose_offset_x < 0 and left_visibility > right_visibility:
            return ("left", 1)
        else:
            if right_shoulder[0] > left_shoulder[0]:
                return ("right", 2)
            else:
                return ("left", 1)


NODE_CLASS_MAPPINGS = {
    "OpenPose - Get direction": DirectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectionNode": "OpenPose - Get direction"
}
