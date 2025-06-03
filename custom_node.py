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
        
        # Extract face keypoints
        if 'people' not in person or not person['people'] or 'face_keypoints_2d' not in person['people'][0]:
            return ("missing keypoints", -1)
            
        face_kps = person['people'][0]['face_keypoints_2d']
        if len(face_kps) < 12:  # Minimum required face keypoints
            return ("missing keypoints", -1)
            
        # Extract key face points (assuming 68-point model)
        if len(face_kps) >= 68 * 3:  # 68-point face model
            # Get left and right eye points
            left_eye_x = sum(face_kps[i*3] for i in range(36, 42)) / 6
            right_eye_x = sum(face_kps[i*3] for i in range(42, 48)) / 6
            # Get nose tip
            nose_tip_x = face_kps[30*3]  # Nose tip is point 30
            # Get mouth corners
            left_mouth_x = face_kps[48*3]
            right_mouth_x = face_kps[54*3]
            # Get jaw points
            left_jaw_x = face_kps[0*3]  # Left jaw point
            right_jaw_x = face_kps[16*3]  # Right jaw point
            # Get eyebrow points
            left_eyebrow_x = sum(face_kps[i*3] for i in range(17, 22)) / 5
            right_eyebrow_x = sum(face_kps[i*3] for i in range(22, 27)) / 5
        else:  # Fallback for smaller face models
            # Use first two pairs as eyes, next two as mouth corners
            left_eye_x = face_kps[0]
            right_eye_x = face_kps[3]
            nose_tip_x = face_kps[6]  # Assuming nose is point 6
            left_mouth_x = face_kps[9]
            right_mouth_x = face_kps[12]
            left_jaw_x = face_kps[0]  # Use eye point as jaw point
            right_jaw_x = face_kps[3]  # Use eye point as jaw point
            left_eyebrow_x = face_kps[0]  # Use eye point as eyebrow point
            right_eyebrow_x = face_kps[3]  # Use eye point as eyebrow point
            
        # Calculate face symmetry metrics
        eye_mid_x = (left_eye_x + right_eye_x) / 2
        mouth_mid_x = (left_mouth_x + right_mouth_x) / 2
        jaw_mid_x = (left_jaw_x + right_jaw_x) / 2
        eyebrow_mid_x = (left_eyebrow_x + right_eyebrow_x) / 2
        
        # Calculate face center using all features
        face_center_x = (eye_mid_x + mouth_mid_x + jaw_mid_x + eyebrow_mid_x) / 4
        
        # Calculate symmetry metrics for each feature
        eye_symmetry = abs(nose_tip_x - eye_mid_x)
        mouth_symmetry = abs(nose_tip_x - mouth_mid_x)
        jaw_symmetry = abs(nose_tip_x - jaw_mid_x)
        eyebrow_symmetry = abs(nose_tip_x - eyebrow_mid_x)
        
        # Calculate feature distances
        eye_distance = abs(left_eye_x - right_eye_x)
        mouth_distance = abs(left_mouth_x - right_mouth_x)
        jaw_distance = abs(left_jaw_x - right_jaw_x)
        eyebrow_distance = abs(left_eyebrow_x - right_eyebrow_x)
        
        # Calculate average feature distance for normalization
        avg_feature_distance = (eye_distance + mouth_distance + jaw_distance + eyebrow_distance) / 4
        
        # Calculate nose offset from face center
        nose_offset = nose_tip_x - face_center_x
        nose_offset_ratio = abs(nose_offset) / avg_feature_distance
        
        # Calculate symmetry scores (0 to 1, where 1 is perfectly symmetric)
        symmetry_scores = [
            eye_symmetry / eye_distance,
            mouth_symmetry / mouth_distance,
            jaw_symmetry / jaw_distance,
            eyebrow_symmetry / eyebrow_distance
        ]
        avg_symmetry = sum(symmetry_scores) / len(symmetry_scores)
        
        # Calculate feature alignment scores
        feature_alignments = [
            abs(eye_mid_x - face_center_x) / eye_distance,
            abs(mouth_mid_x - face_center_x) / mouth_distance,
            abs(jaw_mid_x - face_center_x) / jaw_distance,
            abs(eyebrow_mid_x - face_center_x) / eyebrow_distance
        ]
        avg_alignment = sum(feature_alignments) / len(feature_alignments)
        
        # Thresholds based on example data analysis
        symmetry_threshold = 0.15  # 15% deviation allowed for forward-facing
        alignment_threshold = 0.12  # 12% deviation allowed for feature alignment
        offset_threshold = 0.18    # 18% of average feature distance
        
        # Check if face is symmetric and well-aligned (forward facing)
        if (avg_symmetry < symmetry_threshold and 
            avg_alignment < alignment_threshold and 
            nose_offset_ratio < offset_threshold):
            return ("forward", 0)
            
        # Determine left/right based on nose position relative to face center
        if nose_offset > 0:
            return ("right", 2)
        else:
            return ("left", 1)


NODE_CLASS_MAPPINGS = {
    "OpenPose - Get direction": DirectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectionNode": "OpenPose - Get direction"
}
