import cv2
import numpy as np
import logging
from tqdm import tqdm
from src.llm_judge import judge_answer

logger = logging.getLogger(__name__)

# Define color bounds in HSV
COLOR_BOUNDS = {
    'red': ([0, 120, 70], [10, 255, 255]), # Lower red
    'red2': ([170, 120, 70], [180, 255, 255]), # Upper red
    'green': ([35, 100, 100], [85, 255, 255]),
    'blue': ([100, 150, 0], [140, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'gray': ([0, 0, 40], [180, 50, 220]) # Includes black
}

def detect_objects(image_path):
    """
    Detects colored shapes (circles, squares) in an image using OpenCV.
    
    Returns:
        list: A list of dictionaries, e.g.,
              [{'color': 'red', 'shape': 'square', 'coords': (x, y)}]
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return []
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_objects = []

        for color, (lower, upper) in COLOR_BOUNDS.items():
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Handle red wrapping
            if color == 'red2':
                red_mask_1 = cv2.inRange(hsv, np.array(COLOR_BOUNDS['red'][0]), np.array(COLOR_BOUNDS['red'][1]))
                mask = cv2.bitwise_or(mask, red_mask_1)
                color = 'red' # Consolidate as 'red'
            elif color == 'red':
                continue # Handled by 'red2'
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 100: # Ignore small noise
                    continue
                
                # Get shape
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                shape = "square" if len(approx) == 4 else "circle"
                
                # Get centroid
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                detected_objects.append({'color': color, 'shape': shape, 'coords': (cX, cY)})
                
        return detected_objects
        
    except Exception as e:
        logger.error(f"Error in OpenCV object detection for {image_path}: {e}")
        return []


def run_classic_agent_pipeline(vlm, dataset, judge_vlm):
    """
    Runs the Classic (CV-Enhanced) Agent pipeline on the dataset.
    """
    logger.info("Starting Classic Agent (CV-Enhanced) evaluation...")
    predictions = []
    scores = []
    
    try:
        for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Classic Agent Eval"):
            question = row['question']
            ground_truth = row['answer']
            image_path = row['image_path']
            
            # 1. Detect objects with OpenCV
            objects = detect_objects(image_path)
            if not objects:
                scene_context = "Scene Context: No objects were detected by the CV system."
            else:
                scene_context = "Scene Context: The following objects were detected:\n"
                for obj in objects:
                    scene_context += f"- A {obj['color']} {obj['shape']} at coordinates {obj['coords']}\n"
            
            # 2. Enhance the prompt
            enhanced_prompt = (
                f"{scene_context}\n"
                "Based *only* on the scene context provided above, answer the following question.\n"
                f"Question: {question}"
            )
            
            # 3. Get model's prediction
            model_answer_list = vlm.inference(prompt=enhanced_prompt, image_path=image_path)
            model_answer = model_answer_list[0] if model_answer_list else ""
            predictions.append(model_answer)
            
            # 4. Judge the prediction
            score = judge_answer(judge_vlm, question, model_answer, ground_truth)
            scores.append(score)

        # 5. Calculate final accuracy
        accuracy = sum(scores) / len(scores) if scores else 0
        logger.info(f"Classic Agent Evaluation Complete. Accuracy: {accuracy:.4f}")
        
        return accuracy, predictions

    except Exception as e:
        logger.error(f"Error during Classic Agent loop: {e}", exc_info=True)
        return 0.0, predictions
