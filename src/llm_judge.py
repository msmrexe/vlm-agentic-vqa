import logging
import re

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator for a Visual Question Answering task.
Your goal is to determine if the "Model's Answer" correctly and concisely answers the "Question" based on the "Ground Truth Answer".

The answer must be semantically equivalent, even if phrased differently.
For "Is the..." questions, the answer must match (e.G., "left" matches "on the left").
For "What is the shape..." questions, the answer must be the shape (e.g., "square" matches "a red square").
For "What is the color..." questions, the answer must be the color (e.g., "red" matches "the red one").

Respond with only "Yes" or "No". Do not provide any explanation.

---
Question: "{question}"
Ground Truth Answer: "{ground_truth}"
Model's Answer: "{model_answer}"
---

Is the Model's Answer correct?
"""

def judge_answer(vlm, question, model_answer, ground_truth):
    """
    Uses the VLM to judge if a model's answer is correct given the ground truth.
    
    Returns:
        int: 1 if correct, 0 if incorrect.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer
    )
    
    try:
        response = vlm.inference(prompt=prompt, image_path=None, max_new_tokens=5)
        
        if not response:
            logger.warning("Judge VLM returned no response.")
            return 0
            
        # Clean the response
        decision = response[0].strip().lower()
        
        # Use regex to find 'yes' or 'no'
        if re.search(r'\byes\b', decision):
            return 1
        elif re.search(r'\bno\b', decision):
            return 0
        else:
            logger.warning(f"Judge VLM returned ambiguous response: '{response[0]}'")
            # Default to incorrect if response is not clear
            return 0
            
    except Exception as e:
        logger.error(f"Error during LLM judge inference: {e}")
        return 0
