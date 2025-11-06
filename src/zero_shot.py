import logging
from tqdm import tqdm
from src.llm_judge import judge_answer

logger = logging.getLogger(__name__)

def run_zero_shot(vlm, dataset, judge_vlm):
    """
    Runs zero-shot evaluation on the entire dataset.
    
    Args:
        vlm: The VLM model to test.
        dataset: The pandas DataFrame with questions and image paths.
        judge_vlm: The VLM model to use as a judge.
        
    Returns:
        tuple: (accuracy, predictions_list)
    """
    logger.info("Starting Zero-Shot evaluation...")
    predictions = []
    scores = []

    try:
        for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Zero-Shot Eval"):
            question = row['question']
            ground_truth = row['answer']
            image_path = row['image_path']
            
            # 1. Get model's prediction
            model_answer_list = vlm.inference(prompt=question, image_path=image_path)
            
            if not model_answer_list:
                logger.warning(f"VLM returned no answer for index {index}")
                model_answer = ""
            else:
                model_answer = model_answer_list[0]
            
            predictions.append(model_answer)
            
            # 2. Judge the prediction
            score = judge_answer(judge_vlm, question, model_answer, ground_truth)
            scores.append(score)

        # 3. Calculate final accuracy
        accuracy = sum(scores) / len(scores) if scores else 0
        logger.info(f"Zero-Shot Evaluation Complete. Accuracy: {accuracy:.4f}")
        
        return accuracy, predictions

    except Exception as e:
        logger.error(f"Error during Zero-Shot loop: {e}", exc_info=True)
        return 0.0, predictions
