import logging
from tqdm import tqdm
from src.llm_judge import judge_answer

logger = logging.getLogger(__name__)

def run_dl_agent_pipeline(vlm, dataset, judge_vlm):
    """
    Runs the DL (Chain-of-Thought) Agent pipeline on the dataset.
    This simulates a ReAct/CoT agent by breaking the problem down.
    """
    logger.info("Starting DL Agent (Chain-of-Thought) evaluation...")
    predictions = []
    scores = []
    
    try:
        for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="DL Agent Eval"):
            question = row['question']
            ground_truth = row['answer']
            image_path = row['image_path']
            
            # 1. Agent Step 1: Decompose the question
            plan_prompt = (
                f"To answer the question '{question}', "
                "what is the step-by-step reasoning plan I should follow? "
                "List the steps."
            )
            plan_list = vlm.inference(prompt=plan_prompt, image_path=image_path, max_new_tokens=100)
            plan = plan_list[0] if plan_list else "No plan generated."

            # 2. Agent Step 2: Extract visual context
            extract_prompt = (
                "Describe all objects in the image in detail. For each object, "
                "list its color, its shape, and its relative position (e.g., top-left, bottom-right)."
            )
            context_list = vlm.inference(prompt=extract_prompt, image_path=image_path, max_new_tokens=150)
            context = context_list[0] if context_list else "No context extracted."

            # 3. Agent Step 3: Synthesize the final answer
            final_prompt = (
                f"You are a reasoning agent. Use the following information to answer the question.\n\n"
                f"Original Question: {question}\n\n"
                f"Reasoning Plan:\n{plan}\n\n"
                f"Image Context:\n{context}\n\n"
                "Based on the plan and context, what is the final, concise answer to the original question?"
            )
            model_answer_list = vlm.inference(prompt=final_prompt, image_path=image_path)
            model_answer = model_answer_list[0] if model_answer_list else ""
            predictions.append(model_answer)
            
            # 4. Judge the prediction
            score = judge_answer(judge_vlm, question, model_answer, ground_truth)
            scores.append(score)

        # 5. Calculate final accuracy
        accuracy = sum(scores) / len(scores) if scores else 0
        logger.info(f"DL Agent Evaluation Complete. Accuracy: {accuracy:.4f}")
        
        return accuracy, predictions

    except Exception as e:
        logger.error(f"Error during DL Agent loop: {e}", exc_info=True)
        return 0.0, predictions
