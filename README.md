# VLM Agentic Visual Question Answering

This repository presents an in-depth exploration of agent-based systems for solving complex Visual Question Answering (VQA) tasks. It moves beyond simple zero-shot prompting to build and evaluate two distinct agentic pipelines. 

The first, a **'Classic Agent'**, demonstrates tool-use by integrating with OpenCV to extract explicit scene geometry and object data, which is then fed to a Vision-Language Model (VLM). The second, a **'DL Agent'**, implements a Chain-of-Thought (CoT) reasoning process, forcing the VLM to decompose the problem, analyze the visual evidence, and synthesize an answer step-by-step. 

This project serves as a comprehensive comparison of these advanced methods against a standard zero-shot baseline, all evaluated using an 'LLM-as-Judge' pipeline. It was developed for the System 2 AI (M.S.) course.

## Features

* **Zero-Shot Baseline:** Evaluates the raw performance of the Qwen VLM.
* **Classic Agent Pipeline:** Uses OpenCV for object detection (shape, color, location) to provide structured context to the VLM.
* **DL Agent Pipeline:** Implements a Chain-of-Thought (CoT) process, forcing the VLM to decompose, extract, and synthesize information before answering.
* **LLM-as-Judge:** Employs the VLM itself as a semantic evaluator to score the correctness of answers.
* **Modular & Scriptable:** All logic is refactored into clean Python scripts, runnable via command-line arguments.

## Core Concepts & Techniques

* **Visual Question Answering (VQA):** The task of answering natural language questions about an image.
* **Vision-Language Models (VLMs):** Models (like Qwen-VL) that are pre-trained on vast amounts of image and text data, enabling them to understand and reason about both modalities.
* **Agentic AI:** The concept of using an LLM as a "reasoning engine" or "agent" that can make plans, use tools, and process information to solve complex tasks.
* **Chain-of-Thought (CoT):** A prompting technique that encourages an LLM to "think step-by-step," breaking down a problem into intermediate reasoning steps, which significantly improves performance on complex tasks.
* **Computer Vision (OpenCV):** Used in the "Classic" agent to provide an external, "grounded" tool for object detection. This demonstrates how agents can leverage external functions.
* **LLM-as-Judge:** A modern evaluation technique that uses a powerful LLM to judge the quality or correctness of another model's output, overcoming the limitations of exact-match string comparison.

---

## How It Works

This project evaluates three methods for solving a VQA task. The core goal is to see how much we can improve upon a simple, zero-shot baseline by adding agentic reasoning.

### 1. Baseline: Zero-Shot VLM

This is the control group. We provide the VLM with the image and the question and ask for an answer directly. This measures the model's out-of-the-box reasoning capability.

* **Logic (`src/zero_shot.py`):** `VLM(image, "question") -> "answer"`
* **Analysis:** The baseline model achieved **46% accuracy**. This result is decent but reveals key weaknesses. The model struggles with complex spatial reasoning (e.g., "furthest from the gray object?") and relational questions. It often provides answers that are "plausible" but factually incorrect in the context of the specific image, demonstrating a gap in precise grounding.

### 2. Approach 1: Classic Agent (CV-Enhanced)

This agent simulates a "tool-use" pipeline. We first use a "classic" computer vision tool (OpenCV) to analyze the image and extract a "ground truth" list of all objects, their properties, and their exact coordinates. This structured data is then fed to the VLM *along with* the question.

* **Logic (`src/agent_pipelines/classic_agent.py`):**
    1.  `detect_objects(image_path)`: Uses `cv2` to read the image.
    2.  Converts to HSV color space for reliable color detection.
    3.  Applies color masks for red, green, blue, yellow, and gray.
    4.  Finds contours (`cv2.findContours`) in the masks.
    5.  For each contour, it determines the shape (using `cv2.approxPolyDP`) and centroid (using `cv2.moments`).
    6.  This produces a list: `[{'color': 'red', 'shape': 'square', 'coords': (50, 28)}, ...]`.
    7.  This list is formatted into a "Scene Context" string and pre-pended to the prompt.
* **Analysis:** The Classic Agent achieved **61% accuracy**. This 15-point increase is significant. It demonstrates that **grounding the VLM with explicit, factual data** dramatically improves its reasoning. The VLM is no longer "guessing" the scene's contents; it is *reasoning over a provided knowledge base*. Errors that remain are likely due to failures in the OpenCV pipeline (e.g., poor color masking) or the VLM's inability to correctly parse the provided context.

### 3. Approach 2: DL Agent (Chain-of-Thought)

This agent uses no external tools. Instead, it leverages the VLM's *own internal reasoning* by forcing it to "think step-by-step." This is a lightweight implementation of a Chain-of-Thought (CoT) or ReAct (Reasoning + Action) agent.

* **Logic (`src/agent_pipelines/dl_agent.py`):**
    1.  **Decompose:** The agent first asks the VLM, "To answer '[QUESTION]', what is my step-by-step plan?" This generates a plan.
    2.  **Extract:** The agent then asks the VLM, "Describe all objects in the image in detail (color, shape, position)." This extracts context.
    3.  **Synthesize:** Finally, the agent provides the VLM with the original question, the plan, and the context, asking it to "Use the plan and context to find the final answer."
* **Analysis:** This pipeline is expected to score highly, likely surpassing the Classic Agent (e.g., 75-85% accuracy). By forcing the model to decompose the problem and observe the scene *before* committing to an answer, we mitigate the risk of a fast, "System 1" error. This mimics a more robust "System 2" reasoning process and represents a powerful, tool-free method for enhancing LLM accuracy.

### 4. Evaluation: LLM-as-Judge

We can't just check if `model_answer == ground_truth`. A simple string comparison is too brittle. For example, if the question is "What is the shape of the red object?" and the ground truth is **"square"**:

* A model answer of **"it is a square"** is semantically *correct*, but would fail a simple `==` check.
* A model answer of **"the red square"** is also *correct*, but would fail the check.

We need to evaluate *semantic meaning*, not just exact phrasing.

* **Logic (`src/llm_judge.py`):**
    * We feed a separate VLM instance a prompt containing the question, the ground truth answer, and the model's actual answer.
    * We instruct it to act as an expert evaluator and respond with only "Yes" or "No" based on whether the model's answer is semantically correct.
    * This provides a robust *semantic* evaluation of correctness.

---

## Project Structure

```
VLM-Agentic-VQA/
├── .gitignore                        # Ignores logs, data, and cache files
├── LICENSE                           # MIT License
├── requirements.txt                  # Project dependencies
├── README.md                         # This file
├── data/
│   ├── vqa_dataset.csv               # Dataset file
│   └── images/
│       └── ....png                   # Dataset images
├── logs/
│   └── .gitkeep                      # Placeholder for log files
├── notebooks/
│   └── VQA_Agent_Evaluation.ipynb    # Guided notebook to run all tests
├── scripts/
│   └── evaluate_agents.py            # Main CLI script to run evaluations
└── src/
    ├── __init__.py
    ├── agent_pipelines/              # Contains the different agent logics
    │   ├── __init__.py
    │   ├── classic_agent.py          # Logic for OpenCV-enhanced agent
    │   └── dl_agent.py               # Logic for Chain-of-Thought agent
    ├── data_loader.py                # Loads and visualizes the dataset
    ├── llm_judge.py                  # Implements the LLM-as-Judge
    ├── models.py                     # QwenVLM wrapper class
    ├── utils.py                      # Logging setup
    └── zero_shot.py                  # Logic for the baseline evaluation
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/vlm-agentic-vqa.git
    cd vlm-agentic-vqa
    ```

2.  **Setup Environment and Data:**
    * Install all required packages:
        ```bash
        pip install -r requirements.txt
        ```
    * If you inted to use different data, place your `vqa_dataset.csv` file in the `data/` directory and all your `.png` image files in the `data/images/` directory. Otherwise, the data used for this project would already be available in the aforementioned folders.

3.  **Run Evaluations (Two Ways):**

    **A) Guided Notebook (Recommended):**
    * Open and run the cells in `notebooks/VQA_Agent_Evaluation.ipynb`. This will guide you through each evaluation step-by-step.

    **B) Command-Line Interface:**
    * You can run any evaluation directly from your terminal. All results will be printed and saved to `logs/evaluation.log`.

        ```bash
        # Run the baseline Zero-Shot evaluation
        python scripts/evaluate_agents.py --mode zero_shot
        
        # Run the "Classic" OpenCV-enhanced agent
        python scripts/evaluate_agents.py --mode classic
        
        # Run the "DL" Chain-of-Thought agent
        python scripts/evaluate_agents.py --mode dl
        
        # Run all three evaluations sequentially
        python scripts/evaluate_agents.py --mode all
        
        # Show a sample from the dataset
        python scripts/evaluate_agents.py --mode show_sample --sample_index 5
        ```

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
