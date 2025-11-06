import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import logging

logger = logging.getLogger(__name__)

def load_dataset(csv_path, images_dir):
    """
    Loads the VQA dataset from a CSV file and validates image paths.
    
    Returns:
        pd.DataFrame or None: DataFrame with 'Image', 'question', 'answer', 'image_path'
                               or None if loading fails.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset from {csv_path} with {len(df)} rows.")
        
        # Create full image paths and check existence
        df['image_path'] = df['Image'].apply(lambda x: os.path.join(images_dir, f"{x}.png"))
        
        # Optional: Check for first few images to ensure path is correct
        if not os.path.exists(df['image_path'].iloc[0]):
             logger.warning(f"Image not found at: {df['image_path'].iloc[0]}. Please check --images_dir.")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"Dataset CSV file not found at: {csv_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def show_data(index, df):
    """
    Displays an image, question, and answer for a given index.
    """
    if index >= len(df):
        logger.error(f"Index {index} out of bounds for dataset of size {len(df)}.")
        return
        
    try:
        row = df.iloc[index]
        question = row['question']
        answer = row['answer']
        image_path = row['image_path']
        
        print(f"--- Sample {index} ---")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        
        if os.path.exists(image_path):
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.title(f"Image: {row['Image']}.png")
            plt.axis('off')
            plt.show()
        else:
            print(f"Image file not found: {image_path}")
            
    except Exception as e:
        logger.error(f"Error displaying data for index {index}: {e}")
