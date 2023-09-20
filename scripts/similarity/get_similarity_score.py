import json
import logging
import os
import yaml
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    filename='app_similarity_score.log',
    filemode='w',
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("app_similarity_score.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

cwd = os.getcwd()

READ_RESUME_FROM = os.path.join(cwd, 'Data', 'Processed', 'Resumes')
READ_JOB_DESCRIPTION_FROM = os.path.join(cwd, 'Data', 'Processed', 'JobDescription')
config_path = os.path.join(cwd, "scripts", "similarity")

def read_config(filepath):
    try:
        with open(filepath) as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError as e:
        logger.error(f"Configuration file {filepath} not found: {e}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in configuration file {filepath}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error reading configuration file {filepath}: {e}")
    return None

class CosineSimilarityCalculator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def calculate_cosine_similarity(self, text1, text2):
        try:
            inputs = self.tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}", exc_info=True)

def get_similarity_score(resume_string, job_description_string):
    logger.info("Started getting similarity score")
    
    similarity_calculator = CosineSimilarityCalculator()
    similarity_score = similarity_calculator.calculate_cosine_similarity(resume_string, job_description_string)
    
    logger.info("Finished getting similarity score")
    return similarity_score

if __name__ == "__main__":
    resume_dict = read_config(READ_RESUME_FROM + "/Resume-bruce_wayne_fullstack.pdf4783d115-e6fc-462e-ae4d-479152884b28.json")
    job_dict = read_config(READ_JOB_DESCRIPTION_FROM + "/JobDescription-job_desc_full_stack_engineer_pdf4de00846-a4fe-4fe5-a4d7-2a8a1b9ad020.json")
    resume_keywords = resume_dict["extracted_keywords"]
    job_description_keywords = job_dict["extracted_keywords"]

    resume_string = ' '.join(resume_keywords)
    jd_string = ' '.join(job_description_keywords)
    
    final_result = get_similarity_score(resume_string, jd_string)
    
    print(f"Cosine Similarity Score: {final_result}")
