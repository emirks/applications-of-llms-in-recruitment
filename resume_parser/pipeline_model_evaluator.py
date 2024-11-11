import os
import json
from collections.abc import Mapping
from Levenshtein import distance

from resume.resume_parser.resume_parser import ResumeParser

class PipelineModelEvaluator:
    def __init__(self) -> None:
        pass

    def check_json_suitable(self, json_data) -> bool:
        if json_data["lang_name"] != "English":
            return False
        
        work_experiences = json_data["user"]["work_experiences"]
        educations = json_data["user"]["educations"]
        # We do not test CV labels without work experiences or educations for now
        if not isinstance(work_experiences, list) or len(work_experiences) == 0:
            return False
        if not isinstance(educations, list) or len(educations) == 0:
            return False
        return True
    

    def clean_reference_json(self, reference_user_json: dict) -> dict:
        keys_to_remove = [
            "aasm_state", "avatar", "avatar_medium_url", "avatar_small_url", "desired_job_type",
            "expected_salary_score", "graduate_degree_type", "graduate_year", "headline", "industries",
            "is_freelancer", "is_premium", "job_search_progress", "job_search_progress_score",
            "last_seen_at", "last_seen_at_by_2_weeks", "last_seen_at_by_month", "last_seen_at_days_ago_level",
            "most_received_reputation_category", "most_recent_education", "most_recent_education_normalized_school",
            "most_recent_work_experience", "number_of_management", "organizations", "profession",
            "profile_completion_ratio", "profile_completion_ratio_score", "relevant_work_experience",
            "rn_work_experience", "reputation_credit_count_by_category", "user_industries", "username",
            "work_experience", "work_experience_score"
        ]
        for key in keys_to_remove:
            reference_user_json.pop(key, None)
        reference_user_json["about"] = reference_user_json.get("description_truncated", "")
        reference_user_json.pop("description_truncated", None)
        for work_experience in reference_user_json["work_experiences"]:
            work_experience["organization"] = work_experience["organization"]["name"]
        for education in reference_user_json["educations"]:
            education["organization"] = education["organization"]["name"]
        return reference_user_json

    def evaluate(
        self,
        reference_user_json: dict,
        output_json: dict,
    ) -> list[dict]:
        reference_user_json = self.clean_reference_json(reference_user_json)
        result = self.json_diff(reference_user_json, output_json)
        return result
    
    def json_diff(self, o1, o2):
        def get_levenshtein_distance(o1: str, o2: str):
            max_len = max(len(x) for x in [o1, o2])

            # Normalized Levenshtein distance in [0,1]
            score = 1
            if max_len > 0:
                score = 1 - (distance(o1, o2) / max_len)

            return score

        if isinstance(o1, dict) and isinstance(o2, dict):
            if len(o1) == 0 and len(o2) == 0:
                return 1
            elif len(o1) == 0 or len(o2) == 0:
                return 0

            # We only look at the intersection of the keys
            # Mismatch is not penalized in any way
            all_keys = set(o1.keys()).union(set(o2.keys()))

            base_scores = []
            for key in all_keys:
                base_score = self.json_diff(o1.get(key), o2.get(key))
                if base_score is None:
                    continue

                base_scores.append(base_score)

            return sum(base_scores) / len(base_scores)
        elif isinstance(o1, list) and isinstance(o2, list):
            if len(o1) == 0 and len(o2) == 0:
                return 1
            elif len(o1) == 0 or len(o2) == 0:
                return 0

            base_scores = [self.json_diff(e1, e2) for (e1, e2) in zip(o1, o2)]
            base_scores = [s for s in base_scores if s is not None]
            return sum(base_scores) / len(base_scores)
        elif isinstance(o1, str) and isinstance(o2, str):
            return get_levenshtein_distance(o1, o2)
        elif (isinstance(o1, int) or isinstance(o1, float)) and (
            isinstance(o2, int) or isinstance(o2, float)
        ):
            if o2 == 0 and o1 == 0:
                return 1
            else:
                return 1 - abs(o2 - o1) / (abs(o2) + abs(o1))
        elif o1 is None and o2 is None:
            return 1
        elif o1 is None or o2 is None:
            return 0
        else:
            return 0
        
    def evaluate_model_performance(self, model_pipeline_name, data_folder):
        resume_parser = ResumeParser(model_pipeline_name)
        if not os.path.exists(f"../results/{model_pipeline_name}"):
            os.makedirs(f"../results/{model_pipeline_name}")

        pdf_files = [f for f in os.listdir(f"{data_folder}/pdfs") if f.endswith(".pdf")]
        pdf_files = pdf_files[:10]
        all_scores = []
        for pdf_file in pdf_files:
            try:
                result_path = f"../results/{model_pipeline_name}/{pdf_file.split('.')[0]}.json"
                if os.path.exists(result_path):
                    # load the score from the json file
                    with open(result_path, "r") as f:
                        result_json = json.load(f)
                    all_scores.append(result_json["score"])
                    print(f"Skipping {pdf_file} since it has already been processed")
                    continue

                json_file = pdf_file.replace(".pdf", ".json")
                json_path = f"{data_folder}/jsons/{json_file}"
                with open(json_path, "r") as f:
                    reference_json = json.load(f)

                if not self.check_json_suitable(reference_json):
                    print(f"Label JSON for {pdf_file} is not suitable for evaluation")
                    continue

                pdf_path = f"{data_folder}/pdfs/{pdf_file}"
                print(f"Processing {pdf_path}")
                json_output = resume_parser.process_file(pdf_path, save_jsons_to_space=False)

                score = self.evaluate(reference_json["user"], json_output)
                print(f"Score: {score}")
                all_scores.append(score)

                output_json = {
                    "path": pdf_file.split(".")[0],
                    "output": json_output,
                    "reference": reference_json["user"],
                    "score": score
                }
                resume_parser.save_json(model_pipeline_name, pdf_path, output_json)
            except Exception as e:
                # print(f"Failed to process {pdf_path}")
                print(e)
        
        overall_score = sum(all_scores) / len(all_scores)
        print(f"{len(all_scores)} files processed")
        print(f"Overall Score: {overall_score}")

if __name__ == "__main__":
    data_folder = "../resume-crawler/data"

    evaluator = PipelineModelEvaluator()
    evaluator.evaluate_model_performance("only_gemini_parser", data_folder)
