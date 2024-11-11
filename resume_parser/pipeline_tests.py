import json
import os

from pipeline_model_evaluator import PipelineModelEvaluator
from resume_parser import ResumeParser

def run_ocrtextlayer_pipelines(file_path):
    ocrtextlayer_pipelines = [f for f in os.listdir("yaml_configs") if f.startswith("ocrtextlayerparser_") 
                              and not f.startswith("ocrtextlayerparser_rag")]
    print(ocrtextlayer_pipelines)

    for pipeline in ocrtextlayer_pipelines:
        try: 
            resume_parser = ResumeParser(pipeline.split(".")[0])
            resume_parser.process_file(file_path)
        except Exception as e:
            print(f"Failed to run pipeline {pipeline}")
            print(e)

def run_tesseract_pipelines(file_path):
    tesseract_pipelines = [f for f in os.listdir("yaml_configs") if f.startswith("tesseract")]
    print(tesseract_pipelines)

    for pipeline in tesseract_pipelines:
        try: 
            resume_parser = ResumeParser(pipeline.split(".")[0])
            resume_parser.process_file(file_path)
        except Exception as e:
            print(f"Failed to run pipeline {pipeline}")
            print(e)

def run_ocrtextlayer_rag_pipelines(file_path):
    ocrtextlayer_rag_pipelines = [f for f in os.listdir("yaml_configs") if f.startswith("ocrtextlayerparser_rag")]
    ocrtextlayer_rag_pipelines = ["ocrtextlayerparser_raggemma9b.yaml"]
    print(ocrtextlayer_rag_pipelines)

    for pipeline in ocrtextlayer_rag_pipelines:
        try: 
            resume_parser = ResumeParser(pipeline.split(".")[0])
            resume_parser.process_file(file_path)
        except Exception as e:
            print(f"Failed to run pipeline {pipeline}")
            print(e)

def create_ground_truth_data(model_pipeline_name, data_folder): 
    resume_parser = ResumeParser(model_pipeline_name)
    result_folder = f"../results/{model_pipeline_name}-groundtruth"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    pdf_files = [f for f in os.listdir(f"{data_folder}/pdfs") if f.endswith(".pdf")]
    pdf_files = pdf_files[:10]
    evaluator = PipelineModelEvaluator()
    for pdf_file in pdf_files:
        try:
            result_path = f"{result_folder}/{pdf_file.split('.')[0]}.json"
            if os.path.exists(result_path):
                print(f"Skipping {pdf_file} since it has already been processed")
                continue

            json_file = pdf_file.replace(".pdf", ".json")
            json_path = f"{data_folder}/jsons/{json_file}"
            with open(json_path, "r") as f:
                reference_json = json.load(f)
            if not evaluator.check_json_suitable(reference_json):
                print(f"PDF for {pdf_file} is not suitable for ground truth creation")
                continue
            
            pdf_path = f"{data_folder}/pdfs/{pdf_file}"
            print(f"Processing {pdf_path}")
            json_output = resume_parser.process_file(pdf_path, save_jsons_to_space=False)

            resume_parser.save_json(f"{model_pipeline_name}-groundtruth", pdf_path, json_output)            
        except Exception as e:
            print(f"Failed to process {pdf_path}")
            print(e)



if __name__ == "__main__":
    # run_ocrtextlayer_pipelines(file_path)
    # run_tesseract_pipelines(file_path)
    # run_ocrtextlayer_rag_pipelines(file_path)
    data_folder = "../resume-crawler/data"
    create_ground_truth_data("geminiparser_gemini", data_folder)