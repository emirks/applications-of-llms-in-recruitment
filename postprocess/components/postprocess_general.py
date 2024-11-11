class GeneralResumeProcessor:

    def process(self, raw_text, user_id):
        return {
            "raw_text": raw_text,
            "user_id": user_id,
        }
    

class GeneralApplicationProcessor:

    def process(self, raw_text, job_id):
        return {
            "raw_text": raw_text,
            "job_id": job_id,
        }