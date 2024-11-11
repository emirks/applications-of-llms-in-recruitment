import json
import re
from datetime import datetime
import logging

class ExperienceProcessor:

    def process(self, json_contents):
        # Calculate total years of experience from the resume
        cv_experience = self.calculate_total_years_of_experience(json_contents.get("experiences", []))

        return {
            "experience_years_total": cv_experience
        }

    def calculate_total_years_of_experience(self, experiences):
        total_years = 0
        duration_pattern = re.compile(r"^\d+[MY]$")
        date_ranges = []
        
        for exp in experiences:
            job_title = exp.get('job_title', 'Unknown Job Title')
            start_date_str = exp.get('start_date')
            end_date_str = exp.get('end_date')
            duration = exp.get('duration')
            
            # Try to calculate the duration from start_date and end_date first
            if start_date_str is not None and end_date_str is not None:
                start_date = datetime.strptime(start_date_str, "%Y-%m")
                
                if end_date_str.lower() == "present":
                    end_date = datetime.now()
                else:
                    end_date = datetime.strptime(end_date_str, "%Y-%m")
                
                # Add this date range to the list
                date_ranges.append((start_date, end_date))
                logging.info(f"Duration for '{job_title}' calculated from start_date and end_date: {start_date} to {end_date}.")
            
            # If start_date and end_date are not available, use the duration field
            elif duration:
                if not duration_pattern.match(duration):
                    raise ValueError(f"Duration format '{duration}' is invalid. Expected format: ^(\\d+[MY])$")
                
                if duration.endswith("Y"):  # e.g., "2Y" for 2 years
                    years = int(duration[:-1])
                    total_years += years
                    logging.info(f"Duration for '{job_title}' calculated from duration field: {years} years.")
                elif duration.endswith("M"):  # e.g., "18M" for 18 months
                    months = int(duration[:-1])
                    total_years += months / 12
                    logging.info(f"Duration for '{job_title}' calculated from duration field: {months / 12:.2f} years.")
        
            else:
                logging.info(f"Duration cannot be calculated for '{job_title}': start_date, end_date, and duration are all None.")

        # Merge overlapping and consecutive date ranges
        if date_ranges:
            # Sort the date ranges by start date
            date_ranges.sort(key=lambda x: x[0])
            merged_ranges = [date_ranges[0]]

            for current_start, current_end in date_ranges[1:]:
                last_start, last_end = merged_ranges[-1]

                if current_start <= last_end:
                    # If the current range overlaps or touches the last range, merge them
                    merged_ranges[-1] = (last_start, max(last_end, current_end))
                    logging.info("Found overlapping experiences, merging them.")
                else:
                    # Otherwise, add the current range as a new entry
                    merged_ranges.append((current_start, current_end))

            # Sum the total duration from the merged ranges
            for start, end in merged_ranges:
                duration_years = (end - start).days / 365.25
                total_years += duration_years

        logging.info(f"Total Experience Years: {total_years:.2f}")
        return total_years
        
    def extract_required_experience_years(self, jd_content_json):
        """
        Extract the number of years of experience required from job description.
        
        Args:
            jd_content_json (dict): job description json
        
        Returns:
            int: Number of years of experience required.
        """
        return jd_content_json['experience_years_required']['minimum_years']