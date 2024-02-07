import os

def get_condor_job_ad_settings(variable):
    file_path = os.environ.get('_CONDOR_JOB_AD')
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            print(line)
            if line.startswith(variable + ' = '):
                value = line.split('=')[1].strip()
                return value
    return None