from model_re import medical_re

# medical_re.run_train()
medical_re.load_schema(medical_re.config.PATH_SCHEMA)
medical_re.run_train()