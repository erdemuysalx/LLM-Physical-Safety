from judge import process_csv_files

paths = [#"out/codellama-13b-instruct/codellama-13b-instruct_violation_output.csv",
         #"out/codellama-13b-instruct/codellama-13b-instruct_unintentional_output.csv",
         #"out/codellama-13b-instruct/codellama-13b-instruct_deliberate_output.csv",
         # "out/codellama-34b-instruct/codellama-34b-instruct_violation_output.csv",
         # "out/codellama-34b-instruct/codellama-34b-instruct_unintentional_output.csv",
         # "out/codellama-34b-instruct/codellama-34b-instruct_deliberate_output.csv"
         "out/gemini-pro_icl/gemini-pro_deliberate_output.csv"]

for path in paths:
    print("file:", path, "to be judged...")
    process_csv_files(path)
