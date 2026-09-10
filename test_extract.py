import pprint
from question_manager import QuestionManager

qm = QuestionManager()
topics = qm.extract_questions_from_files()
for k, v in topics.items():
    print(f"--- {k} ---")
    for t in v:
        print("  -", repr(t['topic']))
