import os.path
from collections import Counter
import pandas as pd

number_prefix = "n"
const_prefix = "x"


def convert(raw):
    data = pd.DataFrame(raw)
    texts = []
    problems = []
    questions = []
    numbers = []
    equations = []
    answers = []
    constants = []
    for text, question, eq, nums in zip(data["original_text"], data["question"], data["output_infix"], data["nums"]):
        nums = [num for num in nums.split()]
        text = text.strip()
        question = question.strip()
        problem = text
        for num in sorted(nums, key=lambda x: len(x), reverse=True):
            problem = problem.replace(num, " [NUM] ", 1)
            question = question.replace(num, " [NUM] ", 1)
        assert not any(num in problem for num in nums), f"{text=} {nums=}"

        num_values = [round((eval(n[:-1]) / 100) if n[-1] == "%" else eval(n), 4) for n in nums]
        answer = round(eval(eq, {f"N{i}": v for i, v in enumerate(num_values)}), 4)

        equation = " ".join(
            token.replace("N", number_prefix)
            if token in "+-**/()" or token[0] == "N"
            else const_prefix + token.replace(".", "_")
            for token in eq.split()
        )
        consts = [token[1:].replace("_", ".") for token in equation.split() if token.startswith(const_prefix)]

        texts.append(text)
        problems.append(problem)
        questions.append(question)
        numbers.append(" ".join(map(str, num_values)))
        equations.append(equation)
        answers.append(answer)
        constants.append(" ".join(consts))
    return texts, problems, questions, numbers, equations, answers, constants


def write(in_file, out_file):
    raw = pd.read_json(in_file)

    texts, problems, questions, numbers, equations, answers, constants = convert(raw)
    output = pd.DataFrame(columns=["Text", "Problem", "Question", "Numbers", "Equation", "Answer", "Constants"])
    output["Text"] = texts
    output["Problem"] = problems
    output["Question"] = questions
    output["Numbers"] = numbers
    output["Equation"] = equations
    output["Answer"] = answers
    output["Constants"] = constants
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    output.to_csv(out_file, index=False, encoding="utf-8")

    count_const = Counter(sum([c.split() for c in constants], []))
    print(out_file, ":", len(problems), "/", len(raw))
    print("constants :", sum(count_const.values()), count_const)
    print()


write(".org_data/umwp/train_src.json", "data/umwp/train.csv")
write(".org_data/umwp/valid_src.json", "data/umwp/dev.csv")
write(".org_data/umwp/test_all.json", "data/umwp/test.csv")

for fold in range(5):
    write(f".org_data/umwp-1-n/fold{fold}/train.json", f"data/umwp-1-n/fold{fold}/train.csv")
    write(f".org_data/umwp-1-n/fold{fold}/valid.json", f"data/umwp-1-n/fold{fold}/dev.csv")
    write(f".org_data/umwp-1-n/fold{fold}/test.json", f"data/umwp-1-n/fold{fold}/test.csv")
    write(f".org_data/umwp-1-n/fold{fold}/org.json", f"data/umwp-1-n/fold{fold}/org.csv")

    write(f".org_data/umwp-1-n-noadd/fold{fold}/train.json", f"data/umwp-1-n-noadd/fold{fold}/train.csv")
    write(f".org_data/umwp-1-n-noadd/fold{fold}/valid.json", f"data/umwp-1-n-noadd/fold{fold}/dev.csv")
    write(f".org_data/umwp-1-n-noadd/fold{fold}/test.json", f"data/umwp-1-n-noadd/fold{fold}/test.csv")
