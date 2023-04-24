from collections import Counter
import os
import json
import pandas as pd
import regex as re

num_pattern = re.compile(r"(?<![A-Z]+\d*)\d+(?:\.\d+)?%?|\(\s+\d+\s+/\s+\d+\s+\)")
template_pattern = re.compile("temp_[a-z]")
space_pattern = re.compile(r"\s{2,}")
number_prefix = "n"
const_prefix = "x"


def tonum(i):
    try:
        if int(i) == float(i):
            return int(i)
    except ValueError:
        pass
    return float(i)


def convert(raw):
    data = pd.DataFrame(raw)
    texts = []
    problems = []
    questions = []
    numbers = []
    equations = []
    answers = []
    constants = []
    for text, temp_text, template, nums, answer in zip(
        data["new_split"], data["text"], data["target_template"], data["num_list"], data["answer"]
    ):
        temp_nums = set(t for t in template if t.startswith("temp_"))
        assert template[0] == "x" and template[1] == "=", f"{template} is not correct"
        template = template[2:]
        assert len(set(template_pattern.findall(temp_text))) == len(
            nums
        ), f"{temp_text}, {set(template_pattern.findall(temp_text))} {template=}, {temp_nums=}, {nums=}"
        assert len(nums) >= len(temp_nums), f"{text}, {template=}, {temp_nums=}, {nums=}"
        temp_nums = {t: nums[ord(t[5]) - ord("a")] for t in temp_nums}
        try:
            eval(" ".join(template).replace("^", "**"), {"PI": 3.1416} | temp_nums)
        except Exception as e:
            print(e.args[0])
            print("Wrong equation :", " ".join(template), temp_nums)
            continue

        if "^" in template:
            if any(temp_nums[template[i + 1]] != 2 for i, t in enumerate(template) if t == "^"):
                continue
            for i in (i + 1 for i, t in enumerate(template) if t == "^"):
                template[i] = "2"

        text = text.replace("", "").replace("", "").strip()
        text = text.replace("d ㎡", " d㎡")
        text = text.replace("m2", " ㎡")
        text = space_pattern.sub(" ", text)
        text = text.replace("C10", "C 10")
        problem = num_pattern.sub("[NUM]", text).strip()
        problem = problem.replace("[NUM]dm", "[NUM] dm")
        qe = max(problem.rfind("？"), problem.rfind("?"))
        p = problem[:qe] + problem[qe]
        t = p[:-5]
        question = p[max(t.rfind("．"), t.rfind("."), t.rfind(","), t.rfind("，")) + 1 :].strip()
        assert len(question) > 0, f"{problem=} {question=}"
        text_nums = [n.replace(" ", "") for n in num_pattern.findall(text)]
        text_num_values = [round((eval(n[:-1]) / 100) if n[-1] == "%" else eval(n), 4) for n in text_nums]

        try:
            match_num = {t: text_num_values.index(round(value, 4)) for t, value in temp_nums.items()}
        except ValueError as e:
            print(text)
            print(problem)
            print(temp_nums)
            print(text_nums)
            print()
            raise e
        # assert len(text_nums) == len(nums), f"{text}, {problem}, {template=}, {text_nums=}, {nums=}"

        equation = " ".join(
            f"{number_prefix}{match_num[t]}"
            if t.startswith("temp")
            else t.replace("^", "**")
            if t in "+-*/^()"
            else f"{const_prefix}{'3_1416' if t == 'PI' else t.replace('.', '_')}"
            for t in template
        )
        consts = [
            "3.1416" if c == "PI" else str(tonum(c))
            for c in sorted(set(t for t in template if not (t.startswith("temp") or t in "+-*/()^")))
        ]

        texts.append(text)
        problems.append(problem)
        questions.append(question)
        numbers.append(" ".join(text_nums))
        equations.append(equation)
        answers.append(answer)
        constants.append(" ".join(consts))
    return texts, problems, questions, numbers, equations, answers, constants


def write(in_file, out_file):
    with open(in_file, encoding="utf-8") as f:
        raw = json.load(f)

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


for fold in range(5):
    write(f".org_data/cv_math23k/train_{fold}.json", f"data/math23k/fold{fold}/train.csv")
    write(f".org_data/cv_math23k/test_{fold}.json", f"data/math23k/fold{fold}/dev.csv")
    write(f".org_data/cv_math23k/test_{fold}.json", f"data/math23k/fold{fold}/test.csv")

write(".org_data/math23k_train_valid_test/train23k_processed_nodup.json", "data/math23k/train.csv")
write(".org_data/math23k_train_valid_test/valid23k_processed_nodup.json", "data/math23k/dev.csv")
write(".org_data/math23k_train_valid_test/test23k_processed_nodup.json", "data/math23k/test.csv")

pd.concat(
    [
        pd.read_csv("data/math23k/train.csv", dtype=str),
        pd.read_csv("data/math23k/dev.csv", dtype=str),
        pd.read_csv("data/math23k/test.csv", dtype=str),
    ]
).to_csv("data/math23k/math23k.csv", index=False, encoding="utf-8")

write(".org_data/math23k_train_test/combined_train23k_processed_nodup.json", "data/math23k-traintest/train.csv")
write(".org_data/math23k_train_test/test23k_processed_nodup.json", "data/math23k-traintest/dev.csv")
write(".org_data/math23k_train_test/test23k_processed_nodup.json", "data/math23k-traintest/test.csv")
