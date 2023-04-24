import pandas as pd
import re
import os
from decimal import Decimal

number_pattern = re.compile("number\\d+\\b")
num_token_pattern = re.compile("\[NUM\]")

number_prefix = "n"


def put_numbers(equation, numbers):
    return " ".join(
        repr(numbers[int(token[len(number_prefix) :])]) if token.startswith(number_prefix) else token
        for token in equation.split()
    )


def parse_equation(equation):
    stack = []
    tokens = equation.split()
    for i, token in enumerate(reversed(tokens)):
        if token in "+-*/":
            eq = f"{stack.pop()} {token} {stack.pop()}"
            if i < len(tokens) - 1:
                eq = f"( {eq} )"
        else:
            eq = token
        stack.append(eq)
    if len(stack) > 1:
        raise ValueError(f"Stack is not empty {stack} from {equation}")
    return stack.pop()


def calculate(parsed_eq):
    result = eval(parsed_eq)
    return result.quantize(Decimal(1)) if result == result.to_integral() else result.normalize()


def process_problem(problem, question, numbers):
    numbers = numbers.split()

    text = number_pattern.sub(lambda x: numbers[int(x.group()[6:])], problem)
    problem = number_pattern.sub("[NUM]", problem)
    question = number_pattern.sub("[NUM]", question)
    numbers = [Decimal(n) for n in numbers]

    return text, problem, question, numbers


def convert(data):
    texts = []
    problems = []
    questions = []
    numbers = []
    equations = []
    answers = []

    # Recover Text
    for problem, question, nums, answer, equation in zip(
        data["Question"], data["Ques_Statement"], data["Numbers"], data["Answer"], data["Equation"]
    ):
        if not isinstance(question, str):
            question = problem

        origin_answer = calculate(
            put_numbers(
                parse_equation(equation.replace("number", number_prefix)), [Decimal(str(n)) for n in nums.split()]
            )
        )

        if validate and not is_same_answer(answer, origin_answer):
            if not quiet:
                print("Wrong labeled answer :")
                print(problem)
                print(nums)
                print(equation)
                print(parse_equation(equation.replace("number", number_prefix)))
                print(
                    put_numbers(
                        parse_equation(equation.replace("number", number_prefix)),
                        [Decimal(str(n)) for n in nums.split()],
                    ),
                    "=",
                    origin_answer,
                    "!=",
                    answer,
                )
                print()
            continue

        # Process problem
        text, problem, question, nums = process_problem(problem, question, nums)

        # Convert equation
        equation = " ".join(token.replace("number", number_prefix) for token in equation.split())
        equation = parse_equation(equation)

        answer = calculate(put_numbers(equation, nums))

        if validate and not is_same_answer(origin_answer, answer):
            print(problem)
            print(nums)
            print(equation)
            print(put_numbers(equation, nums))
            print(origin_answer, answer)
            print()
            raise ValueError(f"Answer {answer} is different from given : {origin_answer}")

        texts.append(text)
        problems.append(problem)
        questions.append(question)
        numbers.append(nums)
        answers.append(answer)
        equations.append(equation)

    return texts, problems, questions, equations, answers, numbers


def is_same_answer(a1, a2, max_fp=4):
    if a1 == a2:
        return True
    s1, s2 = str(a1), str(a2)
    fp = min(len(s1[s1.find(".") :]) - 1, len(s2[s2.find(".") :]) - 1, max_fp)
    return round(Decimal(s1), fp) == round(Decimal(s2), fp)


def write(raw, out_file):
    texts, problems, questions, equations, answers, numbers = convert(raw)

    output = pd.DataFrame(columns=["Text", "Problem", "Question", "Numbers", "Equation", "Answer"])
    output["Text"] = texts
    output["Problem"] = problems
    output["Numbers"] = [" ".join(map(str, n)) for n in numbers]
    output["Equation"] = equations
    output["Answer"] = answers
    output["Question"] = questions
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    output.to_csv(out_file, index=False)

    print(out_file, ":", len(problems), "/", len(raw))
    print()


if __name__ == "__main__":
    validate = True
    quiet = False

    write(
        pd.concat(
            [pd.read_csv(f".org_data/cv_asdiv-a/fold0/train.csv"), pd.read_csv(f".org_data/cv_asdiv-a/fold0/dev.csv")]
        ),
        "data/asdiv-a/asdiv-a.csv",
    )
    for fold in range(5):
        write(pd.read_csv(f".org_data/cv_asdiv-a/fold{fold}/train.csv"), f"data/asdiv-a/fold{fold}/train.csv")
        write(pd.read_csv(f".org_data/cv_asdiv-a/fold{fold}/dev.csv"), f"data/asdiv-a/fold{fold}/dev.csv")
        write(pd.read_csv(f".org_data/cv_asdiv-a/fold{fold}/dev.csv"), f"data/asdiv-a/fold{fold}/test.csv")
