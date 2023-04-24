import pandas as pd
import re
from decimal import Decimal
from collections import Counter

number_pattern = re.compile("number\\d+\\b")
num_token_pattern = re.compile("\[NUM\]")

number_prefix = "n"
const_prefix = "x"


def parse_equation(equation):
    stack = []
    tokens = equation.split()
    for i, token in enumerate(reversed(tokens)):
        if token == "1/":
            eq = f"( 1 / {stack.pop()} )"
        elif token in "+-*/":
            eq = f"{stack.pop()} {token} {stack.pop()}"
            if i < len(tokens) - 1:
                eq = f"( {eq} )"
        else:
            eq = token
        stack.append(eq)
    if len(stack) > 1:
        raise ValueError(f"Stack is not empty {stack} from {equation}")
    return stack.pop()


def calculate(parsed_eq, values):
    result = eval(parsed_eq, values)
    if isinstance(result, (float, int)):
        result = Decimal(str(result))
    return result.quantize(Decimal(1)) if result == result.to_integral() else result.normalize()


def process_problem(problem, question, numbers):
    numbers = numbers.split()

    text = number_pattern.sub(lambda x: numbers[int(x.group()[6:])], problem)
    problem = number_pattern.sub("[NUM]", problem)
    question = number_pattern.sub("[NUM]", question)
    numbers = [Decimal(n) for n in numbers]

    return text, problem, question, numbers


def to_const(c):
    c = str(c)
    if float(c) == int(float(c)):
        return f"{const_prefix}{int(float(c))}"
    return f"{const_prefix}{c.replace('.', '_')}"


def from_const(c):
    assert c.startswith(const_prefix)
    c = c[1:].replace("_", ".")
    if float(c) == int(float(c)):
        return int(float(c))
    return Decimal(c)


def convert(data):
    texts = []
    problems = []
    questions = []
    numbers = []
    constants = []
    equations = []
    answers = []

    # Recover Text
    for problem, question, nums, answer, equation in zip(
        data["Question"], data["Ques_Statement"], data["Numbers"], data["Answer"], data["Equation"]
    ):
        if not isinstance(question, str):
            question = problem

        origin_answer = calculate(
            parse_equation(equation.replace("number", number_prefix)),
            {f"{number_prefix}{i}": float(n) for i, n in enumerate(nums.split())},
        )

        if validate and not is_same_answer(answer, origin_answer):
            if not quiet:
                print("Wrong labeled answer :")
                print(problem)
                print(nums)
                print(equation)
                print(parse_equation(equation.replace("number", number_prefix)))
                print({f"{number_prefix}{i}": float(n) for i, n in enumerate(nums.split())})
                print()
            continue

        # Process problem
        text, problem, question, nums = process_problem(problem, question, nums)

        # Convert equation
        origin_equation = equation
        equation = " ".join(
            token.replace("number", number_prefix)
            if "number" in token is not None
            else (token if token in ("+", "-", "*", "/", "1/") else to_const(token))
            for token in equation.replace("/ 1.0 ", "1/ ").split()
        )
        consts = list(set(from_const(c) for c in equation.split() if c.startswith(const_prefix)))

        if not accept_constant and len(consts) > 0:
            continue

        equation = parse_equation(equation)
        answer = calculate(
            equation, {f"{number_prefix}{i}": n for i, n in enumerate(nums)} | {to_const(c): c for c in consts}
        )

        if validate and not is_same_answer(origin_answer, answer):
            print(text)
            print(problem)
            print(nums)
            print(origin_equation)
            print(equation)
            print({f"{number_prefix}{i}": n for i, n in enumerate(nums)} | {to_const(c): c for c in consts})
            print(origin_answer, answer)
            print()
            raise ValueError(f"Answer {answer} is different from given : {origin_answer}, equation={origin_equation}")

        texts.append(text)
        problems.append(problem)
        questions.append(question)
        numbers.append(nums)
        answers.append(answer)
        equations.append(equation)
        constants.append(consts)

    return texts, problems, questions, equations, answers, numbers, constants


def is_same_answer(a1, a2, max_fp=4):
    if a1 == a2:
        return True
    s1, s2 = str(a1), str(a2)
    fp = min(len(s1[s1.find(".") :]) - 1, len(s2[s2.find(".") :]) - 1, max_fp)
    return round(Decimal(s1), fp) == round(Decimal(s2), fp)


def write(raw, out_file):
    texts, problems, questions, equations, answers, numbers, constants = convert(raw)
    count_const = Counter(map(str, sum(constants, [])))

    output = pd.DataFrame(columns=["Text", "Problem", "Question", "Numbers", "Equation", "Answer", "Constants"])
    output["Text"] = texts
    output["Problem"] = problems
    output["Numbers"] = [" ".join(map(str, n)) for n in numbers]
    output["Equation"] = equations
    output["Answer"] = answers
    output["Question"] = questions
    output["Constants"] = [" ".join(map(str, c)) for c in constants]
    output.to_csv(out_file, index=False)

    print(out_file, ":", len(problems), "/", len(raw))
    print("constants :", sum(count_const.values()), count_const)
    print()


if __name__ == "__main__":
    validate = True
    accept_constant = True
    quiet = True

    write(
        pd.concat(
            [pd.read_csv(f".org_data/cv_mawps/fold0/train.csv"), pd.read_csv(f".org_data/cv_mawps/fold0/dev.csv")]
        ),
        "data/mawps/mawps.csv",
    )
    for fold in range(5):
        write(pd.read_csv(f".org_data/cv_mawps/fold{fold}/train.csv"), f"data/mawps/fold{fold}/train.csv")
        write(pd.read_csv(f".org_data/cv_mawps/fold{fold}/dev.csv"), f"data/mawps/fold{fold}/dev.csv")
        write(pd.read_csv(f".org_data/cv_mawps/fold{fold}/dev.csv"), f"data/mawps/fold{fold}/test.csv")
