import os
import random
import pandas as pd
import json


def vary_questions(
    target_paths,
    group_key,
    base_train,
    base_val,
    base_test,
    add_train=True,
    num_key=None,
    eq_key=None,
    question_keys=None,
):
    groups = base_test.groupby(group_key)
    uniques = groups.filter(lambda x: len(x) == 1)
    if base_val is None:
        validation_set = uniques
    else:
        validation_set = pd.concat([base_val, uniques])
    test = groups.filter(lambda x: len(x) > 1)
    for seed in range(5):
        random.seed(seed * 100)
        indices = [random.choice(group.index) for _, group in groups if len(group) > 1]

        org_set = base_test.iloc[indices]
        if add_train:
            training_set = pd.concat([base_train, org_set])
        else:
            training_set = base_train
        test_set = test[~test.index.isin(indices)]
        datasets = [training_set, validation_set, test_set]

        if len(target_paths) == 4:
            org_test = []
            for i, context in zip(test_set.index, test_set[group_key]):
                org_sample = org_set.loc[org_set[group_key] == context]
                if len(org_sample) != 1:
                    raise ValueError(org_sample)
                org_sample = org_sample.squeeze()
                if len(org_sample[num_key].split()) != len(test_set[num_key][i].split()):
                    continue
                if org_sample[eq_key] == test_set[eq_key][i]:
                    continue
                sample = org_sample.copy()
                for question_key in question_keys:
                    sample[question_key] = test_set[question_key][i]
                org_test.append(sample)
            datasets.append(pd.DataFrame(org_test))

        for target_path, dataset in zip(target_paths, datasets):
            folder, filename = os.path.split(target_path)

            folder = os.path.join(folder, f"fold{seed}")
            path = os.path.join(folder, filename)
            os.makedirs(folder, exist_ok=True)
            print(path, ":", len(dataset))

            if "json" in filename:
                with open(path, "w", encoding="utf-8") as file:
                    json.dump(dataset.to_dict(orient="records"), file, indent=2, ensure_ascii=False)
            else:
                dataset.to_csv(path, index=False)
        print()
    print()


if __name__ == "__main__":
    vary_questions(
        target_paths=(
            ".org_data/svamp-1-n/train.csv",
            ".org_data/svamp-1-n/dev.csv",
            ".org_data/svamp-1-n/test.csv",
            ".org_data/svamp-1-n/org.csv",
        ),
        group_key="Body",
        base_train=pd.read_csv(f".org_data/mawps-asdiv-a_svamp/train.csv", dtype=str),
        base_val=None,
        base_test=pd.read_csv(f".org_data/mawps-asdiv-a_svamp/dev.csv", dtype=str),
        num_key="Numbers",
        eq_key="Equation",
        question_keys=("Question", "group_nums", "Ques"),
    )

    vary_questions(
        target_paths=(
            ".org_data/svamp-1-n-noadd/train.csv",
            ".org_data/svamp-1-n-noadd/dev.csv",
            ".org_data/svamp-1-n-noadd/test.csv",
        ),
        group_key="Body",
        base_train=pd.read_csv(f".org_data/mawps-asdiv-a_svamp/train.csv", dtype=str),
        base_val=None,
        base_test=pd.read_csv(f".org_data/mawps-asdiv-a_svamp/dev.csv", dtype=str),
        add_train=False,
    )

    vary_questions(
        target_paths=(
            ".org_data/umwp-1-n/train.json",
            ".org_data/umwp-1-n/valid.json",
            ".org_data/umwp-1-n/test.json",
            ".org_data/umwp-1-n/org.json",
        ),
        group_key="context",
        base_train=pd.read_json(f".org_data/umwp/train_src.json"),
        base_val=pd.read_json(f".org_data/umwp/valid_src.json"),
        base_test=pd.read_json(f".org_data/umwp/test_all.json"),
        num_key="nums",
        eq_key="output_prefix",
        question_keys=("id", "original_text", "question", "nums"),
    )

    vary_questions(
        target_paths=(
            ".org_data/umwp-1-n-noadd/train.json",
            ".org_data/umwp-1-n-noadd/valid.json",
            ".org_data/umwp-1-n-noadd/test.json",
        ),
        group_key="context",
        base_train=pd.read_json(f".org_data/umwp/train_src.json"),
        base_val=pd.read_json(f".org_data/umwp/valid_src.json"),
        base_test=pd.read_json(f".org_data/umwp/test_all.json"),
        add_train=False,
    )
