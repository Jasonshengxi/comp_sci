print("\x1b[?1049h")
from typing import Any, Callable
from colorama import Fore, Style

print(Fore.CYAN, end="")
print("Loading stdlib...")
from time import time
import json
from collections import defaultdict
import os
from os import path
import atexit
from string import ascii_letters
from random import randint
import pickle
from dataclasses import dataclass
import re

INIT_TIME = time()

print("Loading ML Library")
from sklearn.metrics.pairwise import cosine_similarity


def embedding_similarity(x, y):
    return cosine_similarity([x], [y])[0][0]


model = None

ANSI_ESCAPE = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")


def escape_ansi(line):
    return ANSI_ESCAPE.sub("", line)


@dataclass
class Question:
    attempts: int
    successes: int
    answers: list[str]
    embeddings: list

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0


DATA_PATH = "describe/data.p"


def load_data() -> dict[str, Question]:
    try:
        with open(DATA_PATH, "rb") as file:
            return pickle.load(file)
    except:
        return dict()


def store_data(data: dict[str, Question]):
    with open(DATA_PATH, "wb") as file:
        pickle.dump(data, file)


def load_questions() -> dict[str, list[str]]:
    with open("out/Describe.json") as file:
        data: dict = json.load(file)

    things = data["Thing"]
    items = data["Item"]

    questions = defaultdict(list)

    for i in range(len(things)):
        questions[things[i]].append(items[i])

    return dict(questions)


def hangman(x: str) -> str:
    out = ""
    for c in x:
        if c in ascii_letters:
            out += "_"
        else:
            out += c
    return out


def init_model():
    global model

    if model is None:
        print(f"{Fore.CYAN}Loading Sentence Transformer...{Fore.RESET}")
        begin = time()
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Took {Fore.BLUE}{round(time() - begin, 3)}{Fore.RESET}s")


def do_question(
    question_data: dict[str, list[str]], data: dict[str, Question], question: str
):
    answers = question_data[question]
    if question not in data or not data[question].embeddings:
        begin = time()
        init_model()
        print("Cache not found. Embedding answers.")
        data[question] = Question(
            0, 0, answers, [model.encode(f"{question}: {answer}") for answer in answers]
        )
        took = time() - begin
        print(f"Took {Fore.BLUE}{round(took, 3)}{Fore.RESET}s to compute embeddings")
    else:
        print("Cache found.")
    data[question].attempts += 1
    statuses = [list(hangman(answer)) for answer in answers]
    die = False
    while True:
        print()
        print(f"{Fore.RED}Q{Fore.RESET}: {Fore.YELLOW}{question}{Fore.RESET}")
        for i, status in enumerate(statuses):
            if "_" in status:
                print(f"{Fore.BLUE}{i + 1}{Fore.RESET}. {"".join(status)}")
            else:
                print(
                    f"{Fore.BLUE}{i + 1}{Fore.RESET}. {Fore.GREEN}{"".join(status)}{Fore.RESET}"
                )
        print()
        init_model()

        if die:
            break
        if all("_" not in s for s in statuses):
            data[question].successes += 1
            break
        response = input(">> ")
        print("\x1b[2J")
        if response.lower() == "i give up":
            statuses = [list(a) for a in answers]
            die = True
        elif response.lower().startswith("hint"):
            last_part = response.rsplit(" ", maxsplit=1)[-1]
            x = 1
            try:
                x = int(last_part)
            except:
                pass

            for _ in range(x):
                if all("_" not in s for s in statuses):
                    break
                while True:
                    answer_index = randint(0, len(answers) - 1)
                    status = statuses[answer_index]
                    char_index = randint(0, len(status) - 1)

                    c = status[char_index]
                    if c == "_":
                        break

                j = char_index
                while status[j] == "_":
                    statuses[answer_index][j] = answers[answer_index][j]
                    j += 1
                j = char_index - 1
                while status[j] == "_":
                    statuses[answer_index][j] = answers[answer_index][j]
                    j -= 1

        elif response.lower() == "skip":
            data[question].attempts -= 1
            break
        else:
            embedding = model.encode(f"{question}: {response}")
            for i in range(len(answers)):
                answer_embedding = data[question].embeddings[i]
                similarity = embedding_similarity(embedding, answer_embedding)
                print(f"{i + 1} : {round(similarity * 100, 2)}%")
                if similarity > 0.9:
                    statuses[i] = list(answers[i])


def print_sorted(
    data: dict[str, Question],
    key: Callable[[Question], int | float],
    display: Callable[[Question], list[str]],
    count: int = 10000000,
):
    keyed = list(data.items())
    keyed.sort(key=lambda x: key(x[1]))
    count = min(count, len(keyed))
    max_width = max(len(keyed[i][0]) for i in range(count))

    displays = [display(q) for _, q in keyed]
    row_count = len(displays[0])
    display_max_width = [
        max(len(escape_ansi(d[i])) for d in displays) for i in range(row_count)
    ]

    print("─" * max_width, end="")
    for i in range(row_count):
        print("─┬─" + "─" * display_max_width[i], end="")
    print()
    for i in range(count):
        question = keyed[i][0]
        print(
            Fore.GREEN + " " * (max_width - len(question)) + question,
            end=f"{Fore.RESET} │ ",
        )
        for j in range(row_count):
            disp = displays[i][j]
            end = " │ " if j != row_count - 1 else ""
            needs_len = display_max_width[j] - len(escape_ansi(disp))
            print(f"{disp}{" " * needs_len}", end=end)
        print()


def main():
    print("Loading Question Data...")
    question_data = load_questions()
    questions = list(question_data.keys())
    questions.sort()

    print("Loading Other Data...")
    data = load_data()
    atexit.register(lambda: store_data(data))
    atexit.register(lambda: print("\x1b[?1049l"))
    print(Fore.YELLOW + "Ready.")

    init_took = time() - INIT_TIME
    print(
        f"{Fore.RESET}Time to initialise: {Fore.BLUE}{round(init_took, 3)}{Fore.RESET}s"
    )

    def other_columns(x: Question) -> list[str]:
        return [
            f"{Fore.BLUE}{x.successes}{Fore.RESET}/{Fore.BLUE}{x.attempts}{Fore.RESET}",
            f"{Fore.YELLOW}{round(x.success_rate * 100, 2)}%{Fore.RESET}",
        ]

    while True:
        response = input("> ").lower()
        print("\x1b[2J")

        if response == "print":
            print_sorted(data, lambda x: x.success_rate, other_columns)
        elif response == "random":
            index = randint(0, len(questions) - 1)
            do_question(question_data, data, questions[index])
        elif response == "unseen":
            while True:
                index = randint(0, len(questions) - 1)
                if questions[index] in data:
                    continue
                break
            do_question(question_data, data, questions[index])
        elif response == "seen":
            while True:
                index = randint(0, len(questions) - 1)
                if questions[index] in data:
                    break
            do_question(question_data, data, questions[index])
        elif response == "bye":
            break


if __name__ == "__main__":
    main()
