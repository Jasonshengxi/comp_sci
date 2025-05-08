from __future__ import annotations

is_main = __name__ == "__main__"
if is_main:
    print("\x1b[?1049h")
from typing import Any, Callable
from colorama import Fore, Style

if is_main:
    print(Fore.CYAN, end="")
print("Loading stdlib...")
from time import time
import json
from collections import defaultdict
import os
from os import get_terminal_size, path
import atexit
from string import ascii_letters
import random
import pickle
from dataclasses import dataclass
import re

INIT_TIME = time()

print("Loading ML Library")
from sklearn.metrics.pairwise import cosine_similarity


def embedding_similarity(x, y):
    return cosine_similarity([x], [y])[0][0]


model = None


def encode(s: str):
    if model is not None:
        return model.encode(s)
    else:
        raise Exception("Model wasn't initialised.")


ANSI_ESCAPE = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")


def escape_ansi(line):
    return ANSI_ESCAPE.sub("", line)


@dataclass
class Question:
    question: str
    attempts: int
    successes: int
    answers: list[str]
    embeddings: list

    @staticmethod
    def default(question_data: dict[str, list[str]], question: str) -> Question:
        return Question(question, 0, 0, question_data[question], [])

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0

    def try_init(self, question_data: dict[str, list[str]]):
        if not self.embeddings:
            self.init_embeddings()
        if not self.answers:
            self.answers = question_data[self.question]

    def init_embeddings(self):
        begin = time()
        init_model()
        print("Cache not found. Embedding answers.")
        self.embeddings = [
            encode(f"{self.question}: {answer}") for answer in self.answers
        ]
        took = time() - begin
        print(f"Took {Fore.BLUE}{round(took, 3)}{Fore.RESET}s to compute embeddings")


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


def clear_screen():
    print("\x1b[2J")


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


@dataclass
class AnswerStatus:
    data: Question
    statuses: list[list[str]]

    @staticmethod
    def new(data: Question) -> AnswerStatus:
        return AnswerStatus(data, [list(hangman(answer)) for answer in data.answers])

    @staticmethod
    def new_correct(data: Question) -> AnswerStatus:
        return AnswerStatus(data, [list(answer) for answer in data.answers])

    def display(self):
        print(f"{Fore.RED}Q{Fore.RESET}: {Fore.YELLOW}{self.data.question}{Fore.RESET}")
        for i, status in enumerate(self.statuses):
            if "_" in status:
                print(f"{Fore.BLUE}{i + 1}{Fore.RESET}. {"".join(status)}")
            else:
                print(
                    f"{Fore.BLUE}{i + 1}{Fore.RESET}. {Fore.GREEN}{"".join(status)}{Fore.RESET}"
                )

    def reveal_word(self, answer_index: int, char_index: int):
        status = self.statuses[answer_index]
        j = char_index
        while j < len(status) and status[j] == "_":
            self.statuses[answer_index][j] = self.data.answers[answer_index][j]
            j += 1
        j = char_index - 1
        while j >= 0 and status[j] == "_":
            self.statuses[answer_index][j] = self.data.answers[answer_index][j]
            j -= 1

    def reveal_correct(self, response: str):
        embedding = encode(f"{self.data.question}: {response}")
        for i in range(len(self.statuses)):
            answer_embedding = self.data.embeddings[i]
            similarity = embedding_similarity(embedding, answer_embedding)
            print(f"{i + 1} : {round(similarity * 100, 2)}%")
            if similarity > 0.9:
                self.reveal_answer(i)

    def reveal_answer(self, index: int):
        self.statuses[index] = list(self.data.answers[index])

    def reveal_all(self):
        self.statuses = [list(x) for x in self.data.answers]

    def finished(self) -> bool:
        return all("_" not in s for s in self.statuses)

    def reveal_random_word(self):
        while True:
            answer_index = random.randint(0, len(self.statuses) - 1)
            status = self.statuses[answer_index]
            char_index = random.randint(0, len(status) - 1)

            c = status[char_index]
            if c == "_":
                break
        self.reveal_word(answer_index, char_index)


def do_question_with_init(
    question_data: dict[str, list[str]], data: dict[str, Question], question: str
):
    if question not in data:
        data[question] = Question.default(question_data, question)
    do_question(question_data, data[question])


def do_question(
    question_data: dict[str, list[str]],
    question: Question,
):
    question.try_init(question_data)
    question.attempts += 1
    statuses = AnswerStatus.new(question)
    while True:
        print()
        statuses.display()
        print()
        init_model()

        if statuses.finished():
            question.successes += 1
            break
        response = input(">> ")
        clear_screen()
        if response.lower() == "i give up":
            statuses.reveal_all()
            statuses.display()
            break
        elif response.lower().startswith("hint"):
            last_part = response.rsplit(" ", maxsplit=1)[-1]
            try:
                x = int(last_part)
            except:
                x = 1

            for _ in range(x):
                if statuses.finished():
                    break
                statuses.reveal_random_word()
        elif response.lower() == "skip":
            question.attempts -= 1
            break
        else:
            statuses.reveal_correct(response)


def do_memorise(question_data: dict[str, list[str]], questions: list[Question]):
    for question in questions:
        AnswerStatus.new_correct(question).display()
        print()
    init_model()

    m = input(
        f"{Fore.CYAN}Press enter once you're ready. (the questions will disappear){Fore.RESET}"
    )
    if m:
        print(f"{Fore.RED}Memorise has been cancelled.{Fore.RESET}")
        return

    clear_screen()
    random.shuffle(questions)
    for question in questions:
        do_question(question_data, question)


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

    last_response = ""
    while True:
        response = input("> ").lower()
        clear_screen()

        if not response:
            response = last_response
        else:
            last_response = response

        if response == "print":
            print_sorted(data, lambda x: x.success_rate, other_columns)
        elif response == "random":
            index = random.randint(0, len(questions) - 1)
            do_question_with_init(question_data, data, questions[index])
        elif response == "unseen":
            while True:
                index = random.randint(0, len(questions) - 1)
                if questions[index] in data:
                    continue
                break
            do_question_with_init(question_data, data, questions[index])
        elif response == "seen":
            while True:
                index = random.randint(0, len(questions) - 1)
                if questions[index] in data:
                    break
            do_question_with_init(question_data, data, questions[index])
        elif response == "bye":
            break
        elif response.startswith("memorise"):
            last_part = response.rsplit(" ", maxsplit=1)[-1]
            try:
                x = int(last_part)
            except:
                x = 1
            qs = random.choices(questions, k=x)
            term_size = get_terminal_size()
            total_height = sum(len(question_data[q]) + 2 for q in qs) + 3
            if total_height >= term_size.lines:
                print(
                    f"{Fore.RED}The questions won't fit on the screen. Try fewer questions.{Fore.RESET}"
                )
            else:
                for q in qs:
                    if q not in data:
                        data[q] = Question.default(question_data, q)
                do_memorise(question_data, [data[q] for q in qs])


if is_main:
    try:
        main()
    except Exception as e:
        print(e)
        input(f"{Fore.RED}it crashed.{Fore.RESET}")
else:
    print("Ready.")
