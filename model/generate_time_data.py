import json
import random
import os

TITLES = [
    "buy milk", "buy bread", "buy groceries",
    "meeting with boss", "meeting with the team",
    "review PR #42", "review the deployment plan",
    "call the dentist", "call mom",
    "pick up kids from school",
    "finish the report", "finish the quarterly review",
    "send the invoice to accounting",
    "book a flight to Berlin",
    "fix the login bug",
    "write unit tests for the API",
    "update dependencies",
    "prepare slides for the demo",
    "pay electricity bill",
    "take the dog to the vet",
    "water the plants", "clean the kitchen",
    "read chapter 5 of the Go book",
    "deploy version 2.3 to staging",
    "schedule 1-on-1 with Anna",
    "order new monitor",
    "backup the database",
]

TIME_AFTER = [
    ("at 6pm", ["at", "6pm"]),
    ("at 19:00", ["at", "19:00"]),
    ("at 3:30pm", ["at", "3:30pm"]),
    ("at noon", ["at", "noon"]),
    ("at midnight", ["at", "midnight"]),
    ("today", ["today"]),
    ("today at 5pm", ["today", "at", "5pm"]),
    ("tomorrow", ["tomorrow"]),
    ("tomorrow at 10am", ["tomorrow", "at", "10am"]),
    ("tomorrow morning", ["tomorrow", "morning"]),
    ("tomorrow evening", ["tomorrow", "evening"]),
    ("tonight", ["tonight"]),
    ("next monday", ["next", "monday"]),
    ("next week", ["next", "week"]),
    ("next friday", ["next", "friday"]),
    ("this evening", ["this", "evening"]),
    ("this afternoon", ["this", "afternoon"]),
    ("in 20 minutes", ["in", "20", "minutes"]),
    ("in an hour", ["in", "an", "hour"]),
    ("in 2 hours", ["in", "2", "hours"]),
    ("by friday", ["by", "friday"]),
    ("by end of day", ["by", "end", "of", "day"]),
    ("by tomorrow", ["by", "tomorrow"]),
    ("by next week", ["by", "next", "week"]),
    ("after lunch", ["after", "lunch"]),
    ("before noon", ["before", "noon"]),
    ("on wednesday", ["on", "wednesday"]),
    ("on the weekend", ["on", "the", "weekend"]),
]

TIME_BEFORE = [
    ("tonight", ["tonight"]),
    ("tomorrow", ["tomorrow"]),
    ("later today", ["later", "today"]),
]

PRIORITY_EXPRESSIONS = [
    "urgent", "asap", "high priority", "low priority",
    "not important", "not urgent", "whenever possible",
    "when I have time", "it's important", "critical",
    "top priority", "urgently", "quickly", "immediately",
]


def make_example(text, bio_tags):
    return {"text": text, "bio": bio_tags}


def generate_dataset(n_per_class):
    data = []

    for _ in range(n_per_class):
        title = random.choice(TITLES)
        title_words = title.lower().split()

        if random.random() < 0.8:
            time_text, time_words = random.choice(TIME_AFTER)
            text = f"{title} {time_text}"
            tags = ["O"] * len(title_words) + ["B"] + ["I"] * (len(time_words) - 1)
        else:
            time_text, time_words = random.choice(TIME_BEFORE)
            text = f"{time_text} {title}"
            tags = ["B"] + ["I"] * (len(time_words) - 1) + ["O"] * len(title_words)

        data.append(make_example(text, tags))

    for _ in range(n_per_class):
        title = random.choice(TITLES)
        title_words = title.lower().split()
        tags = ["O"] * len(title_words)
        data.append(make_example(title, tags))

    for _ in range(n_per_class):
        title = random.choice(TITLES)
        title_words = title.lower().split()
        prio_text = random.choice(PRIORITY_EXPRESSIONS)
        prio_words = prio_text.lower().split()
        text = f"{title} {prio_text}"
        tags = ["O"] * (len(title_words) + len(prio_words))
        data.append(make_example(text, tags))

    for _ in range(n_per_class // 2):
        title = random.choice(TITLES)
        title_words = title.lower().split()
        prio_text = random.choice(PRIORITY_EXPRESSIONS)
        prio_words = prio_text.lower().split()
        time_text, time_words = random.choice(TIME_AFTER)
        text = f"{title} {prio_text} {time_text}"
        tags = ["O"] * len(title_words) + ["O"] * len(prio_words) + ["B"] + ["I"] * (len(time_words) - 1)
        data.append(make_example(text, tags))

    for _ in range(n_per_class // 2):
        title = random.choice(TITLES)
        title_words = title.lower().split()
        time_text, time_words = random.choice(TIME_AFTER)
        prio_text = random.choice(PRIORITY_EXPRESSIONS)
        prio_words = prio_text.lower().split()
        text = f"{title} {time_text} {prio_text}"
        tags = ["O"] * len(title_words) + ["B"] + ["I"] * (len(time_words) - 1) + ["O"] * len(prio_words)
        data.append(make_example(text, tags))

    random.shuffle(data)
    return data


def main():
    random.seed(43)
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    train = generate_dataset(400)
    val = generate_dataset(100)

    with open(os.path.join(out_dir, "time_train.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(out_dir, "time_val.json"), "w") as f:
        json.dump(val, f, indent=2)

    has_time = sum(1 for ex in train if "B" in ex["bio"])
    print(f"Time training: {len(train)} examples ({has_time} with time, {len(train)-has_time} without)")
    print(f"Time validation: {len(val)} examples")
    print(f"Sample: {json.dumps(train[0], indent=2)}")


if __name__ == "__main__":
    main()
