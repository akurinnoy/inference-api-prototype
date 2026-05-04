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

PRIORITY_AFTER = [
    ("urgent", ["urgent"]),
    ("asap", ["asap"]),
    ("high priority", ["high", "priority"]),
    ("low priority", ["low", "priority"]),
    ("not important", ["not", "important"]),
    ("not urgent", ["not", "urgent"]),
    ("whenever possible", ["whenever", "possible"]),
    ("when I have time", ["when", "i", "have", "time"]),
    ("it's important", ["it's", "important"]),
    ("critical", ["critical"]),
    ("top priority", ["top", "priority"]),
]

PRIORITY_BEFORE = [
    ("urgently", ["urgently"]),
    ("quickly", ["quickly"]),
    ("immediately", ["immediately"]),
]

TIME_EXPRESSIONS = [
    "today", "today at 5pm", "today at 19:00",
    "tomorrow", "tomorrow at 10am", "tomorrow morning", "tomorrow evening",
    "tonight", "next monday", "next week", "next friday",
    "this evening", "this afternoon",
    "in 20 minutes", "in an hour", "in 2 hours",
    "by friday", "by end of day", "by tomorrow", "by next week",
    "after lunch", "before noon",
    "on wednesday", "on the weekend",
    "at 6pm", "at 19:00", "at 3:30pm", "at noon",
]


def make_example(text, bio_tags):
    return {"text": text, "bio": bio_tags}


def generate_dataset(n_per_class):
    data = []

    for _ in range(n_per_class):
        title = random.choice(TITLES)
        title_words = title.lower().split()

        if random.random() < 0.7:
            prio_text, prio_words = random.choice(PRIORITY_AFTER)
            text = f"{title} {prio_text}"
            tags = ["O"] * len(title_words) + ["B"] + ["I"] * (len(prio_words) - 1)
        else:
            prio_text, prio_words = random.choice(PRIORITY_BEFORE)
            text = f"{prio_text} {title}"
            tags = ["B"] + ["I"] * (len(prio_words) - 1) + ["O"] * len(title_words)

        data.append(make_example(text, tags))

    for _ in range(n_per_class):
        title = random.choice(TITLES)
        title_words = title.lower().split()
        tags = ["O"] * len(title_words)
        data.append(make_example(title, tags))

    for _ in range(n_per_class):
        title = random.choice(TITLES)
        title_words = title.lower().split()
        time_text = random.choice(TIME_EXPRESSIONS)
        time_words = time_text.lower().split()
        text = f"{title} {time_text}"
        tags = ["O"] * (len(title_words) + len(time_words))
        data.append(make_example(text, tags))

    for _ in range(n_per_class // 2):
        title = random.choice(TITLES)
        title_words = title.lower().split()
        time_text = random.choice(TIME_EXPRESSIONS)
        time_words = time_text.lower().split()
        prio_text, prio_words = random.choice(PRIORITY_AFTER)
        text = f"{title} {time_text} {prio_text}"
        tags = ["O"] * len(title_words) + ["O"] * len(time_words) + ["B"] + ["I"] * (len(prio_words) - 1)
        data.append(make_example(text, tags))

    for _ in range(n_per_class // 2):
        title = random.choice(TITLES)
        title_words = title.lower().split()
        prio_text, prio_words = random.choice(PRIORITY_AFTER)
        time_text = random.choice(TIME_EXPRESSIONS)
        time_words = time_text.lower().split()
        text = f"{title} {prio_text} {time_text}"
        tags = ["O"] * len(title_words) + ["B"] + ["I"] * (len(prio_words) - 1) + ["O"] * len(time_words)
        data.append(make_example(text, tags))

    random.shuffle(data)
    return data


def main():
    random.seed(42)
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    train = generate_dataset(400)
    val = generate_dataset(100)

    with open(os.path.join(out_dir, "priority_train.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(out_dir, "priority_val.json"), "w") as f:
        json.dump(val, f, indent=2)

    has_prio = sum(1 for ex in train if "B" in ex["bio"])
    print(f"Priority training: {len(train)} examples ({has_prio} with priority, {len(train)-has_prio} without)")
    print(f"Priority validation: {len(val)} examples")
    print(f"Sample: {json.dumps(train[0], indent=2)}")


if __name__ == "__main__":
    main()
