import json
import random
import os

TITLES = [
    "buy milk", "buy milk today at 19:00", "buy bread", "buy groceries",
    "meeting with boss tomorrow at 12:15", "meeting with the team on Friday",
    "meeting with boss", "meeting with the client", "meeting",
    "review PR #42", "review the deployment plan",
    "call the dentist", "call mom", "call the plumber",
    "call the vet", "call the client", "call John", "call the manager",
    "pick up kids from school", "pick up the dry cleaning",
    "finish the report by Friday", "finish the quarterly review",
    "finish the report", "finish the presentation", "finish the task",
    "send the invoice to accounting", "send email to the client",
    "send the proposal", "send the update",
    "book a flight to Berlin", "book hotel for the conference",
    "fix the login bug", "fix CSS on the landing page",
    "write unit tests for the API", "write documentation for the SDK",
    "update dependencies", "update the README",
    "prepare slides for the demo", "prepare the budget proposal",
    "pay electricity bill", "pay rent",
    "take the dog to the vet", "take out the trash",
    "water the plants", "clean the kitchen",
    "read chapter 5 of the Go book", "read the architecture RFC",
    "deploy version 2.3 to staging", "deploy the hotfix",
    "schedule 1-on-1 with Anna", "schedule dentist appointment",
    "order new monitor", "order lunch for the team",
    "backup the database", "backup photos to cloud",
    "check the API", "check the database", "check the backup",
    "run the deployment", "run the tests",
    "plan the meeting", "plan the demo",
]

CREATE_TEMPLATES = [
    "add {t}",
    "create {t}",
    "new task {t}",
    "new task: {t}",
    "remind me to {t}",
    "remember to {t}",
    "schedule {t}",
    "I need to {t}",
    "i need to {t}",
    "don't forget to {t}",
    "don't let me forget to {t}",
    "please add {t}",
    "can you add {t}",
    "put {t} on my list",
    "add to my list: {t}",
    "{t}",
    "todo: {t}",
    "task: {t}",
    "I have to {t}",
    "i should {t}",
    "gotta {t}",
    "need to {t}",
    "must {t}",
    "plan to {t}",
    "set a reminder to {t}",
    "set reminder for {t}",
    "make sure to {t}",
    "make sure I {t}",
    "note to self: {t}",
    "jot down {t}",
]

COMPLETE_TEMPLATES = [
    "mark {t} as done",
    "complete {t}",
    "finish {t}",
    "done with {t}",
    "{t} is done",
    "{t} is finished",
    "{t} is complete",
    "check off {t}",
    "I finished {t}",
    "i finished {t}",
    "I completed {t}",
    "I did {t}",
    "i'm done with {t}",
    "just finished {t}",
    "mark {t} as complete",
    "mark {t} as finished",
    "mark {t} done",
    "tick off {t}",
    "{t} done",
    "checked {t}",
]

DELETE_TEMPLATES = [
    "delete {t}",
    "remove {t}",
    "scratch {t}",
    "cancel {t}",
    "delete the {t}",
    "remove the {t}",
    "cancel the {t}",
    "drop {t}",
    "get rid of {t}",
    "I don't need {t} anymore",
    "i don't need {t} anymore",
    "forget about {t}",
    "never mind {t}",
    "nevermind {t}",
    "remove {t} from my list",
    "take {t} off my list",
    "discard {t}",
    "trash {t}",
    "nah forget {t}",
    "skip {t}",
    "don't need {t}",
    "clear {t}",
    "erase {t}",
]

LIST_TEXTS = [
    "show all",
    "show everything",
    "show my tasks",
    "show me my tasks",
    "show me everything",
    "show me my list",
    "show me what I have to do",
    "list all",
    "list everything",
    "list my tasks",
    "list my todos",
    "list todos",
    "what do I have to do",
    "what do i have to do",
    "what's on my list",
    "what's on my plate",
    "what tasks do I have",
    "what are my tasks",
    "what are my todos",
    "anything to do",
    "any tasks",
    "do I have any tasks",
    "do i have anything to do",
    "give me my tasks",
    "give me the list",
    "all tasks",
    "all todos",
    "everything",
    "my tasks",
    "my list",
    "pending tasks",
    "what's pending",
    "what needs to be done",
    "what haven't I done yet",
    "open tasks",
    "show open tasks",
    "what's left",
    "remaining tasks",
    "how many tasks do I have",
    "overview",
]

UNKNOWN_TEXTS = [
    # greetings and chat
    "hello", "hi", "hey", "good morning", "good evening",
    "how are you", "what's up", "what time is it",
    "who are you", "what can you do",
    "thank you", "thanks", "ok", "okay", "sure",
    "help", "help me", "I'm confused",
    "nevermind", "forget it",
    "yes", "no", "maybe", "I don't know",
    "goodbye", "bye", "see you later",
    "this is cool", "nice", "awesome",
    # questions and requests that aren't todo commands
    "tell me a joke", "what's the weather like",
    "what is the meaning of life",
    "can you sing", "tell me something interesting",
    "how does this work", "what is this app",
    "I'm bored", "entertain me",
    "calculate 2 + 2", "what's 10 times 5",
    "translate hello to Spanish",
    "who won the game last night",
    # verb + object patterns that look like commands but aren't todos
    "fly to the moon",
    "run a marathon next week",
    "cook dinner for 10 people",
    "drive to the airport",
    "swim across the lake",
    "climb mount everest",
    "sing me a song",
    "draw a picture of a cat",
    "play some music",
    "open the door",
    "close the window",
    "turn off the lights",
    "turn on the TV",
    "start the car",
    "stop the engine",
    "build a house",
    "paint the walls blue",
    "move the couch",
    "throw the ball",
    "catch the bus",
    "ride a bike to work",
    "walk the dog around the park",
    "feed the fish",
    "wash the dishes",
    "iron my shirt",
    "charge my phone",
    "restart the computer",
    "download the latest version",
    "upload the file to the server",
    "print the document",
    "scan the barcode",
    "take a photo",
    "record a video",
    "send a message to John",
    "call an ambulance",
    "order pizza",
    "book a table for two",
    "find the nearest gas station",
    "search for flights to Paris",
    "check the stock price",
    "measure the room",
    "count the chairs",
    "sort the files by date",
    "compare these two options",
    "explain quantum physics",
    "describe the sunset",
    "predict the weather tomorrow",
    "analyze the data",
    "summarize the article",
    "rewrite this paragraph",
    "fix the leaking faucet",
    "replace the battery",
    "install the update",
    "configure the router",
    "debug the program",
    "deploy to production",
    "rollback the release",
    "merge the branches",
    "I want to go home",
    "I need a vacation",
    "let's grab lunch",
    "time to sleep",
    "it's raining outside",
    "the server is down",
    "my computer is slow",
    "that meeting was boring",
    "I love this app",
    "this doesn't make sense",
]


def make_bio_tags(template, title, text):
    words = text.lower().split()
    title_words = title.lower().split()
    tags = ["O"] * len(words)

    title_start = -1
    for i in range(len(words) - len(title_words) + 1):
        if words[i:i + len(title_words)] == title_words:
            title_start = i
            break

    if title_start >= 0:
        tags[title_start] = "B"
        for j in range(1, len(title_words)):
            tags[title_start + j] = "I"

    return tags


def generate_examples(templates, intent, titles, count):
    examples = []
    for _ in range(count):
        title = random.choice(titles)
        template = random.choice(templates)
        text = template.format(t=title)
        tags = make_bio_tags(template, title, text)
        examples.append({"text": text, "intent": intent, "bio": tags})
    return examples


def generate_list_examples(texts, count):
    examples = []
    for _ in range(count):
        text = random.choice(texts)
        words = text.lower().split()
        tags = ["O"] * len(words)
        examples.append({"text": text, "intent": "list", "bio": tags})
    return examples


def generate_unknown_examples(texts, count):
    examples = []
    for _ in range(count):
        text = random.choice(texts)
        words = text.lower().split()
        tags = ["O"] * len(words)
        examples.append({"text": text, "intent": "unknown", "bio": tags})
    return examples


def generate_dataset(n_per_intent):
    data = []
    data.extend(generate_examples(CREATE_TEMPLATES, "create", TITLES, n_per_intent))
    data.extend(generate_examples(COMPLETE_TEMPLATES, "complete", TITLES, n_per_intent))
    data.extend(generate_examples(DELETE_TEMPLATES, "delete", TITLES, n_per_intent))
    data.extend(generate_list_examples(LIST_TEXTS, n_per_intent))
    data.extend(generate_unknown_examples(UNKNOWN_TEXTS, n_per_intent))
    random.shuffle(data)
    return data


def main():
    random.seed(42)
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    train = generate_dataset(750)
    val = generate_dataset(150)

    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump(train, f, indent=2)

    with open(os.path.join(out_dir, "val.json"), "w") as f:
        json.dump(val, f, indent=2)

    intent_counts = {}
    for ex in train:
        intent_counts[ex["intent"]] = intent_counts.get(ex["intent"], 0) + 1

    print(f"Training examples: {len(train)}")
    print(f"Validation examples: {len(val)}")
    print(f"Intent distribution (train): {intent_counts}")
    print(f"Sample: {json.dumps(train[0], indent=2)}")


if __name__ == "__main__":
    main()
