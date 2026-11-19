"""Fixed prompt text. Change the wording only together with a new PROMPT_VERSION."""

PROMPT_VERSION = "v1"

ONE_IMAGE = (
    'A robot was asked: "{prompt}"\n'
    "The image shows the scene at the end of its attempt "
    "(top camera view, with left and right wrist camera views below).\n"
    "Did the robot fully complete the task? Answer with exactly one word: PASS or FAIL."
)

TWO_IMAGES = (
    'A robot was asked: "{prompt}"\n'
    "The first image shows the scene at the start of its attempt and the second image shows "
    "the scene at the end of its attempt (each: top camera view, with left and right wrist "
    "camera views below).\n"
    "Did the robot fully complete the task? Answer with exactly one word: PASS or FAIL."
)


def question(prompt: str, n_images: int) -> str:
    template = ONE_IMAGE if n_images == 1 else TWO_IMAGES
    return template.format(prompt=prompt)
