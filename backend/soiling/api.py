
import hug


@hug.get('/{event}', versions=1)
def soiling(event: str):
    return ""
