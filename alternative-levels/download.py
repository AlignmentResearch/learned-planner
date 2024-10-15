#!/usr/bin/env python3


import requests


# Slugify function
def slugify(text):
    return text.lower().replace(" ", "-").replace(".", "").replace("'", "")


# List of level names
level_names = """
Intro
Sokoban
Sokoban Jr. 1
Sokoban Jr. 2
Deluxe
Sokogen 990602
Xsokoban
David Holland 1
David Holland 2
Howard's 1st set
Howard's 2nd set
Howard's 3rd set
Howard's 4th set
Sasquatch
Mas Sasquatch
Sasquatch III
Sasquatch IV
Still more levels
Nabokosmos
Microcosmos
Microban
Simple sokoban
Dimitri and Yorick
Yoshio Automatic
"""

# Iterate through each level name
for name in level_names.strip().split("\n"):
    # Slugify the name
    n_slug = slugify(name)

    # Create the URL
    url = f"http://borgar.net/programs/sokoban/levels/{name}.txt"

    # Fetch the content (mocked with a placeholder here)
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content to a file with the slugified name
        with open(f"{n_slug}.txt", "w") as file:
            file.write(response.text)
        print(f"Saved {n_slug}.txt")
    else:
        print(f"Failed to download {n_slug}.txt, status code: {response.status_code}")
        print(response.reason)
        exit(-1)
