import os
import re

import numpy as np
import skimage.segmentation


def process_game_levels(file_content):
    processed_content = []
    level = []
    for files_line in file_content.split("\n"):
        if "#" in files_line and not re.findall("[0-9]", files_line):
            # first_hash = line.find("#")
            # last_hash = line.rfind("#")
            # new_line = "#" * first_hash + line[first_hash:last_hash] + (len(line) - last_hash) * "#"
            # assert len(line) == len(new_line)
            level.append(files_line)
        else:
            if level:
                max_len = max(map(len, level))
                level_np = np.zeros((len(level), max_len), dtype=int)
                for i, line in enumerate(level):
                    for j in range(len(line)):
                        level_np[i, j] = ord(line[j])
                    level_np[i, len(line) :] = ord("#")
                player_y, player_x = np.where((level_np == ord("@")) | (level_np == ord("+")))
                if not player_y:
                    raise ValueError(f"Could not find player in {level=}")
                level_mask = level_np == ord("#")

                flooded = skimage.segmentation.flood(level_mask, (int(player_y[0]), int(player_x[0])))
                level_np[~flooded] = ord("#")

                processed_content.extend(("".join(map(chr, line)) for line in level_np))
                level = []
            processed_content.append(files_line)

    return "\n".join(processed_content)


for fname in os.listdir():
    if fname.endswith(".txt") and not fname.startswith("proc."):
        with open(fname, "r") as f:
            with open(f"proc.{fname}", "w") as g:
                g.write(process_game_levels(f.read()))
