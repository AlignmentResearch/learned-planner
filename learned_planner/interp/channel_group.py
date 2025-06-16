from typing import Any, Tuple, Union, no_type_check

import pandas as pd

from learned_planner.interp.offset_fns import apply_inv_offset_lc

layer_groups: dict[str, list[dict[str, Any]]] = {
    "B up": [
        {
            "layer": 0,
            "idx": 13,
            "description": "H/C/I/J/O: +future box up moves [1sq up]. Activates negatively a wall square (since box can't be pushed up from there).",
            "type": "H",
            "sign": 1,
            "long-term": False,
            "i-sign": 1,
            "o-sign": 1,
        },
        {
            "layer": 0,
            "idx": 24,
            "description": "H/C/I/J/O: -future box up moves. long-term because it doesn't fade away after short-term also starts firing [1sq up,left]",
            "type": "H",
            "sign": -1,
            "long-term": True,
            "i-sign": 1,
            "o-sign": -1,
        },
        {
            "layer": 2,
            "idx": 6,
            "description": "H: +box-up-moves (~5-10 steps). -agent-up-moves. next-target (not always) [1q left]",
            "type": "H",
            "sign": 1,
            "long-term": False,
            "i-sign": 1,
            "o-sign": 1,
        },
    ],
    "B down": [
        {
            "layer": 0,
            "idx": 2,
            "description": "H/-C/-I/J/-O: +future box down moves [1sq left]",
            "type": "H",
            "sign": 1,
            "long-term": False,
            "i-sign": -1,
            "o-sign": -1,
        },
        {
            "layer": 0,
            "idx": 14,
            "description": "H/-I/O/C/H: -future-box-down-moves. Is more future-looking than other channels in this group. Box down moves fade away as other channels also start representing them. Sometimes also activates on -agent-right-moves [on sq]",
            "type": "H",
            "sign": -1,
            "long-term": True,
            "i-sign": 1,
            "o-sign": -1,
        },
        {
            "layer": 0,
            "idx": 20,
            "description": "H: box down moves. Upper right corner positively activates (0.47 start -> 0.6 in a few steps -> 0.7 very later on). I: -box down moves. O: +box down moves -box horizontal moves. [1sq up]",
            "type": "H",
            "sign": -1,
            "long-term": True,
            "i-sign": -1,
            "o-sign": 1,
        },
        {
            "layer": 1,
            "idx": 14,
            "description": "H: longer-term down moves? [1sq up]",
            "type": "H",
            "sign": -1,
            "long-term": True,
            "i-sign": -1,
            "o-sign": 1,
        },
        {
            "layer": 1,
            "idx": 17,
            "description": "H/C/I/-J/-F/O: -box-future down moves [on sq]",
            "type": "H",
            "sign": -1,
            "long-term": False,
            "i-sign": -1,
            "o-sign": 1,
            "representative": True,
        },
        {
            "layer": 1,
            "idx": 19,
            "description": "H/-F/-J: -box-down-moves (follower?) [1sq up]",
            "type": "H",
            "sign": -1,
            "long-term": False,
            "i-sign": -1,
            "o-sign": 1,
        },
    ],
    "B left": [
        {
            "layer": 0,
            "idx": 23,
            "description": "H/C/I/J/O: box future left moves [1sq up,left]",
            "type": "H",
            "sign": 1,
            "long-term": True,
            "i-sign": 1,
            "o-sign": 1,
        },
        {
            "layer": 0,
            "idx": 31,
            "description": "",
            "type": "H",
            "sign": 1,
            "long-term": False,
            "i-sign": 1,
            "o-sign": 1,
        },
        {
            "layer": 1,
            "idx": 11,
            "description": "-box-left-moves (-0.6).",
            "type": "H",
            "sign": -1,
            "long-term": False,
            "i-sign": -1,
            "o-sign": 1,
        },
        {
            "layer": 1,
            "idx": 27,
            "description": "H: box future left moves [1sq left]",
            "type": "H",
            "sign": -1,
            "long-term": True,
            "i-sign": -1,
            "o-sign": 1,
        },
        {
            "layer": 2,
            "idx": 20,
            "description": "H: -box future left moves [1sq left]",
            "type": "H",
            "sign": -1,
            "long-term": False,
            "i-sign": -1,
            "o-sign": 1,
        },
    ],
    "B right": [
        {
            "layer": 0,
            "idx": 9,
            "description": "-H/-C/-O/I/J/F: +agent +future box right moves -box. -H/J/F: +agent-near-future-down-moves [on sq]",
            "type": "H",
            "sign": -1,
            "long-term": False,
            "i-sign": 1,
            "o-sign": -1,
        },
        {
            "layer": 0,
            "idx": 17,
            "description": "H/I/J/F/O: +box-future-right moves. O: +agent [1sq up]",
            "type": "H",
            "sign": 1,
            "long-term": False,
            "i-sign": 1,
            "o-sign": 1,
        },
        {
            "layer": 1,
            "idx": 13,
            "description": "H: box-right-moves(+0.75),agent-future-pos(+0.02) [1sq left]",
            "type": "H",
            "sign": 1,
            "long-term": False,
            "i-sign": 1,
            "o-sign": 1,
        },
        {
            "layer": 1,
            "idx": 15,
            "description": "H/-O: box-right-moves-that-end-on-target (with high activations towards target). Activates highly when box is on the left side of target [on sq].",
            "type": "H",
            "sign": -1,
            "long-term": True,
            "i-sign": 1,
            "o-sign": 1,
        },
        {
            "layer": 2,
            "idx": 9,
            "description": "H/C/I/J/O: +future box right moves [1sq up]",
            "type": "H",
            "sign": 1,
            "long-term": True,
            "i-sign": 1,
            "o-sign": 1,
        },
        {
            "layer": 2,
            "idx": 15,
            "description": "-box-right-moves [1sq up,left]. O contains +box-down-moves. All squares become dark when the level becomes unsolvable (representing value?).",
            "type": "H",
            "sign": -1,
            "long-term": False,
            "i-sign": 1,
            "o-sign": -1,
        },
    ],
    "A up": [
        {
            "layer": 0,
            "idx": 18,
            "description": "H: -agent-exclusive-up-moves",
            "type": "H",
            "sign": -1,
            "long-term": False,
        },
        {
            "layer": 1,
            "idx": 5,
            "description": "H: +agent-exclusive-future-up moves [2sq up, 1sq left]",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
        {
            "layer": 1,
            "idx": 29,
            "description": "agent-near-future-up-moves(+0.5) (~5-10steps, includes box-up-pushes as well). I: future up moves (~almost all moves) + agent sq [1sq up]",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 28,
            "description": "near-future up moves (NFA). O: future up moves (not perfectly though) [1sq up]",
            "type": "H",
            "sign": 1,
            "i-sign": -1,
            "o-sign": -1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 29,
            "description": "Max-pooled Up action channel (MPA).",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
    ],
    "A down": [
        {
            "layer": 0,
            "idx": 10,
            "description": "H: -agent-exclusive-down-moves [1sq left,down]. Positively activates on agent-exclusive-up-moves.",  # sneaky
            "type": "H",
            "sign": -1,
            "long-term": False,
        },
        {
            "layer": 1,
            "idx": 18,
            "description": "H/-O: +agent future down moves (stores alternate down moves as well?) [on sq]",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 4,
            "description": "O: +near-future agent down moves (NFA). I: +agent/box future pos [1sq left]",
            "type": "H",
            "sign": 1,
            "i-sign": 1,
            "o-sign": 1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 8,
            "description": "down action (MPA).",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
    ],
    "A left": [
        {
            "layer": 2,
            "idx": 23,
            "description": "H: future left moves (does O store alternate left moves?) (NFA). [1sq left]",
            "type": "H",
            "sign": 1,
            "i-sign": -1,
            "o-sign": -1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 27,
            "description": "left action (MPA). T0: negative agent sq with positive sqs up/left.",
            "type": "H",
            "sign": -1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 31,
            "description": "some +agent-left-moves (includes box-left-pushes).",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
    ],
    "A right": [
        {
            "layer": 1,
            "idx": 21,
            "description": "H: agent-right-moves(-0.5) (includes box-right-pushes as well)",
            "type": "H",
            "sign": -1,
            "long-term": False,
        },
        {
            "layer": 1,
            "idx": 28,
            "description": "some-agent-exclusive-right-moves(+0.3),box-up-moves-sometimes-unclear(-0.1)",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 3,
            "description": "H: +right action (MPA) + future box -down -right moves + future box +left moves",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 5,
            "description": "H/C/I/J: +agent-future-right-incoming-sqs, O: agent-future-sqs [1sq up, left]",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 26,
            "description": "H/O: near-future right moves (NFA). [on sq]",
            "type": "H",
            "sign": 1,
            "i-sign": 1,
            "o-sign": 1,
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 21,
            "description": "H: -far-future-agent-right-moves. Negatively contributes to L2H26 to remove far-future-sqs. Also represents -agent/box-down-moves. [1sq up]",
            "type": "H",
            "sign": -1,
            "long-term": True,
        },
    ],
    "Misc plan": [
        {
            "layer": 0,
            "idx": 15,
            "description": "H/I/J/-F/-O: +box-future-moves. More specifically, +box-down-moves +box-left-moves. searchy (positive field around target). (0.42 corr across i,j,f,o).",
            "type": "H",
            "sign": 1,
            "long-term": False,
        },
        {
            "layer": 0,
            "idx": 16,
            "description": "H +box-right-moves (not all). High negative square when agent has to perform DRU actions. [1sq up,left]",
            "type": "H",
            "sign": 1,
        },
        {
            "layer": 0,
            "idx": 28,
            "description": "H/C/I/J/F/-O: -future box down moves (follower?) [on sq]. Also represents agent up,right,left directions (but not down).",
            "type": "H",
            "sign": -1,
        },
        {"layer": 0, "idx": 30, "description": "H/I: future positions (0.47 corr across i,j,f,o).", "type": "H"},
        {
            "layer": 1,
            "idx": 0,
            "description": "H: -agent -agent near-future-(d/l/r)-moves + box-future-pos [on sq]",
            "type": "H",
        },
        {
            "layer": 1,
            "idx": 4,
            "description": "+box-left moves -box-right moves [1sq up].",
            "type": "H",
        },
        {
            "layer": 1,
            "idx": 8,
            "description": "box-near-future-down-moves(-0.4),agent-down-moves(+0.3),box-near-future-up-moves(+0.25) [on sq]",
            "type": "H",
        },
        {
            "layer": 1,
            "idx": 9,
            "description": "O/I/H: future pos (mostly down?) (seems to have alternate paths as well. Ablation results in sligthly longer sols on some levels). Fence walls monotonically increase in activation across steps (tracking time).  [on sq]",
            "type": "H",
            "sign": 1,
            "long-term": True,  # for down moves
        },
        {
            "layer": 1,
            "idx": 20,
            "description": "+near-future-all-box-moves [1sq up].",
            "type": "H",
        },
        {
            "layer": 1,
            "idx": 25,
            "description": "all-possible-paths-leading-to-targets(-0.4),agent-near-future-pos(-0.07),walls-and-out-of-plan-sqs(+0.1),boxes(+0.6). H: +box -agent -empty -agent-future-pos | O/-C: -agent +future sqs (probably doing search in init steps) | I: box + agent + walls | F: -agent future pos | J: +box +wall -agent near-future pos [1sq up,left]",
            "type": "H",
        },
        {"layer": 2, "idx": 0, "description": "-box-all-moves.", "type": "H"},
        {"layer": 2, "idx": 1, "description": "H/O: future-down/right-sqs [1sq up]", "type": "H"},
        {
            "layer": 2,
            "idx": 13,
            "description": "H: +box-future-left -box-long-term-future-right(fades 5-10moves before taking right moves) moves. Sometimes blurry future box up/down moves [1sq up]",
            "type": "H",
            "sign": 1,  # - for right
            "long-term": False,
        },
        {
            "layer": 2,
            "idx": 14,
            "description": "H: all-other-sqs(-0.4) agent-future-pos(+0.01) O: -agent-future-pos. I: +box-future-pos",
            "type": "H",
        },
        {
            "layer": 2,
            "idx": 17,
            "description": "H/C: target(+0.75) box-future-pos(-0.3). O: target. J: +target -agent +agent future pos. I/F: target. [1sq up]",
            "type": "H",
        },
        {
            "layer": 2,
            "idx": 18,
            "description": "box-down/left-moves(-0.2). Very noisy/unclear at the start and converges later than other box-down channels.",
            "type": "H",
        },
        {"layer": 0, "idx": 7, "description": "(0.37 corr across i,j,f,o).", "type": "H"},
        {
            "layer": 0,
            "idx": 1,
            "description": "box-to-target-lines which light up when agent comes close to the box.",
            "type": "H",
        },
        {"layer": 0, "idx": 21, "description": "-box-left-moves. +up-box-moves", "type": "H"},
        {"layer": 1, "idx": 2, "description": "-box-left-moves", "type": "H", "sign": -1, "long-term": False},
        {"layer": 1, "idx": 23, "description": "-box-left-moves.", "type": "H", "sign": -1, "long-term": False},
        {
            "layer": 2,
            "idx": 11,
            "description": "-box-left-moves(-0.15),-box-right-moves(-0.05)",
            "type": "H",
            "sign": -1,
            "long-term": False,
        },
        {"layer": 2, "idx": 22, "description": "H: box-right-moves(+0.3),box-down-moves(0.15). O future sqs???", "type": "H"},
        {"layer": 2, "idx": 24, "description": "box-right/up-moves (long-term)", "type": "H"},
        {
            "layer": 2,
            "idx": 25,
            "description": "unclear but (8, 9) square tracks value or timesteps (it is a constant negative in the 1st half episode and steadily increases in the 2nd half)?",
            "type": "H",
        },
        #
        {"layer": 2, "idx": 12, "description": "", "type": "H"},
        {"layer": 2, "idx": 16, "description": "", "type": "H"},
        {"layer": 0, "idx": 19, "description": "", "type": "H"},
        {"layer": 2, "idx": 30, "description": "unclear", "type": "H"},
    ],
    "T": [
        {
            "layer": 0,
            "idx": 6,
            "description": "H/-C: +target -box -agent . F: +agent +agent future pos. I: +agent. O: -agent future pos. J: +target -agent[same sq]",
            "type": "H",
        },
        {
            "layer": 0,
            "idx": 26,
            "description": "H: -agent  . I/C/-O: all agent future positions. J/F: agent + target + BRwalls, [1sq up]",
            "type": "H",
        },
        {"layer": 1, "idx": 6, "description": "J: player (with fainted target)", "type": "H"},
        {
            "layer": 1,
            "idx": 10,
            "description": "J/H/C: -box + target +agent future pos. (neglible in H) O,-I: +agent +box -agent future pos [1sq up] (very important feature -- 18/20 levels changed after ablation)",
            "type": "H",
        },
        {"layer": 1, "idx": 22, "description": "-target", "type": "H"},
        {
            "layer": 1,
            "idx": 31,
            "description": "H: squares above and below target (mainly above) [1sq left & maybe up]",
            "type": "H",
        },
        {
            "layer": 2,
            "idx": 2,
            "description": "H: high activation when agent is below a box on target and similar positions. walls at the bottom also activate negatively in those positions.",
            "type": "H",
        },
        {"layer": 2, "idx": 7, "description": "+unsolved box/target", "type": "H"},
    ],
    "Other": [],
    "no-label": [
        {"layer": 0, "idx": 0, "description": "some box-left-moves?", "type": "H"},
        {"layer": 0, "idx": 3, "description": "", "type": "H"},
        {"layer": 0, "idx": 4, "description": "", "type": "H"},
        {"layer": 0, "idx": 5, "description": "[1sq left]", "type": "H"},
        {"layer": 0, "idx": 8, "description": "", "type": "H"},
        {"layer": 0, "idx": 22, "description": "", "type": "H"},
        {"layer": 0, "idx": 25, "description": "", "type": "H"},
        {"layer": 0, "idx": 27, "description": "", "type": "H"},
        {"layer": 0, "idx": 29, "description": "", "type": "H"},
        {"layer": 1, "idx": 1, "description": "", "type": "H"},
        {"layer": 1, "idx": 3, "description": "", "type": "H"},
        {"layer": 1, "idx": 12, "description": "", "type": "H"},
        {"layer": 1, "idx": 16, "description": "", "type": "H"},
        {"layer": 1, "idx": 26, "description": "", "type": "H"},
        {"layer": 1, "idx": 30, "description": "", "type": "H"},
        {"layer": 2, "idx": 10, "description": "", "type": "H"},
        {"layer": 2, "idx": 19, "description": "H: future agent down/right/left sqs (unclear) [1sq up]", "type": "H"},
        {
            "layer": 0,
            "idx": 11,
            "description": "H: CO. O: box-right moves C/I: -box future pos [1sq up (left-right noisy)]",
            "type": "H",
        },
        {
            "layer": 0,
            "idx": 12,
            "description": "H: very very faint horizontal moves (could be long-term?). I/O: future box horizontal moves (left/right). [on sq]",
            "type": "H",
        },
        {
            "layer": 1,
            "idx": 7,
            "description": "H: - some left box moves or right box moves (ones that end at a target)? Sometimes down moves? (unclear)",
            "type": "H",
            "sign": -1,
            "long-term": False,
        },
        {"layer": 1, "idx": 24, "description": "H: -box -agent-future-pos -agent, [1sq left]", "type": "H"},
    ],
}

layer_group_descriptions: dict[str, str] = {
    "B up": "Activates on squares from where a box would be pushed up",
    "B down": "Activates on squares from where a box would be pushed down",
    "B left": "Activates on squares from where a box would be pushed left",
    "B right": "Activates on squares from where a box would be pushed right",
    "A up": "Activates on squares from where an agent would move up",
    "A down": "Activates on squares from where an agent would move down",
    "A left": "Activates on squares from where an agent would move left",
    "A right": "Activates on squares from where an agent would move right",
    "Misc plan": "Channels that combine information from multiple directions",
    "T": "Highly activate on target tiles. Some also activate on agent or box tiles",
    "Other": "Channels that do not fit into any other category",
    "no-label": "Uninterpreted channels. These channels do not have a clear meaning but they are also not very useful",
    "non-plan": "Channels that do not contribute in representing the long-term plan",
    "nfa": "Channels that activate on squares that the agent will move in the next few moves. One separate channel for each direction",
    "mpa": "A channel for each action that activates highly across all squares at the last tick to predict the action",
}


def check_exhaustiveness(json_data):
    expected_layers = [0, 1, 2]
    expected_indices = set(range(32))  # Channels range from 0 to 31

    # Collect all(layer, idx) pairs from the JSON
    found_channels = set()
    for category, items in json_data.items():
        for item in items:
            if (item["layer"], item["idx"]) in found_channels:
                print(f"Duplicate channel: {item}")
            found_channels.add((item["layer"], item["idx"]))

        # Generate the full expected set
    missing_channels = []
    for layer in expected_layers:
        for idx in expected_indices:
            if (layer, idx) not in found_channels:
                missing_channels.append((layer, idx))

    if missing_channels:
        print("Missing channels:")
        for layer, idx in missing_channels:
            print(f"Layer {layer}, Index {idx}")
    else:
        print("All channels are accounted for!")


def get_group_channels(
    group_names: str,
    return_dict: bool = False,
    exclude_nfa_mpa: bool = False,
    long_term: bool = True,
) -> list[list[Union[dict["str", Any], Tuple[int, int]]]]:
    """Returns the channels for the given group names.

    Args:
        group_names (str): Comma-separated group names. Valid group names are:
            - "box": Box channels
            - "agent": Agent channels
            - "box_agent": Box and Agent channels combined by direction idx
            - "B up", "B down", "B left", "B right": Box channels by direction idx
            - "A up", "A down", "A left", "A right": Agent channels by direction idx
            - "nfa": NFA channels
            - "mpa": MPA channels
            - "nfa_mpa": NFA and MPA channels
            - "Misc plan", "T", "Other", "no-label": Other channels
            - "non-plan": All channels in "T", "Other", "no-label", "NFA", and "MPA"
        return_dict (bool, optional): Whether to return the channel dictionaries. Defaults to False.
        exclude_nfa_mpa (bool, optional): Whether to exclude NFA and MPA channels. Defaults to False.

    Returns:
        list[list[dict]] or list[list[tuple[int, int]]]: List of channels for each
            group. If return_dict is True, returns a list of dictionaries containing
            the channel information. Otherwise, returns a list of tuples containing
            the layer and index of the channel.
    """
    desired_groups = []
    group_name_list = group_names.split(",")
    assert ("box_and_agent" not in group_names) or len(
        group_name_list
    ) == 1, "box_and_agent cannot be combined with other groups"
    assert ("non-plan" not in group_names) or len(group_name_list) == 1, "non-plan cannot be combined with other groups"
    assert ("nfa_mpa" not in group_names) or len(group_name_list) == 1, "nfa_mpa cannot be combined with other groups"
    for group_name in group_name_list:
        if "box" in group_name:
            desired_groups += ["B up", "B down", "B left", "B right"]
        if "agent" in group_name or "nfa" in group_name or "mpa" in group_name:
            desired_groups += ["A up", "A down", "A left", "A right"]
        if group_name == "plan":
            desired_groups += ["B up", "B down", "B left", "B right", "A up", "A down", "A left", "A right"]
            desired_groups += ["Misc plan"]
            exclude_nfa_mpa = False
        if group_name == "non-plan":
            desired_groups += ["T", "Other", "no-label"]
        if group_name in layer_groups:
            desired_groups.append(group_name)
    assert desired_groups, f"Invalid group names: {group_names}. Choose from: box/agent/box_agent/nfa/mpa/Misc plan/non-plan."

    if group_names == "nfa_mpa" or group_names == "nfa" or group_names == "mpa":
        exclude_nfa_mpa = False

    group_channels = []
    for group in desired_groups:
        group_channels.append([])
        for lc in layer_groups[group]:
            if exclude_nfa_mpa and ("NFA" in lc["description"] or "MPA" in lc["description"]):
                continue
            if "long-term" in lc and not long_term and lc["long-term"]:
                continue

            group_channels[-1].append(lc)

    if "nfa" in group_names or "mpa" in group_names:

        def is_nfa_mpa(lc):
            if "nfa" in group_names and "NFA" in lc["description"]:
                return True
            if "mpa" in group_names and "MPA" in lc["description"]:
                return True
            return False

        group_channels = [[lc for lc in group if is_nfa_mpa(lc)] for group in group_channels]

    if "box_agent" in group_names:
        group_channels = [g + group_channels[i + 4] for i, g in enumerate(group_channels) if i < 4]
    if "non-plan" in group_names and not exclude_nfa_mpa:
        group_channels += get_group_channels("nfa_mpa", return_dict=True)
        group_channels = [c for g in group_channels for c in g]
        group_channels = [group_channels]

    if return_dict:
        return group_channels
    group_channels = [[(lc["layer"], lc["idx"]) for lc in group] for group in group_channels]
    return group_channels  # type: ignore


def split_by_layer(group_channels: list[list[dict | tuple[int, int]]]) -> list[list[dict | tuple[int, int]]]:
    """Splits the group channels by layer.

    Args:
        group_channels (list[list[Union[dict, Tuple[int, int]]]): List of channels for each group.

    Returns:
        list[list[Union[dict, int]]]: List of channels for each layer
    """
    split_channels = [[] for _ in range(3)]
    for channels in group_channels:
        for lc in channels:
            if isinstance(lc, dict):
                split_channels[lc["layer"]].append(lc)
            elif isinstance(lc, tuple) and len(lc) == 2:
                split_channels[lc[0]].append(lc[1])
            else:
                raise ValueError("Invalid channel obj type (should be dict or tuple of size 2): ", lc)

    return split_channels


def is_connected(lc1, lc2):
    """Checks if lc1 is fed as input to lc2"""
    l1, c1 = lc1
    l2, c2 = lc2

    return l1 == l2 or ((l1 + 1) % 3 == l2)


def get_group_connections(group_channels):
    group_connections = [[[] for _ in range(len(group_channels))] for _ in range(len(group_channels))]
    for i, channels1 in enumerate(group_channels):
        for lc1 in channels1:
            for j, channels2 in enumerate(group_channels):
                for lc2 in channels2:
                    if is_connected(lc1, lc2):
                        group_connections[i][j].append((lc1, lc2))
    return group_connections


def get_channel_dict(layer: int, channel: int) -> dict[str, Any]:
    """Returns the channel dictionary for the given layer and channel.

    Args:
        layer (int): Layer of the channel
        channel (int): Channel index

    Returns:
        dict[str, Any]: Channel dictionary
    """
    for group in layer_groups.values():
        for lc in group:
            if lc["layer"] == layer and lc["idx"] == channel:
                return lc
    raise ValueError(f"Channel L{layer}H{channel} not found")


def get_channel_sign(layer: int, channel: int, gate: str = "h") -> int:
    """Returns the sign of the channel.

    Args:
        layer (int): Layer of the channel
        channel (int): Channel index
        gate (str): Gate type. Defaults to "h". Choices: "h", "c", "i", "j", "f", "o".

    Returns:
        int: Sign of the channel
    """
    c_dict = get_channel_dict(layer, channel)
    gate = gate.lower()
    if gate == "h":
        if "sign" not in c_dict:
            raise ValueError(f"Sign not found for channel L{layer}H{channel}")
        return c_dict["sign"]
    if f"{gate}-sign" not in c_dict:
        raise ValueError(f"{gate}-sign not found for channel L{layer}H{channel}")
    return c_dict[f"{gate}-sign"]


def standardize_channel(channel_value, channel_info: tuple[int, int] | dict):
    """Standardize the channel value based on its sign and index."""
    assert len(channel_value.shape) >= 2, f"Invalid channel value shape: {channel_value.shape}"
    if isinstance(channel_info, tuple):
        l, c = channel_info
        channel_dict = get_channel_dict(l, c)
    else:
        channel_dict = channel_info
    channel_value = apply_inv_offset_lc(channel_value, channel_dict["layer"], channel_dict["idx"], last_dim_grid=True)
    sign = channel_dict.get("sign", 1)
    if isinstance(sign, str):
        assert sign in ["+", "-"], f"Invalid sign: {sign}"
        sign = 1 if sign == "+" else -1
    elif not isinstance(sign, int):
        raise ValueError(f"Invalid sign type: {type(sign)}")
    return channel_value * sign


@no_type_check
def all_channel_latex_table():
    """Prints a longtable latex table with layer, idx, long-term, and description for all channels"""
    # use pd.DataFrame to create a table
    df = pd.DataFrame(
        [
            {
                "Layer": lc["layer"],
                "Channel": lc["idx"],
                "Long-term": lc.get("long-term", False),
                "Description": lc["description"],
            }
            for group in layer_groups.values()
            for lc in group
        ]
    )
    # sort by layer and channel
    df = df.sort_values(["Layer", "Channel"])
    # convert boolean to latex
    df["Long-term"] = df["Long-term"].apply(lambda x: "Yes" if x else "No")
    # drop rows with empty or "unclear" description
    df = df[df["Description"].str.strip() != ""]
    df = df[df["Description"].str.strip() != "unclear"]
    df["Channel"] = df.apply(lambda x: f"L{x['Layer']}H{x['Channel']}", axis=1)
    df = df.drop(columns=["Layer"])
    # convert to latex
    table = df.to_latex(
        index=False,
        longtable=True,
        caption="Description of all channels",
        label="tab:all_channels",
        escape=True,
        column_format="p{0.08\\linewidth}p{0.08\\linewidth}p{0.77\\linewidth}",
    )
    print(table)


@no_type_check
def grouped_channel_latex_table():
    """Prints a latex table with group name, group description, and channels for each group"""
    # use pd.DataFrame to create a table
    group_names = ["box", "agent", "Misc plan", "T", "no-label", "nfa", "mpa"]
    pretty_group_names = ["Box", "Agent", "Combined Plan", "Target", "no-label", "NFA", "MPA"]
    udlr = ["up", "down", "left", "right"]
    group_info = []
    for name_idx, group_name in enumerate(group_names):
        group_channels = get_group_channels(group_name, return_dict=True)
        group_description = layer_group_descriptions.get(group_name, "")
        if "nfa" in group_name or "mpa" in group_name:
            channels = [(lc_dict[0]["layer"], lc_dict[0]["idx"]) for lc_dict in group_channels]
            group_info.append(
                {
                    "Group": pretty_group_names[name_idx],
                    "Description": group_description,
                    "Channels": ", ".join([f"L{lc[0]}H{lc[1]} ({udlr[i]})" for i, lc in enumerate(channels)]),
                }
            )
        elif len(group_channels) == 4:
            for i, direction in enumerate(udlr):
                group_description = layer_group_descriptions.get(group_name[0].upper() + " " + direction, group_description)
                channels = [(lc_dict["layer"], lc_dict["idx"], lc_dict["long-term"]) for lc_dict in group_channels[i]]
                group_info.append(
                    {
                        "Group": f"{pretty_group_names[name_idx]} {direction}",
                        "Description": group_description,
                        "Channels": ", ".join([f"L{lc[0]}H{lc[1]}{'*' if lc[2] else ''}" for lc in sorted(channels)]),
                    }
                )
        elif len(group_channels) == 1:
            group_info.append(
                {
                    "Group": pretty_group_names[name_idx],
                    "Description": group_description,
                    "Channels": ", ".join([f"L{lc['layer']}H{lc['idx']}" for lc in group_channels[0]]),
                }
            )
        else:
            raise ValueError(f"Invalid number of channels for group {group_name}: {len(group_channels)}")

    df = pd.DataFrame(group_info)
    # convert to latex
    table = df.to_latex(
        index=False,
        longtable=True,
        caption="Grouped channels and their descriptions. * indicates long-term channels.",
        label="tab:grouped-channels",
        escape=True,
        column_format="p{0.12\\linewidth}p{0.45\\linewidth}p{0.36\\linewidth}",
    )
    print(table)


if __name__ == "__main__":
    check_exhaustiveness(layer_groups)
    nfa_channels = get_group_channels("nfa", return_dict=True)
    print("NFA channels:", [f"L{lc[0]['layer']}H{lc[0]['idx']}" for lc in nfa_channels])  # type: ignore
    all_channel_latex_table()

    print("\n\n" + "=" * 50 + "\n\n")
    grouped_channel_latex_table()
