import pickle
import copy
import numpy as np
from resource import *
from random import random, randint, shuffle, sample
from math import factorial, e
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import math
import ford_fulkerson as FF
import sys
# from preflibtools.instances import generate_mallows_mix
import preflib_io as io
from serialDictatorship import serialDictatorship
from BlockingPairs import blockingPairs
import shutil
import csv

sys.setrecursionlimit(10000000)


# List nodes
class ListNode:
    def __init__(self, val=None, next=None, prev=None) -> None:
        self.val = val
        self.next = next


class DoubleListNode:
    def __init__(self, val=None, next=None, prev=None) -> None:
        self.val = val
        self.next = next
        self.prev = prev


class DoubleListNodeRegret:
    def __init__(self, val=None, regret=0, next=None, prev=None) -> None:
        self.val = val
        self.regret = regret
        self.next = next
        self.prev = prev


def populate_prefs(instance):
    men_prefs_list = []
    women_prefs_list = []
    men_prefs_dict = {}
    women_prefs_dict = {}
    men_prefs = instance[4].split('\n')
    women_prefs = instance[5].split('\n')
    n = int(instance[1])
    for i in range(n):
        row, prev = None, None
        men_prefs_dict[i] = {}
        man_list = men_prefs[i].split(',')
        for j, ind in enumerate(man_list):
            partner = int(ind)-1
            men_prefs_dict[i][partner] = j
            if not prev:
                row = prev = ListNode(partner)
            else:
                node = ListNode(partner)
                prev.next = node
                prev = prev.next
        men_prefs_list.append(row)
        row, prev = None, None
        women_prefs_dict[i] = {}
        woman_list = women_prefs[i].split(',')
        for j, ind in enumerate(woman_list):
            partner = int(ind)-1
            women_prefs_dict[i][partner] = j
            if not prev:
                row = prev = ListNode(partner)
            else:
                node = ListNode(partner)
                prev.next = node
                prev = prev.next
        women_prefs_list.append(row)

    return men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict


def print_preferences(men_prefs_list, women_prefs_list):
    print("Preferences for men: ")
    for i in range(len(men_prefs_list)):
        print(f"For man {i}: ", end=" ")
        row = copy.deepcopy(men_prefs_list[i])
        while row:
            print(row.val, end=" ")
            row = row.next
        print()
    print()
    print("Preferences for women: ")
    for i in range(len(women_prefs_list)):
        print(f"For woman {i}: ", end=" ")
        row = copy.deepcopy(women_prefs_list[i])
        while row:
            print(row.val, end=" ")
            row = row.next
        print()


def preferences_shifted(
    men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict
):
    men_prefs_list_shifted = [None]
    women_prefs_list_shifted = [None]
    men_prefs_dict_shifted = {}
    women_prefs_dict_shifted = {}
    for i in range(len(men_prefs_dict)):
        men_prefs_dict_shifted[i + 1] = {}
        women_prefs_dict_shifted[i + 1] = {}
        for j in men_prefs_dict[i]:
            men_prefs_dict_shifted[i + 1][j + 1] = men_prefs_dict[i][j]
            women_prefs_dict_shifted[i + 1][j + 1] = women_prefs_dict[i][j]
    for i in range(len(men_prefs_list)):
        row, prev = None, None
        pref = men_prefs_list[i]
        while pref:
            if not prev:
                row = prev = DoubleListNode(pref.val + 1)
            else:
                node = DoubleListNode(pref.val + 1)
                prev.next = node
                node.prev = prev
                prev = prev.next
            pref = pref.next
        men_prefs_list_shifted.append(row)
    for i in range(len(women_prefs_dict)):
        row, prev = None, None
        pref = women_prefs_list[i]
        while pref:
            if not prev:
                row = prev = DoubleListNode(pref.val + 1)
            else:
                node = DoubleListNode(pref.val + 1)
                prev.next = node
                node.prev = prev
                prev = prev.next
            pref = pref.next
        women_prefs_list_shifted.append(row)
    return (
        men_prefs_list_shifted,
        women_prefs_list_shifted,
        men_prefs_dict_shifted,
        women_prefs_dict_shifted,
    )


# Vanilla Deferred Acceptance
def gale_shapley(
    men_prefs_list,
    women_prefs_list,
    men_prefs_dict,
    women_prefs_dict,
    direction_change=False,
    verbose=False
):
    if direction_change:
        man_text = "Woman"
        woman_text = "Man"
        man_letter = "W"
        woman_letter = "M"
    else:
        man_text = "Man"
        woman_text = "Woman"
        man_letter = "M"
        woman_letter = "W"

    comments = []

    male_side = {i: None for i in range(len(men_prefs_list))}
    female_side = {i: None for i in range(len(women_prefs_list))}
    proposer = 0
    while proposer < len(men_prefs_list):
        if male_side[proposer] is not None or not men_prefs_list[proposer]:
            proposer += 1
            continue
        candidate = men_prefs_list[proposer].val
        # print(f"{man_text} {proposer} proposes to {woman_text} {candidate}.")
        comments.append(f"M{proposer+1} is free. M{proposer+1} proposes to W{candidate+1}")
        men_prefs_list[proposer] = men_prefs_list[proposer].next
        if female_side[candidate] is None:
            male_side[proposer] = candidate
            female_side[candidate] = proposer
            # print(f"{man_text} {proposer} gets engaged to {woman_text} {candidate} who was free before this.")
            comments.append(f"Since W{candidate+1} is free, W{candidate+1} accepts the proposal. Now M{proposer+1} and W{candidate+1} are matched.")
            # print(f"{man_letter}{proposer} --> {woman_letter}{candidate} (free) | Accepted")
            proposer += 1
        else:
            other = female_side[candidate]
            # print(f"{man_text} {proposer} proposes to {woman_text} {candidate} who is currently paired with {man_text} {other}.")
            if (
                women_prefs_dict[candidate][proposer]
                > women_prefs_dict[candidate][other]
            ):
                comments.append(f"Since W{candidate+1} prefers their current partner M{other+1} to M{proposer+1}, W{candidate+1} rejects the proposal. M{other+1} and W{candidate+1} are still matched, and M{proposer+1} is still free.")
                # print(f"{woman_text} {candidate} prefers their old partner, {man_text} {other} to {man_text} {proposer}.")
                # print(f"{man_letter}{proposer} --> {woman_letter}{candidate} ({man_letter}{other}) | Rejected")
                # print("", end="")
            else:
                female_side[candidate] = proposer
                male_side[proposer] = candidate
                male_side[other] = None
                comments.append(f"Since W{candidate+1} prefers M{proposer+1} to their current partner M{other+1}, W{candidate+1} accepts the proposal. Now M{proposer+1} and W{candidate+1} are matched, and M{other+1} is free.")
                # print(f"{woman_text} {candidate} prefers {man_text} {proposer} to their old partner, {man_text} {other}. The switch is made.")
                # print(f"{man_letter}{proposer} --> {woman_letter}{candidate} ({man_letter}{other}) | Accepted")
                proposer = other
    if verbose:
        return comments
    return male_side, female_side, men_prefs_list, women_prefs_list

# Egalitarian Matching
def update_shortlist(
    man,
    woman,
    woman_old,
    men_prefs_list,
    men_prefs_dict,
    women_prefs_list,
    node_pointers_men,
    node_pointers_women,
    male_side,
    female_side,
):
    elim_pairs = {
        't1': [],
        't2': []
    }

    for key in range(1, len(men_prefs_dict)+1):
        if men_prefs_dict[man][woman] >  men_prefs_dict[man][key] and  men_prefs_dict[man][key] > men_prefs_dict[man][woman_old]:
            elim_pairs['t1'].append((man, key))

    while men_prefs_list[man].val != woman:
        stale = men_prefs_list[man]
        men_prefs_list[man] = men_prefs_list[man].next
        # print(man, stale.val)
        # if stale.val != male_side[man]:
        #     elim_pairs['t1'].append((man, stale.val))
        men_prefs_list[man].prev = None
        stale.next = None
        left_man = node_pointers_women[stale.val][man].prev
        right_man = node_pointers_women[stale.val][man].next
        if left_man and right_man:
            left_man.next = right_man
            right_man.prev = left_man
        elif right_man:
            right_man.prev = left_man
            women_prefs_list[stale.val] = right_man
        elif left_man:
            left_man.next = right_man
    # print(f"Man {man} and woman {woman} to be made partners")
    # print(
    #     f"Old partner of man {man} is woman {male_side[man]} and old partner of woman {woman} is {female_side[woman]}"
    # )
    if female_side[woman]:
        m = node_pointers_women[woman][female_side[woman]]
    # Condition only for Gale-Shapley
    else:
        # print("Update shrotlist for GS")
        m = women_prefs_list[woman]
        while m.next:
            m = m.next
    while m.val != man:
        stale = m
        m = m.prev
        if stale.val != female_side[woman]:
            elim_pairs['t2'].append((stale.val, woman))
        m.next = None
        stale.prev = None
        left_woman = node_pointers_men[stale.val][woman].prev
        right_woman = node_pointers_men[stale.val][woman].next
        if left_woman and right_woman:
            left_woman.next = right_woman
            right_woman.prev = left_woman
        elif right_woman:
            right_woman.prev = left_woman
            men_prefs_list[stale.val] = right_woman
        elif left_woman:
            left_woman.next = right_woman
    # print(elim_pairs)
    return elim_pairs


def gale_shapley_shortlist(
    men_prefs_list,
    women_prefs_list,
    men_prefs_dict,
    women_prefs_dict,
    node_pointers_men,
    node_pointers_women,
    direction_change=False,
    verbose=False,
):
    if direction_change:
        man_text = "Woman"
        woman_text = "Man"
        man_letter = "W"
        woman_letter = "M"
    else:
        man_text = "Man"
        woman_text = "Woman"
        man_letter = "M"
        woman_letter = "W"

    male_side = {i: None for i in range(1, len(men_prefs_list))}
    female_side = {i: None for i in range(1, len(women_prefs_list))}
    proposer = 1
    while proposer < len(men_prefs_list):
        if male_side[proposer] is not None or not men_prefs_list[proposer]:
            proposer += 1
            continue
        candidate = men_prefs_list[proposer].val
        if verbose: 
            print(f"Man {proposer} proposes to woman {candidate}")
        if candidate not in men_prefs_dict[proposer]:
            continue
        update_shortlist(
            proposer,
            candidate,
            candidate,
            men_prefs_list,
            men_prefs_dict,
            women_prefs_list,
            node_pointers_men,
            node_pointers_women,
            male_side,
            female_side,
        )
        if female_side[candidate] is None:
            male_side[proposer] = candidate
            female_side[candidate] = proposer
            if verbose:
                print(f"{man_text} {proposer} gets engaged to {woman_text} {candidate} who was free before this.")
                print(f"{man_letter}{proposer} --> {woman_letter}{candidate} (free) | Accepted")
            proposer += 1
        else:
            other = female_side[candidate]
            if verbose: print(f"{man_text} {proposer} proposes to {woman_text} {candidate} who is currently paired with {man_text} {other}.")
            if (
                women_prefs_dict[candidate][proposer]
                > women_prefs_dict[candidate][other]
            ):
                if verbose:
                    print(f"{woman_text} {candidate} prefers their old partner, {man_text} {other} to {man_text} {proposer}.")
                    print(f"{man_letter}{proposer} --> {woman_letter}{candidate} ({man_letter}{other}) | Rejected")
                passed = men_prefs_list[proposer]
                men_prefs_list[proposer] = men_prefs_list[proposer].next
                passed.next = None
                men_prefs_list[proposer].prev = None
            else:
                female_side[candidate] = proposer
                male_side[proposer] = candidate
                male_side[other] = None
                if verbose:
                    print(f"{woman_text} {candidate} prefers {man_text} {proposer} to their old partner, {man_text} {other}. The switch is made.")
                    print(f"{man_letter}{proposer} --> {woman_letter}{candidate} ({man_letter}{other}) | Accepted")
                proposer = other
    if verbose: print_preferences(men_prefs_list, women_prefs_list)
    # print("\n\n\n\nGS COMPLETED FOR ONE SIDE\n\n\n\n")
    return male_side, female_side


def get_node_pointers(men_prefs_list, women_prefs_list):
    node_pointers_men = [
        [None] * (len(women_prefs_list) + 1) for j in range(len(men_prefs_list) + 1)
    ]
    node_pointers_women = [
        [None] * (len(men_prefs_list) + 1) for j in range(len(women_prefs_list) + 1)
    ]
    for i in range(len(men_prefs_list)):
        row = men_prefs_list[i]
        while row:
            j = row.val
            node_pointers_men[i][j] = row
            row = row.next
    for i in range(1, len(women_prefs_list)):
        row = women_prefs_list[i]
        while row:
            j = row.val
            node_pointers_women[i][j] = row
            row = row.next
    return node_pointers_men, node_pointers_women


def find_exposed_rotations(men_prefs_list, male_side, female_side):
    graph = nx.DiGraph()
    for man in male_side:
        if men_prefs_list[man].next:
            second = men_prefs_list[man].next.val
            graph.add_edge((man, male_side[man]), (female_side[second], second))
    return nx.simple_cycles(graph)


def plot_poset(poset, stage, rotation_levels):
    pos = nx.spring_layout(poset)
    left = -10
    hstep = 20 // stage if stage else 20
    for stage in rotation_levels:
        rotations = rotation_levels[stage]
        x = left + hstep * stage
        vstep = 10 // (len(rotations) + 1)
        for i, rotation in enumerate(rotations):
            pos[rotation] = [x, 10 - vstep * (i + 1)]
    nx.draw(
        poset,
        pos,
        node_size=500,
        alpha=0.9,
        labels={node: node for node in poset.nodes()},
    )
    plt.show()
    print(
        "_______________________________________________________________________________"
    )


def plot_flow_graph(flow, stage, rotation_levels):
    pos = nx.spring_layout(flow)
    left = -10
    hstep = 30 // stage if stage else 30
    pos["s"] = [-2, 10]
    pos["t"] = [2, 2]
    for stage in rotation_levels:
        rotations = rotation_levels[stage]
        x = left + hstep * stage
        vstep = 10 // (len(rotations) + 1)
        for i, rotation in enumerate(rotations):
            pos[rotation] = [x, 10 - vstep * (i + 1)]
    nx.draw(
        flow,
        pos,
        node_size=400,
        alpha=0.9,
        labels={node: node for node in flow.nodes()},
    )
    edge_lables = dict(
        [
            ((u, v), d["capacity"] if "capacity" in d else "inf")
            for u, v, d in flow.edges(data=True)
        ]
    )
    nx.draw_networkx_edge_labels(flow, pos, edge_labels=edge_lables)
    plt.show()
    print(
        "_______________________________________________________________________________"
    )


def truncate_preferences(
    men_prefs_list,
    women_prefs_list,
    men_prefs_dict,
    women_prefs_dict,
    node_pointers_men,
    node_pointers_women,
    regret,
):
    for man in range(1, len(men_prefs_list)):
        stale = men_prefs_list[man]
        while stale:
            # If the man in question is beyond the worst regret that the woman can have
            # or
            # If the woman is beyond the worst regret that the man in question can have
            if (
                women_prefs_dict[stale.val][man] > regret
                or men_prefs_dict[man][stale.val] > regret
            ):
                # Removing the woman from the man's list
                # print(f"Need to remove man {man} and woman {stale.val} from each other's lists.")
                left = stale.prev
                right = stale.next
                if left and right:
                    left.next = right
                    right.prev = left
                elif left:
                    left.next = right
                elif right:
                    right.prev = left
                    men_prefs_list[man] = right
                stale.prev = None
                stale.next = None
                rem_woman = stale.val
                stale = right
                # Removing the man from the woman's list
                extra = node_pointers_women[rem_woman][man]
                left = extra.prev
                right = extra.next
                if left and right:
                    left.next = right
                    right.prev = left
                elif left:
                    left.next = right
                elif right:
                    right.prev = left
                    women_prefs_list[rem_woman] = right
                extra.next = None
                extra.prev = None
            else:
                stale = stale.next


def egalitarian_matching_plain(preferences_file, regret, verbose=False):
    matchings = {
        "lattice": []
    }
    with open(preferences_file, "rb") as f:
        (
            men_prefs_list,
            women_prefs_list,
            men_prefs_dict,
            women_prefs_dict,
        ) = pickle.load(f)

    # Generating the matrices for the pointers to each doubly linked-list node
    node_pointers_men, node_pointers_women = get_node_pointers(
        men_prefs_list, women_prefs_list
    )

    if verbose:
        print_preferences(men_prefs_list, women_prefs_list)

    # Truncating the preferences based on the maximum regret allowed
    truncate_preferences(
        men_prefs_list,
        women_prefs_list,
        men_prefs_dict,
        women_prefs_dict,
        node_pointers_men,
        node_pointers_women,
        regret,
    )

    if verbose:
        print("These are supposed to be truncated prefs")
        print_preferences(men_prefs_list, women_prefs_list)
        print()

    # Running the DA algorithm, with women as proposers, given the truncated preferences to get the best solution for women given the regret constraint
    # print("Starting GS algo to get women optimal solution.")
    female_side_wopt, male_side_wopt = gale_shapley_shortlist(
        women_prefs_list,
        men_prefs_list,
        women_prefs_dict,
        men_prefs_dict,
        node_pointers_women,
        node_pointers_men,
        True,
        False
    )
    if verbose:
        print(f"\nWomen Optimal: {list(male_side_wopt.values())}\n")

    matchings['women_opt'] = copy.deepcopy(male_side_wopt)

    # Re-loading the preferences for generating the men-optimal solution as the preferences would be modified after running DA once
    with open(preferences_file, "rb") as f:
        (
            men_prefs_list,
            women_prefs_list,
            men_prefs_dict,
            women_prefs_dict,
        ) = pickle.load(f)

    # Generating node pointer matrices again
    node_pointers_men, node_pointers_women = get_node_pointers(
        men_prefs_list, women_prefs_list
    )

    # Truncating preferences again
    truncate_preferences(
        men_prefs_list,
        women_prefs_list,
        men_prefs_dict,
        women_prefs_dict,
        node_pointers_men,
        node_pointers_women,
        regret,
    )

    # Obtaining the men optimal solutions and updating the shortlists accordingly. These would be the starting point for identifying exposed rotations.
    # print("Starting GS algo to get men optimal solution.")
    male_side_mopt, female_side_mopt = gale_shapley_shortlist(
        men_prefs_list,
        women_prefs_list,
        men_prefs_dict,
        women_prefs_dict,
        node_pointers_men,
        node_pointers_women,
        False,
        False
    )
    if verbose:
        print(f"\nMen Optimal: {list(male_side_mopt.values())}\n")

    if verbose:
        print_preferences(men_prefs_list, women_prefs_list)
        print()

    matchings['men_opt'] = copy.deepcopy(male_side_mopt)

    male_side = male_side_mopt
    female_side = female_side_mopt
    rotations = {}
    counter = 0
    rotation_pairs = {}
    elim_pairs = {
        't1': {},
        't2': {}
    }
    rotation_levels = {}
    stage = 0

    # While the women-optimal matching is not reached
    while male_side != male_side_wopt:
        # Identifying the cycles that are exposed in the current matching
        cycles = list(
            find_exposed_rotations(men_prefs_list, male_side_mopt, female_side_mopt)
        )
        if verbose:
            print("Rotations exposed at the current stage:")
        rotation_levels[stage] = []
        # Iterating over the rotations
        for cycle in cycles:
            if verbose:
                print(cycle)
            counter += 1
            rotation_levels[stage].append(counter)
            rotations[counter] = {"cycle": cycle, "elim_pairs": {'t1': [], 't2': []}}
            weight = 0
            vector = [0]*(len(men_prefs_list)-1)
            # Iterating over each pair in the current rotation
            for i in range(len(cycle)):
                m1 = cycle[i][0]  # Man in the pair, m1
                w1 = cycle[i][1]  # Woman in the pair, w1
                mn = cycle[i - 1][0]  # New partner of w1, m(i-1)
                w2 = cycle[(i + 1) % len(cycle)][1]  # New partner of the man, w2
                rotation_pairs[
                    (cycle[i])
                ] = counter  # Assigning the rotation number to the pair

                # (m1, w2) will be a new pair; updating the shortlists accrordingly and marking the pairs that are not part of any rotation but are eliminated by this one.
                cycle_elims = update_shortlist(
                    m1,
                    w2,
                    w1,
                    men_prefs_list,
                    men_prefs_dict,
                    women_prefs_list,
                    node_pointers_men,
                    node_pointers_women,
                    male_side,
                    female_side,
                )
                rotations[counter]["elim_pairs"]['t1'] += cycle_elims['t1']
                rotations[counter]["elim_pairs"]['t2'] += cycle_elims['t2']
                if verbose: print(rotations[counter]['elim_pairs'])
                for pair in rotations[counter]["elim_pairs"]['t1']:
                    elim_pairs['t1'][pair] = counter
                for pair in rotations[counter]["elim_pairs"]['t2']:
                    elim_pairs['t2'][pair] = counter
                # Updating the weight of the current rotation
                weight += (
                    men_prefs_dict[m1][w1]
                    - men_prefs_dict[m1][w2]
                    + women_prefs_dict[w2][female_side[w2]]
                    - women_prefs_dict[w2][m1]
                )
                vector[men_prefs_dict[m1][w1]] -= 1
                vector[men_prefs_dict[m1][w2]] += 1
                vector[women_prefs_dict[w2][m1]] += 1
                vector[women_prefs_dict[w2][female_side[w2]]] -= 1

                male_side[m1] = w2
                female_side[w2] = m1
            if verbose:
                print(f"net weight of rotation: {weight}")
            rotations[counter]["weight"] = weight
            rotations[counter]["egal_lex"] = [weight] + [-v for v in vector][::-1]
            rotations[counter]["lex_egal"] = [-v for v in vector][::-1] + [weight]
            # print(rotations[counter]['vector'])
            # rotations[counter]["vector"] = vector
        stage += 1  # This will help us draw the graph by keeping track of the number of times we look for exposed rotations
        if verbose:
            print()
            print_preferences(men_prefs_list, women_prefs_list)
    # print(rotations)

    if verbose:
        with open(preferences_file, "rb") as f:
            (
                men_prefs_list_,
                women_prefs_list_,
                men_prefs_dict_,
                women_prefs_dict_,
            ) = pickle.load(f)
        print_preferences(men_prefs_list_, women_prefs_list_)
        for rot in rotations:
            print(rot, rotations[rot])

    # print(rotation_pairs)
    # print(elim_pairs)
    # print(rotation_levels)

    last_rotation = {
        man: None for man in range(1, len(men_prefs_list) + 1)
    }  # Dictionary to keep track of the previous rotation where a man was seen in a pair.

    poset = nx.DiGraph()  # DAG for relationship between rotations
    poset.add_nodes_from(range(1, counter + 1))

    # Application of Rule 1
    for pair in rotation_pairs:
        if last_rotation[pair[0]]:
            # Adding an edge from the last rotation where the man was seen to the current rotation which contains a pair with the man in it.
            poset.add_edge(last_rotation[pair[0]], rotation_pairs[pair])
            if verbose:
                print(
                    f"Previous rotation where man {pair[0]} was seen was rotation {last_rotation[pair[0]]}. The next rotation where his pair is seen is {rotation_pairs[pair]}"
                )
                # plot_poset(poset, stage, rotation_levels)
        last_rotation[pair[0]] = rotation_pairs[pair]

    if verbose:
        plot_poset(poset, stage, rotation_levels)

    # Re-loading the preference lists. This is needed for correctly adding edges to the relation graph based on eliminated pairs
    # and for arriving at the egalitarian matching by performing the rotations which are identified by the min-cut algorithm
    with open(preferences_file, "rb") as f:
        (
            men_prefs_list,
            women_prefs_list,
            men_prefs_dict,
            women_prefs_dict,
        ) = pickle.load(f)

    node_pointers_men, node_pointers_women = get_node_pointers(
        men_prefs_list, women_prefs_list
    )

    truncate_preferences(
        men_prefs_list,
        women_prefs_list,
        men_prefs_dict,
        women_prefs_dict,
        node_pointers_men,
        node_pointers_women,
        regret,
    )

    # Application of Rule 2
    # for pair in elim_pairs:
    #     m, w_ = pair # Eliminate pair (m, w')
    #     pi = elim_pairs[pair] # Rotation that eliminated the pair
    #     pointer = node_pointers_men[m][w_] # Position of w' in m's preference list
    #     # Finding the first woman (w) above w' in m's preference list such that (m, w) is part of some rotation.
    #     while pointer and (m, pointer.val) not in rotation_pairs:
    #         pointer = pointer.prev
    #         if pointer and (m, pointer.val) in rotation_pairs and rotation_pairs[(m, pointer.val)] != pi:
    #             rho = rotation_pairs[(m, pointer.val)] # Rotation containing (m, w)
    #             if verbose: print(f"The first woman above {w_} (w') in {m}'s (m's) preference order is {pointer.val}. \nThe pair ({m}, {pointer.val}) belongs to rotation {rho}. \nHence there has to be an edge from rotation {pi} to rotation {rho}.\n")
    #             # Adding an edge from pi to rho.
    #             if not poset.has_edge(rho, pi): poset.add_edge(pi, rho)
    #             if verbose: plot_poset(poset, stage, rotation_levels)

    # Cooper and Manlove's rule 2
    for pair1 in elim_pairs['t1']:
        for pair2 in elim_pairs['t2']:
            rho = elim_pairs["t1"][pair1]
            rho_ = elim_pairs["t2"][pair2]
            if pair1 == pair2 and rho != rho_:
                poset.add_edge(rho_, rho)
        

    # Creating the P'(s, t) flow graph
    
    flow = nx.DiGraph()
    flow.add_nodes_from(["s", "t"] + list(range(1, counter + 1)))

    for rotation in rotations:
        # If the weight of the rotation is -ve, add an edge from 's' to it.
        if rotations[rotation]["weight"] < 0:
            flow.add_edge("s", rotation, capacity=abs(rotations[rotation]["weight"]))
        # If the weight of the rotation is +ve, add an edge from it to 't'.
        elif rotations[rotation]["weight"] > 0:
            flow.add_edge(rotation, "t", capacity=abs(rotations[rotation]["weight"]))

    flow.add_edges_from(list(poset.edges()))

    if verbose:
        plot_flow_graph(flow, stage, rotation_levels)

    # Obtain the minimum cut on the flow graph and obtraining the cutset
    cut_value, partition = nx.minimum_cut(flow, "s", "t")
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, flow[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    if verbose:
        print(reachable, non_reachable)
        print(cutset)

    egalitarianRotations = []
    # Iterate over all rotations.
    for rotation in rotations:
        weight = rotations[rotation]["weight"]
        # If the rotation is not part of the nodes whose edges are in the cutset and if the weight of the rotation is +ve, include it in the closed subset.
        if weight > 0 and ((rotation, "t") not in cutset):
            # Add all the rotations that precede this rotation into the closed subset
            for pred in list(nx.edge_dfs(flow, rotation, orientation="reverse")):
                # print(pred)
                if type(pred[0]) != str and pred[0] not in egalitarianRotations:
                    egalitarianRotations.append(pred[0])
            if rotation not in egalitarianRotations:
                egalitarianRotations.append(rotation)
    egalitarianRotations.sort()
    if verbose:
        print(egalitarianRotations)

    # Reaching the men optimal solution again.
    if verbose:
        print("Starting GS algo to get men optimal solution.")
    male_side, female_side = gale_shapley_shortlist(
        men_prefs_list,
        women_prefs_list,
        men_prefs_dict,
        women_prefs_dict,
        node_pointers_men,
        node_pointers_women,
    )
    if verbose:
        print(f"\nMen Optimal: {list(male_side.values())}\n")

    # Apply all rotations in the closed subset obtained in order to reach the egalitarian stable matching.
    for rotation in egalitarianRotations:
        cycle = rotations[rotation]["cycle"]
        if verbose:
            print(f"Applying rotation {rotation}: {cycle}")
        for i in range(len(cycle)):
            m1 = cycle[i][0]
            w1 = cycle[i][1]
            mn = cycle[i - 1][0]
            w2 = cycle[(i + 1) % len(cycle)][1]

            # (m1, w2) will be a new pair
            _ = update_shortlist(
                m1,
                w2,
                w1,
                men_prefs_list,
                men_prefs_dict,
                women_prefs_list,
                node_pointers_men,
                node_pointers_women,
                male_side,
                female_side,
            )
            male_side[m1] = w2
            female_side[w2] = m1
        if male_side != male_side_wopt:
            matchings["lattice"].append(copy.deepcopy(male_side))
        if verbose:
            print()

    # Calculate the egalitarian cost (sum of all regrets)
    egal_cost = 0
    for man in male_side:
        egal_cost += (
            men_prefs_dict[man][male_side[man]] + women_prefs_dict[male_side[man]][man]
        )

    return male_side, female_side, egal_cost, matchings


def format_matching(matching_dict):
    if not matching_dict:
        return ''
    matching_list = "["
    for man in range(1, len(matching_dict)+1):
        matching_list += f"[M{man}, W{matching_dict[man]}],"
    matching_list += "]"
    return matching_list

# Testing
def compute_matchings(raw_file, num_instances = 20, size_max = 50, delete_files=True):

    instances = pd.read_csv(raw_file)
    new_dict = {col: [] for col in instances.columns}
    new_dict['men_opt'] = []
    new_dict['women_opt'] = []
    new_dict['lattice'] = []
    
    if not os.path.exists('../instances_matchings/'):
        os.mkdir('../instances_matchings/')

    # print(raw_file)

    size = int(raw_file.split('/')[2].split('_')[0][1:])
    culture = raw_file.split('/')[2].split('_')[1]

    out_file_name = f'../instances_matchings/{size}_{culture}_processed.csv'
    if os.path.exists(out_file_name):
        print(f'Matchings data already exists for {raw_file}!')
        return True

    if size > size_max:
        return False

    new_dict[f'random_1'] = []
    new_dict[f'random_{round(math.sqrt(size))}'] = []
    new_dict[f'random_{size}'] = []
    new_dict["random"] = []

    # new_dict['level1_q'] = []
    # new_dict['level1_a'] = []
    # new_dict['level2_q'] = []
    # new_dict['level2_a'] = []
    # new_dict['level2n_q'] = []
    # new_dict['level2n_a'] = []


    if not os.path.exists(f'../instances_matchings/{size}/'):
        os.mkdir(f'../instances_matchings/{size}/')
        
    for i, instance in tqdm(enumerate(instances.values)):
        # print(instance)
        men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict =  populate_prefs(instance) 

        preference_file_0 = f'../instances_matchings/{size}/prefs_{i}_0'
        preference_file_1 = f'../instances_matchings/{size}/prefs_{i}_1'

        # print_preferences(men_prefs_list, women_prefs_list)

        with open(preference_file_0, "wb") as f:
            pickle.dump(
                [men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict], f
            )

        (
            men_prefs_list_shifted,
            women_prefs_list_shifted,
            men_prefs_dict_shifted,
            women_prefs_dict_shifted,
        ) = preferences_shifted(
            men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict
        )

        with open(preference_file_1, "wb") as f:
            pickle.dump(
                [
                    men_prefs_list_shifted,
                    women_prefs_list_shifted,
                    men_prefs_dict_shifted,
                    women_prefs_dict_shifted,
                ],
                f,
            )

        male_side_egal, female_side_egal, egal_cost_re, matchings = egalitarian_matching_plain(
            preference_file_1, len(men_prefs_list)+1, False
        )

        if culture == "womanmaster" or matchings['lattice']:
            
            ntries = 1000
            candidate = [matchings['men_opt'][man] for man in matchings['men_opt']]
            while blockingPairs(size, instance[4], instance[5], candidate, "weak")['blockingPairCount'] != 1 and ntries:
                candidate = [matchings['men_opt'][man] for man in matchings['men_opt']]
                pos1 = randint(0,size-2)
                pos2 = randint(pos1+1, size-1)
                candidate = candidate[:pos1] + [candidate[pos2]] + candidate[pos1+1:pos2] + [candidate[pos1]] + candidate[pos2+1:]
                ntries -= 1
            random_1 = copy.deepcopy(candidate)
            
            num_blocking_1 = blockingPairs(size, instance[4], instance[5], candidate, "weak")['blockingPairCount']
            # print("R1", num_blocking_1)

            ntries = 1000
            while True and ntries:
                candidate = [matchings['men_opt'][man] for man in matchings['men_opt']]
                while blockingPairs(size, instance[4], instance[5], candidate, "weak")['blockingPairCount'] < round(math.sqrt(size)):
                    pos1 = randint(0,size-2)
                    pos2 = randint(pos1+1, size-1)
                    candidate = candidate[:pos1] + [candidate[pos2]] + candidate[pos1+1:pos2] + [candidate[pos1]] + candidate[pos2+1:]
                
                if blockingPairs(size, instance[4], instance[5], candidate, "weak")['blockingPairCount'] == round(math.sqrt(size)):
                    break
                ntries -= 1
            random_sqrt = copy.deepcopy(candidate)

            num_blocking_2 = blockingPairs(size, instance[4], instance[5], candidate, "weak")['blockingPairCount']
            # print("Rsqrt", num_blocking_2)

            ntries = 1000
            while True and ntries:
                candidate = [matchings['men_opt'][man] for man in matchings['men_opt']]
                while blockingPairs(size, instance[4], instance[5], candidate, "weak")['blockingPairCount'] < size:
                    pos1 = randint(0,size-2)
                    pos2 = randint(pos1+1, size-1)
                    candidate = candidate[:pos1] + [candidate[pos2]] + candidate[pos1+1:pos2] + [candidate[pos1]] + candidate[pos2+1:]
                
                if blockingPairs(size, instance[4], instance[5], candidate, "weak")['blockingPairCount'] == size:
                    break
                ntries -= 1
            random_size = copy.deepcopy(candidate)

            num_blocking_3 = blockingPairs(size, instance[4], instance[5], candidate, "weak")['blockingPairCount']
            # print("Rn", num_blocking_3)

            if num_blocking_1 != 1 or num_blocking_2 != round(math.sqrt(size)) or num_blocking_3 != size:
                print("FAILED TO FIND UNSTABLE MATCHINGS FOR INSTANCE ", i)
                continue

            for j in range(len(instance)):
                new_dict[instances.columns[j]].append(instance[j])
            new_dict['random_1'].append(format_matching({man+1: random_1[man] for man in range(size)}))
            new_dict[f'random_{round(math.sqrt(size))}'].append(format_matching({man+1: random_sqrt[man] for man in range(size)}))
            new_dict[f'random_{size}'].append(format_matching({man+1: random_size[man] for man in range(size)}))

            new_dict["men_opt"].append(format_matching(matchings['men_opt']))
            new_dict["women_opt"].append(format_matching(matchings['women_opt']))
            if culture == "womanmaster":
                new_dict["lattice"].append(None)
            else:
                lattice_matching = matchings['lattice'][len(matchings['lattice'])//2]
                new_dict["lattice"].append(format_matching(lattice_matching))

            random_list = [i for i in range(1,size+1)]
            shuffle(random_list)
            random_matching = {i: random_list[i-1] for i in range(1, size+1)}
            new_dict["random"].append(format_matching(random_matching))
        
        if len(new_dict["men_opt"]) > num_instances-1:
            break

    if delete_files: shutil.rmtree(f'../instances_matchings/{size}/')

    data = pd.DataFrame.from_dict(new_dict)
    data.to_csv(out_file_name, index=False)

    return True

def generate_questions(file_name):
    size = int(file_name.split('/')[2].split('_')[0])
    fields = ["pref_type","n_man","n_woman","combined_pref_json","man_pref_string","woman_pref_string","men_opt","women_opt","lattice","random_1",f"random_{round(math.sqrt(size))}",f"random_{size}","random", "level1_q", "level1_a", "level2_q", "level2_a", "level2n_q", "level2n_a"]

    # print(file_name.split('_')[0])

    # if file_name.split('/')[2].split('_')[0] != '10':
    #     return

    # for i in sizes:
    #     fields.extend([f"unstable_swap{i}", f"bp_count_swap{i}", f"bp_list_swap{i}"])
    rows = []
    # cnt = 0
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        # Optional: Get headers if they exist
        headers = next(csv_reader)  # Skip header row
        if 'level1_q' in headers:
            print(f'Question data present for {file_name}!')
            return 'ALL DONE'
        # Loop through rows
        for row in csv_reader:

            curr_row = copy.deepcopy(row)
            # level 1
            n = int(row[1])
            man_pref_str = row[4]
            woman_pref_str = row[5]
            agent = randint(1,n)
            position = randint(1,n)
            question_lvl_1 = f"Who is agent W{agent}'s, {position}-most preferred agent?"
            pref_list = woman_pref_str.split()
            agent_pref = pref_list[agent-1].split(",")
            answer_lvl_1 = f"M{agent_pref[position-1]}"
            curr_row.extend((question_lvl_1, answer_lvl_1))


            # level 2
            agent = randint(1,n)
            comp1, comp2, partner = sample(range(1, n), 3)
            andFlag = "and" if randint(0,1) == 1 else "or"

            question_lvl_2 = f"Would agent W{agent}, prefer M{comp1} {andFlag} M{comp2} over M{partner}?"
            pref_list = woman_pref_str.split()
            agent_pref = pref_list[agent-1].split(",")

            comp1_pos = agent_pref.index(f"{comp1}")
            comp2_pos = agent_pref.index(f"{comp2}")
            partner_pos = agent_pref.index(f"{partner}")

            conditionTrue = False
            if andFlag == "and":
                conditionTrue = comp1_pos < partner_pos and comp2_pos < partner_pos
            else:
                conditionTrue = comp1_pos < partner_pos or comp2_pos < partner_pos

            answer_lvl_2 = "yes" if conditionTrue else "no"
            curr_row.extend((question_lvl_2, answer_lvl_2))

            # noisy level 2
            question_lvl_2_noisy = f"If agent W{agent} is currently engaged to M{partner}, would she accept proposals from M{comp1} {andFlag} M{comp2}?"
            curr_row.extend((question_lvl_2_noisy, answer_lvl_2))

            # print(curr_row)

            rows.append(curr_row)

            # if cnt > 5:
            #     break 

    
    # file_name = '../instance_tests/'+file_name
    writeCSV(fields=fields, rows=rows, csv_name=file_name)
    return "Successfully created .txt and .csv files"

def writeCSV(fields, rows, csv_name=None):
    file_name = csv_name
    with open(file_name, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)


def main():
    instance_files = os.listdir('../instance_files/')
    # print(instance_files)
    for instance_file in instance_files:
        print("GENERATING MATCHINGS FOR", instance_file)
        compute_matchings(f'../instance_files/{instance_file}', num_instances=100, size_max=50)

    instance_files = os.listdir('../instances_matchings/')
    for instance_file in instance_files:
        print("GENERATING QUESTIONS FOR", instance_file)
        generate_questions(f'../instances_matchings/{instance_file}')
    
if __name__ == "__main__":
    main()