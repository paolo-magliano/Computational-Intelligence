
NUM_FRIENDS = 4
from gx_utils import *
from itertools import product, combinations
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

def node2sets(node):
    pizzeria, pub = node.split("-")
    return Multiset(pizzeria), Multiset(pub)


def sets2node(pizzeria, pub):
    return "".join(pizzeria) + "-" + "".join(pub)


sspace = nx.Graph()
for c, d, b in product(range(NUM_FRIENDS // 2 + 1), range(NUM_FRIENDS // 2 + 1), [True, False]):
    pizzeria = ("*" if b else "") + "C" * c + "D" * d
    pub = ("*" if not b else "") + "C" * (NUM_FRIENDS // 2 - c) + "D" * (NUM_FRIENDS // 2 - d)
    sspace.add_node(pizzeria + "-" + pub)
SOURCE = "*" + "C" * (NUM_FRIENDS // 2) + "D" * (NUM_FRIENDS // 2) + "-"
DESTINATION = "-*" + "C" * (NUM_FRIENDS // 2) + "D" * (NUM_FRIENDS // 2)

plt.figure(figsize=(12, 8))
nx.draw(sspace, with_labels=True)

def valid_node(node):
    pizzeria, pub = node2sets(node)
    return valid_location(pizzeria) and valid_location(pub)


def valid_location(loc):
    if loc.count("*") == 1 and len(loc) == 1:
        return False
    if 0 < loc.count("D") < loc.count("C"):
        return False
    return True


plt.figure(figsize=(12, 8))
nx.draw(
    sspace,
    with_labels=True,
    node_color=["lightgreen" if valid_node(n) else "lightcoral" for n in sspace],
)

def bike(from_, to):
    possible = list()
    for b in ["*C", "*CC", "*D", "*DD", "*CD"]:
        bike = Multiset(b)
        if bike <= from_:
            possible.append(sets2node(from_ - bike, to + bike))
    return possible


valid_sspace = nx.Graph()

for node in sspace.nodes:
    if valid_node(node):
        valid_sspace.add_node(node)

for n1, n2 in [(a, b) for a in valid_sspace.nodes for b in bike(*node2sets(a)) if valid_node(b)]:
    valid_sspace.add_edge(n1, n2)

plt.figure(figsize=(12, 8))
node_color = ["pink" if n == SOURCE or n == DESTINATION else "lightblue" for n in valid_sspace]
nx.draw(valid_sspace, pos=graphviz_layout(valid_sspace, prog="neato"), with_labels=True, node_color=node_color)

nx.shortest_path(valid_sspace, SOURCE, DESTINATION)
['*CCDD-', 'DD-*CC', '*CDD-C', 'C-*CDD', '*CD-CD', '-*CCDD']
 