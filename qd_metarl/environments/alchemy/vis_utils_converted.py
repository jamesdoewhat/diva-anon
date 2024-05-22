# Trying to convert original vpython code to mayavi.
from mayavi import mlab
import numpy as np

from vis_constants_converted import NODES_MAP, ARROWS_COLOR_MAP, STONE_COLORS, COORDS

# Draw a line in 3D space between two points.
def draw_edge(a, b):
    mlab.plot3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], tube_radius=0.025, color=(0, 0, 0))

# Draw an arrow in 3D space from point a to point b with a specified color.
def draw_arrow(a, b, color=(0, 1, 0)):
    mlab.quiver3d(a[0], a[1], a[2], b[0]-a[0], b[1]-a[1], b[2]-a[2], color=color, scale_factor=0.5, mode='arrow')

# Draw multiple edges based on a graph structure.
def draw_cube(coords, graph):
    for i, (coord_a, coord_b) in enumerate(coords):
        start_node = graph.node_list.get_node_by_idx(NODES_MAP[i][0])
        end_node = graph.node_list.get_node_by_idx(NODES_MAP[i][1])
        if graph.edge_list.has_edge(start_node, end_node):
            draw_edge(coord_a, coord_b)

# Draw cylinders representing potions.
def draw_potions(state):
    p_pos = -1.35
    for i, idx in enumerate(range(15, 39, 2)):
        potion_color = tuple(np.array(ARROWS_COLOR_MAP[int(np.round(state[idx]*3 + 3))]))
        mlab.cylinder(x=p_pos, y=2, z=0, radius=0.1, height=0.1, color=potion_color)
        p_pos += 0.25

# Draw spheres representing stones.
def draw_stones(stones_state):
    for stone in stones_state:
        mlab.points3d(*stone.latent, color=STONE_COLORS[stone.idx], scale_factor=0.2)

# Draw with Mayavi.
def draw_with_mayavi(game_state, state, stones_state, coords, graph):
    draw_cube(coords, graph)
    draw_potions(state)
    draw_stones(stones_state)

    mlab.show()

# Assuming ARROWS_COLOR_MAP and STONE_COLORS are previously defined,
# you can call the function draw_with_mayavi with the appropriate parameters.
