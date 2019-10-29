import numpy as np

def _get_certain_tiles(map, values):
    tiles = []
    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            if map[y][x] in values:
                tiles.append((x, y))
    return tiles

def _flood_fill(x, y, color_map, map, region_index, values):
    num_tiles = 0
    queue = [(x, y)]
    while len(queue) > 0:
        (cx, cy) = queue.pop(0)
        if color_map[cy][cx] != -1 or map[cy][cx] not in values:
            continue
        num_tiles += 1
        color_map[cy][cx] = region_index
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= map.shape[1] or ny >= map.shape[0]:
                continue
            queue.append((nx, ny))
    return num_tiles

def calc_num_regions(map, values=[0]):
    empty_tiles = _get_certain_tiles(map, values)
    region_index=0
    color_map = np.full(map.shape, -1)
    for (x,y) in empty_tiles:
        num_tiles = _flood_fill(x, y, color_map, map, region_index + 1, values)
        if num_tiles > 0:
            region_index += 1
        else:
            continue
    return region_index

def _run_dikjstra(x, y, map, values):
    dikjstra_map = np.full(map.shape,-1)
    visited_map = np.zeros(map.shape)
    queue = [(x, y, 0)]
    while len(queue) > 0:
        (cx,cy,cd) = queue.pop(0)
        if map[cy][cx] not in values or (dikjstra_map[cy][cx] >= 0 and dikjstra_map[cy][cx] <= cd):
            continue
        visited_map[cy][cx] = 1
        dikjstra_map[cy][cx] = cd
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= map.shape[1] or ny >= map.shape[0]:
                continue
            queue.append((nx, ny, cd + 1))
    return dikjstra_map, visited_map

def calc_longest_path(map, values):
    empty_tiles = _get_certain_tiles(map, values)
    final_visited_map = np.zeros(map.shape)
    final_value = 0
    for (x,y) in empty_tiles:
        if final_visited_map[y][x] > 0:
            continue
        dikjstra_map, visited_map = _run_dikjstra(x, y, map, values)
        final_visited_map += visited_map
        (mx,my) = np.unravel_index(np.argmax(dikjstra_map, axis=None), dikjstra_map.shape)
        dikjstra_map, _ = _run_dikjstra(mx, my, map, values)
        max_value = np.max(dikjstra_map)
        if max_value > final_value:
            final_value = max_value
    return final_value

def calc_certain_tile(map, values):
    total_value = 0
    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            if map[y][x] in values:
                total_value += 1
    return total_value

def calc_num_reachable_tile(map, start_value, passable_values, reachable_values):
    (sx,sy) = _get_certain_tiles(map, [start_value])[0]
    dikjstra_map, _ = _run_dikjstra(sx, sy, map, passable_values)
    tiles = _get_certain_tiles(map, reachable_values)
    total = 0
    for (tx,ty) in tiles:
        if dikjstra_map[ty][tx] >= 0:
            total += 1
    return total
