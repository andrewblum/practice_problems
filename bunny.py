def max_carrot_path(grid):
    most_carrots_so_far = 0
    best_spot = (0,0)
    for x, row in enumerate(grid): 
        for y, _ in enumerate(row):
            best_if_we_start_here = hop(x, y, grid)
            if best_if_we_start_here > most_carrots_so_far:
                best_spot = (x, y)
                most_carrots_so_far = best_if_we_start_here
    return best_spot


def hop(x, y, grid):
    seen = set()
    current_plot = (x, y)
    total_carrots = 0
    while current_plot:   
        x, y = current_plot[0], current_plot[1]
        plot_with_most_carrots = None
        most_nearby_carrots = 0
        for x2, y2 in [(x,y-1), (x+1, y), (x, y+1), (x-1, y)]:
            if (x2 >= 0 and y2 >= 0 and 
                x2 < len(grid) and 
                y2 < len(grid[0]) and 
                not (x2, y2) in seen and 
                grid[x2][y2] != 0):
                seen.add((x2, y2))
                if not plot_with_most_carrots or grid[x2][y2] > most_nearby_carrots:
                    plot_with_most_carrots = (x2, y2)
                    most_nearby_carrots = grid[x2][y2]
        current_plot = plot_with_most_carrots
        total_carrots += most_nearby_carrots
    return total_carrots



garden = [[2, 3, 1, 4, 2, 2, 3],
[2, 3, 0, 4, 0, 3, 0],
[1, 7, 0, 2, 1, 2, 3],
[9, 3, 0, 4, 2, 0, 3]]

print(max_carrot_path(garden))



