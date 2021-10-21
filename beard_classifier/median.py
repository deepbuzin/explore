def minimumDays(rows, columns, grid):
    import queue
    q = queue.Queue()
    days = [[-1 for _ in range(columns)] for _ in range(rows)]
    for i in range(rows):
        for j in range(columns):
            if grid[i][j] == 1:
                days[i][j] = 0
                q.put((i, j, 0))
    while not q.empty():
        (i, j, day) = q.get()
        bfs(i, j, day, rows, columns, grid, days, q)
    return max(map(max, days))


def bfs(i, j, day, rows, cols, grid, days, q):
    if not 0 <= i < rows:
        return 0
    if not 0 <= j < cols:
        return 0
    if days[i][j] > -1:
        return 0
    days[i][j] = day
    grid[i][j] = 1
    q.put(i - 1, j, day + 1)
    q.put(i + 1, j, day + 1)
    q.put(i, j - 1, day + 1)
    q.put(i, j + 1, day + 1)


if __name__ == '__main__':
    a = [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0]]
    print(minimumDays(5, 5, a))
