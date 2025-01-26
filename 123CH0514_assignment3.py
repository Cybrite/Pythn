from collections import deque

def dfs(adj_List, start, visited=None):
    if visited is None:
        visited = []  
    
    visited.append(start)  
    
    for neighbor in adj_List[start]:
        if neighbor not in visited:  
            dfs(adj_List, neighbor, visited)
    
    return visited


def bfs(adj_List, start, target):
    visited = set()
    queue = deque([(start, [start])])

    while queue:
        current_node, path = queue.popleft()

        if current_node == target:
            return path

        if current_node not in visited:
            visited.add(current_node)
            for neighbor in adj_List[current_node]:
              if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return 'no path found'

adj_List = {
    'A' : ['B','C'],
    'B' : ['A','D'],
    'C' : ['A','E'],
    'D' : ['B','F'],
    'E' : ['C','F'],
    'F' : ['D','E']
}

print("BFS : ",bfs(adj_List, 'A', 'F'))


print("DFS : ",dfs(adj_List, 'A'))



# def dfs(adj_List, start, target, visited=None,path=None):
#     if visited is None:
#         visited = set()
#     if path is None:
#         path = []

#     visited.add(start)
#     path.append(start)

#     if start == target:
#         return path

#     for neighbor in adj_List[start]:
#         if neighbor not in visited:
#             result = dfs(adj_List, neighbor, target, visited, path)
#             if result is not None:
#                 return result

#     visited.remove(start)
#     path.pop()

#     return None