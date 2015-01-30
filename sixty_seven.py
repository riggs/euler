def qbertRun(tree):
    paths = [(tree[0][0], '')]
    for row in tree[1:]:
        paths.append((paths[-1][0] + row[-1], paths[-1][1] + 'R'))
        for i in range(len(paths)-2, 0, -1):
            paths[i] = max((paths[i-1][0] + row[i], paths[i-1][1] + 'R'),
                            (paths[i][0] + row[i], paths[i][1] + 'L'))
        else:
            paths[0] = (paths[0][0] + row[0], paths[0][1] + 'L')
    return max(paths)
