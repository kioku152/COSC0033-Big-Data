import numpy as np
from scipy.sparse import csr_matrix

file_ = 'Data.txt'

def inital_data():
    #遍历，记录节点信息
    all_nodes = set()
    degrees = {}  # 记录每个节点的出度

    #读取文件获取所有节点
    with open(file_, 'r') as f:
        for line in f:
            #滤去空行
            if line.strip() == '':
                continue
            #筛出出入点对，记录总共的节点信息
            parts = line.split()
            FromNode = int(parts[0])
            ToNode = int(parts[1])
            all_nodes.add(FromNode)
            all_nodes.add(ToNode)
            #统计 出节点 的个数
            if FromNode in degrees:
                degrees[FromNode] += 1
            else:
                degrees[FromNode] = 1

    #排序节点，建立索引
    nodes_sort = sorted(all_nodes)
    total_node = len(nodes_sort)
    #创建节点序号映射信息，防止有空缺的序号占用空间
    node_id = {node: i for i, node in enumerate(nodes_sort)}

    #再遍历，构建矩阵
    row = []
    col = []
    values = []

    with open(file_, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            parts = line.split()
            FromNode = int(parts[0])
            ToNode = int(parts[1])
            #滤去黑洞节点
            if FromNode in degrees:
                FromId = node_id[FromNode]
                col.append(FromId)
                ToId = node_id[ToNode]
                row.append(ToId)
                #初始化权重
                weight = 1.0 / degrees[FromNode]
                values.append(weight)

    #创建稀疏矩阵
    ans = csr_matrix((values, (row, col)), shape=(total_node, total_node))
    return ans, nodes_sort, total_node, node_id


def pagerank(M, total, rounds, teleport, target):
    #初始化权重
    pr = np.full(total, 1.0 / total)
    #筛出黑洞节点
    sum_ = np.array(M.sum(axis=0)).flatten()
    blackholes = np.where(sum_ == 0)[0]

    for i in range(rounds):
        last = pr.copy()

        pr = M.dot(last) * teleport
        #处理黑洞节点，将权重平均分配到所有节点
        if len(blackholes) > 0:
            pr += np.sum(last[blackholes]) * teleport / total
        pr += (1 - teleport) / total

        # 计算收敛情况
        ans = 0.0
        for i in range(total):
            ans += abs(pr[i] - last[i])
        if ans < target:
            break

    return pr


def answer(ans, nodes, node_id, top):
    rank = []
    for i in range(len(nodes)):
        rank.append((ans[node_id[nodes[i]]], nodes[i]))
    #分数降序，相同分数节点升序
    rank.sort(key=lambda x: (-x[0], x[1]))
    return rank[:top]


if __name__ == '__main__':
    M, N, total, node_id = inital_data()

    results = pagerank(M, total, 100, 0.85, 1e-6)

    ans = answer(results, N, node_id, 100)

    with open('Res.txt', 'w') as f:
        for score, node in ans:
            f.write(f"{node} {score:.10f}\n")