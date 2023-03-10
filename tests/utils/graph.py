from typing import List


class Graph(object):
    def __init__(self):
        self._adj: dict = dict()
        self._node: dict = dict()

    def __construct_nodes(self,
                          edge: list) -> None:
        for item in edge:
            if item[0] not in self._node:
                self._node[item[0]] = {}
                self._adj[item[0]] = {}

    def __construct_adj(self,
                        edge: List) -> None:
        self.__construct_nodes(edge)
        for item in edge:
            source, child, weight = item
            if child not in self._adj[source]:
                self._adj[source][child] = {}
            index = len(self._adj[source][child])
            self._adj[source][child] = {index: {'weight': weight}}

    def add_weighted_edges(self,
                           edges: List) -> None:
        self.__construct_adj(edges)

    # def edges(self, srce:List=None)->List:
    #     if isinstance(srce, (int, float)):
    #         srce = [int(srce)]
    #     if srce is None:
    #         return [(k, i) for k, subdict in self._adj.items()
    # for i in subdict]
    #     else:
    #         return [(k, i) for k, subdict in self._adj.items()
    # for i in subdict if k in srce]

    def edges(self,
              srce: int = None) -> List:
        if isinstance(srce, list):
            srce = srce[0]
        if srce is None:
            return [(k, i) for k, subdict in self._adj.items()
                    for i in subdict]
        if srce in self._adj:
            return [(srce, i) for i in self._adj[srce]]
        else:
            return []
