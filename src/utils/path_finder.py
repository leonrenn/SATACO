"""
############################################################################################
# Script written to find optimum signal regions, as a continuation of the LH2019 project   #
# 'Determination of Independent Signal Regions in LHC Searches for New Physics'            #
# by A. Buckley, B. Fuks, H. Reyes-González, W. Waltenberger, S. Williamson and J. Yellen  #
############################################################################################
"""

from itertools import combinations
from typing import Callable

import numpy as np
from graph import Graph
from more_itertools import chunked


class PathFinder():
    def __init__(self, corelations: np.ndarray, threshold: float = 0.01,
                 source: int = 0,
                 weights: list = None,
                 ignore_subset: bool = True) -> None:
        """
        Calculate available paths useing the 
        Hereditary Depth First (HDFS) algorithm:

        corelations: np.ndarray -> NxN matrix from which 
        the Binary Acceptance (BA) is drawn
        threshold: float -> minimum value for which BA_ij = True
        source: int -> initial index for HDFS
        """
        self.run_num, self.run_sum = 0.0, 0.0
        self.corr = corelations
        self.threshold = threshold
        self.source = source
        self.no_subset = ignore_subset
        # dimentions of corelations
        self.dim = np.array(self.corr.shape)
        # boolian matrix of allowed transitions
        self.corr_mask = (self.corr < self.threshold)
        self.path_algorithm = self.top_weighted_cpath
        # set allowed paths (source)
        self.set_allowed_paths()
        self.set_weighted_graph(weights)
        self.set_weight_func(self.get_weight)
        self.set_weight_limit_func(self.get_weight)

    def set_weight_func(self, weight_funk: Callable) -> None:
        self.weight_func = weight_funk

    def set_weight_limit_func(self, max_weighting: Callable) -> None:
        self.weight_lim_func = max_weighting

    def set_allowed_paths(self) -> None:
        """
        set allowed paths attribute with
        default source_mask to 
        """
        self.allowed_paths = self.path_bool()

    @property
    def get_allowed_tril(self) -> np.ndarray:
        """
        lower tri of self.allowed_path with trace = 0
        """
        return np.tril(self.allowed_paths, -1)

    @property
    def get_allowed_triu(self) -> np.ndarray:
        """
        upper tri of self.allowed_path with trace = 0
        """
        return np.triu(self.allowed_paths, 1)

    def get_weight(self, path: list) -> float:
        """ Get the sum of the weights for a given path of indices"""
        if len(path) > 0:
            return np.sum(self.weights[path])
        else:
            return 0.0

    def reset_source(self, source: int = 0) -> None:
        """
        reset the source node
        """
        if source >= self.dim[0]:
            print('Source out of range Defaulting to zero')
            source = 0
        self.source = source
        self.set_allowed_paths()
        try:
            self.set_weighted_graph(weights=self.weights)
        except AttributeError:
            print('weights not set: Defaulting to a uniform weighting of 1')
            self.set_weighted_graph(weights=None)

    def set_weighted_graph(self, weights: list) -> None:
        """
        add weights to the node edges
        """
        if weights is None:
            weights = self.__no_weight
        else:
            weights = list(weights)
        if len(weights) == self.dim[0]:
            weights += [0.0]
        edges = self.find_edges(weights=weights)

        self.graph = Graph()
        self.graph.add_weighted_edges(edges)
        self.weights = np.array(weights)

    def path_bool(self, idx: int = None, msk: np.ndarray = None) -> np.ndarray:
        """
        Set up the binary independency matrix setting the 
        source diagonal element to True. 

        source_mask: mask matrix by source column (increase selection efficiency)
        """
        if idx is None:
            idx = self.source
        if msk is None:
            msk = np.ones(self.dim+1, dtype='bool')
            msk[:-1:, :-1:] = self.corr_mask
            msk[idx, idx] = True
        # TODO Is this necessary
        #
        sub = ~msk[idx, :]
        msk[:, sub] = False
        msk[sub, :] = False
        return msk

    def find_edges(self, weights: list = None) -> list:
        """
        finds edges of the graph
        weights: (optional) default to 1 if None
        """
        edges = []
        self.allowed_paths
        if weights is None:
            weights = self.__no_weight
        for ij in np.argwhere(self.get_allowed_triu):
            edges.append((*ij, weights[ij[1]]))
        return edges

    def update_runsum(self, path_length: int) -> None:
        self.run_num += 1
        self.run_sum += path_length

    def reset_runsum(self) -> None:
        self.run_num, self.run_sum = 0.0, 0.0

    @property
    def mpl(self):
        return self.run_sum/self.run_num

    @staticmethod
    def strip_subdict(dct: dict, target: str) -> list:
        return [p[target] for _, p in dct.items()]

    def check_if_subset(self, path: list) -> bool:
        if not path:
            return False
        # get allowed nodes that are less than the initial
        prev = self.corr_mask[path[0], :][:path[0]:]
        idx = np.argwhere(prev).flatten().tolist()
        if not idx:
            return False
        for l in range(1, len(idx)+1):
            for comb in combinations(idx, l):
                if self.is_allowed(list(comb) + path):
                    return True
        return False

    def all_conditional_paths(self, trim: bool = True) -> list:
        """
        Hereditary Depth First Search 
        Returns all paths under the Hereditary condition.

        target: finishing node
        cutoff: maximum length of path
        """
        target = self.dim[0]
        cutoff = self.dim[0]
        visited = dict.fromkeys([self.source])
        stack = [(v for _, v in self.graph.edges(self.source))]
        good_nodes = [set(v for _, v in self.graph.edges(self.source))]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                good_nodes.pop()
                visited.popitem()
            elif len(visited) < cutoff:
                if child in visited:
                    continue
                if child == target:
                    if trim:
                        yield list(visited)
                    else:
                        yield list(visited) + [child]
                visited[child] = None
                if target not in visited:
                    # if self.graph.edges([child]):
                    good_children = set(v for _, v in self.graph.edges(child))
                    good_nodes += [good_children.intersection(good_nodes[-1])]
                    stack.append((v for _, v in self.graph.edges(
                        child) if v in good_nodes[-1]))
                else:
                    visited.popitem()
            else:  # len(visited) == cutoff:
                if trim:
                    yield list(visited)
                else:
                    yield list(visited) + [child]
                stack.pop()
                good_nodes.pop()
                visited.popitem()

    def top_weighted_cpath(self, path_weight: dict = None, top: int = 1) -> dict:
        """
        Weighted Hereditary Depth First Search 
        Returns best path for a given source under 
        the weighted Hereditary condition.

        max_wgt: maximum weight for running comparison 
        trim: (bool) trim the target node from result

        """
        if path_weight is None:
            path_weight = {i: {'path': [], 'weight': 0.0} for i in range(top)}
        # initiate empty list for best path
        # max_pth = {i:None for i in range(len(max_wgt))}
        # set cutoff and limit to the length of correlations
        cutoff = self.dim[0] + 1
        target = self.dim[0]
        # chk_ss = False
        # initiate the visited list with the source node
        visited = dict.fromkeys([self.source])
        # stack is a list of generators that builds to provide the subset of
        # available nodes for each child with all nodes > child
        stack = [(v for _, v in self.graph.edges(self.source))]
        # good nodes are the compleat set of available nodes for each child
        # good_nodes = [self.good_nodes(self.source)]
        good_nodes = [set(v for _, v in self.graph.edges(self.source))]
        max_wgt = np.array(self.strip_subdict(path_weight, 'weight'))
        # itterate over nodes building and dropping from stack until empty
        while stack:
            # define childern as the generator from the last element of stack
            children = stack[-1]
            # The child node is the next element from children
            child = next(children, None)
            # if no child drop last elements from stack, good nodes and visited
            if child is None:
                stack.pop()
                good_nodes.pop()
                visited.popitem()
            # number of nodes in path less then the length of the correlations
            elif len(visited) < cutoff:
                # ensure no repeted nodes
                if child in visited:
                    continue
                # define current path bing considered
                pth = list(visited) + [child]
                # take the intersection of nodes available to the child
                # with those available to all previous nodes in path
                # gn = set(v for _, v in self.graph.edges(child)).intersection(*good_nodes)
                gn = set(v for _, v in self.graph.edges(
                    child)).intersection(good_nodes[-1])
                # list the available nodes from the set gn
                child_pths = np.array(list(gn))
                # weight of current path
                currnt_wgt = self.weight_func(pth)
                # upper limit on the weight available to the child
                remain_wgt = self.weight_lim_func(
                    list(child_pths[(child_pths > child)]))
                # target reached
                if child == target:
                    # self.update_runsum(len(pth)-1)
                    paths = self.strip_subdict(path_weight, 'path')
                    # is the current weight the best so far
                    if currnt_wgt > max_wgt.min():
                        # if self.no_subset:
                        #     chk_ss = any([set(pth[:-1:]).issubset(item) for item in paths])
                        #     if not chk_ss:
                        #         chk_ss = self.check_if_subset(pth[:-1:])
                        # if not chk_ss:
                        # self.update_runsum(len(pth)-1)
                        if top > 1:
                            # paths = self.strip_subdict(path_weight, 'path')
                            wgts = self.strip_subdict(path_weight, 'weight')
                            paths.append(pth[:-1:])
                            wgts.append(currnt_wgt)
                            path_weight = self.rank_path_by_weight(
                                paths, weights=wgts, top=top)
                        else:
                            path_weight = {
                                0: {'path': pth[:-1:], 'weight': currnt_wgt}}
                        max_wgt = np.array(
                            self.strip_subdict(path_weight, 'weight'))
                # is the remaining weight enough to continue down this path
                if (currnt_wgt + remain_wgt) > max_wgt.min():
                    visited[child] = None
                    if target not in visited:
                        # add gn to good nodes
                        good_nodes.append(gn)
                        # add the nest node generator to stack
                        stack.append((v for _, v in self.graph.edges(
                            child) if v in good_nodes[-1]))
                    else:
                        visited.popitem()
            else:  # len(visited) == cutoff:
                stack.pop()
                good_nodes.pop()
                visited.popitem()
        return path_weight

    def find_path(self, runs: int = None, top: int = 1) -> dict:

        if runs is None or runs > self.dim[0]:
            runs = self.dim[0]
        # print('runs:', runs)
        pth = None
        for i in range(0, runs):
            pth = self.top_weighted_cpath(path_weight=pth, top=top)
            if i < self.dim[0]-1:
                self.reset_source(i+1)
            # if i > max_c:
            #     print('Breaking!! Source beyond dimension')
            #     break

        return pth

    def find_all_paths(self, runs: int = None, top: int = None) -> dict:

        max_c = self.dim[0]
        if runs is None or runs > max_c:
            runs = max_c
        pths = []
        for i in range(0, runs):
            all_p = self.all_conditional_paths(trim=True)
            top_p = []
            for item in chunked(all_p, 500):
                cp = list(item)
                if top_p:
                    cp += top_p
                all_dict = self.rank_path_by_weight(cp, top=top)
                top_p = self.strip_subdict(all_dict, target='path')
            pths += top_p
            if i < max_c-1:
                self.reset_source(i+1)
            # else:
            #     print('Breaking!! Source beyond dimension')
            #     break
        return self.rank_path_by_weight(pths, top=top)

    def is_allowed(self, path, skip: int = 0) -> bool:
        """
        Check if given path is allowed
        """
        for item in combinations(path[skip::], 2):
            if not self.corr_mask[item]:
                return False
        return True

    def rank_path_by_weight(self, paths: list, weights: list = None, top=None) -> dict:
        """
        Sort a list of paths by the weights
        returns dictionary of paths + weights ranked from 0 (best)
        If no weights is set best = longest

        paths: list of paths
        top: only return top number of paths
        """
        if weights is None:
            weights = self.get_path_weights(paths)
        ret = {}
        w_sort = np.argsort(weights)[::-1]
        if top is not None:
            w_sort = w_sort[:top]
        for i, item in enumerate(w_sort):
            if weights[item] != np.NaN:
                ret[i] = {'path': paths[item], 'weight': weights[item]}
        return ret

    def get_path_weights(self, paths: list) -> list:
        """
        Get the weight of a given path
        path: single path list
        """
        paths_weights = np.zeros(len(paths))
        for i, path in enumerate(paths):
            if path:
                # paths_weights[i] = self.get_weight(path)
                paths_weights[i] = self.weight_func(path)
            else:
                paths_weights[i] = 0
        return paths_weights

    @property
    def __no_weight(self):
        """
        artificialy set weights == 1
        """
        return [1] * len(self.allowed_paths)


if __name__ == '__main__':
    import time

    # seed = np.random.randint(low=0, high=1e6)
    seed = 500
    np.random.seed(seed)
    print(f"seed = {seed}")
    N = 1000
    p = 0.1
    rand_bool = np.triu(np.random.choice(
        [True, False], size=(N, N), p=[p, 1-p]), 1)
    dummy = rand_bool + rand_bool.T
    weights = np.sort(np.random.rand(len(dummy)))[::-1]

    # def get_weight(path:list, weight:np.ndarray)-> np.ndarray:
    #     if path:
    #         return np.sum(weight[path]) - np.log(len(path))
    #     else:
    #         return 0

    pf = PathFinder(np.array(~dummy, dtype=int),
                    weights=weights, ignore_subset=True)
    # pf.set_weight_func(lambda pth: get_weight(pth, pf.weights))
    t1 = time.time()
    print(pf.find_path(top=5))
    print(f"Time: {time.time() - t1} s")
    pf.reset_source()
    # pf.set_weight_func(lambda pth: get_weight(pth, weights))
    t2 = time.time()
    print(pf.find_all_paths(top=5))
    print(f"Time: {time.time() - t2} s")
