import logging, numpy as np


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class Graph:
    def __init__(self, dataset, max_hop=3, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'coco':
            # keypoints = {
            #     0: "nose",
            #     1: "left_eye", "Mid of left_shoulder and right_shoulder"
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            #     
            # }
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            # L F 14
            neighbor_link = [(0, 1), (5, 1), (1, 6),
                             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            
            
            """
            # Org
            neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            """
            
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            connect_joint = np.array([0,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            parts = [
                np.array([5, 7, 9]),       # left_arm
                np.array([6, 8, 10]),      # right_arm
                np.array([11, 13, 15]),    # left_leg
                np.array([12, 14, 16]),    # right_leg
                np.array([0, 1, 2, 3, 4]),  # head
            ]

        elif self.dataset == 'oumvlp':
            # keypoints = {
            #     0: "nose",
            #     1: "neck"
            #     2: "Rshoulder"
            #     3: "Relbow"
            #     4: "Rwrist"
            #     5: "Lshoudler"
            #     6："Lelbow"
            #     7: "Lwrist"
            #     8: "Rhip"
            #     9: "Rknee"
            #     10: "Rankle"
            #     11: "Lhip"
            #     12: "Lknee"
            #     13: "Lankle"
            #     14: "Reye"
            #     15: "Leye"
            #     16: "Rear"
            #     17: "Lear"
            # }
            num_node = 18
            
            self_link = [(i, i) for i in range(num_node)]
            # L F 14
            neighbor_link = [(0, 1), 
                             (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (2, 8), (8, 9), (9, 10), (5, 11), (11, 12), (12, 13), (11, 8)]
            """
            # Org
            neighbor_link = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
                             (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
            """
            self.edge = self_link + neighbor_link
            self.center = 1
            self.flip_idx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),  # left_arm
                np.array([2, 3, 4]),  # right_arm
                np.array([11, 12, 13]),  # left_leg
                np.array([8, 9, 10]),  # right_leg
                np.array([0, 1, 14, 15, 16, 17]),  # head
            ]

        else:
            num_node, neighbor_link, connect_joint, parts = 0, [], [], []
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    @staticmethod
    def _normalize_digraph(A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
