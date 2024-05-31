import numpy as np
import time
filter5 = [1,1,1,1,1]
filter4 = [0,1,1,1,1,0]

filterClash4a = [0,1,1,1,1,2]
filterClash4b = [2,1,1,1,1,0]
filterClash4c = [1,0,1,1,1]
filterClash4d = [1,1,1,0,1]
filterClash4e = [1,1,0,1,1]

filterLive3a = [0,1,1,1,0]
filterLive3b = [0,1,0,1,1,0]
filterLive3c = [0,1,1,0,1,0]

filterClash3d = [0,0,1,1,1,2]
filterClash3f = [2,1,1,1,0,0]
filterClash3e = [2,0,1,1,1,0,2]
filterClash3g = [0,1,0,1,1,2]
filterClash3h = [2,1,1,0,1,0]
filterClash3i = [0,1,1,0,1,2]
filterClash3j = [2,1,0,1,1,0]
filterClash3k = [1,0,0,1,1]
filterClash3l = [1,1,0,0,1]
filterClash3m = [1,0,1,0,1]

filter2a = [0,1,1,0]
filter2b = [0,1,0,1,0]

class app:
    def __init__(self,mapsize):
        self.pos_score = np.array(
            [[(mapsize//2 - max(abs(x - mapsize//2), abs(y - mapsize//2))) for x in range(mapsize)] for y in range(mapsize)])
        self.map = np.zeros((mapsize,mapsize))
        self.mapsize = mapsize

    def getcount(self):
        self.mapsize+=2
        map = np.ones((self.mapsize,self.mapsize))*2
        map[1:-1,1:-1]=self.map
        self.map=map
        count = []
        # gu=time.time()


        for i in range(self.mapsize-7,self.mapsize):
            for j in range(self.mapsize-7):
                # ------------------------------------------------
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter5)] == filter5 for k in
                          range(7 - len(filter5))]
                if Filter[-1]:
                    count.append("5")
                elif True in Filter and j == 0:
                    count.append("5")

                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter4)] == filter4 for k in
                          range(7 - len(filter4))]
                if Filter[-1]:
                    count.append("4")
                elif True in Filter and j == 0:
                    count.append("4")

                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4a)] == filterClash4a for k in
                          range(7 - len(filterClash4a))]
                if Filter[-1]:
                    count.append("4a")
                elif True in Filter and j == 0:
                    count.append("4a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4b)] == filterClash4b for k in
                          range(7 - len(filterClash4b))]
                if Filter[-1]:
                    count.append("4b")
                elif True in Filter and j == 0:
                    count.append("4b")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4c)] == filterClash4c for k in
                          range(7 - len(filterClash4c))]
                if Filter[-1]:
                    count.append("4c")
                elif True in Filter and j == 0:
                    count.append("4c")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4d)] == filterClash4d for k in
                          range(7 - len(filterClash4d))]
                if Filter[-1]:
                    count.append("4d")
                elif True in Filter and j == 0:
                    count.append("4d")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4e)] == filterClash4e for k in
                          range(7 - len(filterClash4e))]
                if Filter[-1]:
                    count.append("4e")
                elif True in Filter and j == 0:
                    count.append("4e")

                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterLive3a)] == filterLive3a for k in
                          range(7 - len(filterLive3a))]
                if Filter[-1]:
                    count.append("3a")
                elif True in Filter and j == 0:
                    count.append("3a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterLive3b)] == filterLive3b for k in
                          range(7 - len(filterLive3b))]
                if Filter[-1]:
                    count.append("3b")
                elif True in Filter and j == 0:
                    count.append("3b")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterLive3c)] == filterLive3c for k in
                          range(7 - len(filterLive3c))]
                if Filter[-1]:
                    count.append("3c")
                elif True in Filter and j == 0:
                    count.append("3c")

                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3d)] == filterClash3d for k in
                          range(7 - len(filterClash3d))]
                if Filter[-1]:
                    count.append("3d")
                elif True in Filter and j == 0:
                    count.append("3d")
                Filter = self.map[i, j:j + 7].tolist() == filterClash3e
                if Filter:
                    count.append("3e")

                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3f)] == filterClash3f for k in
                          range(7 - len(filterClash3f))]
                if Filter[-1]:
                    count.append("3f")
                elif True in Filter and j == 0:
                    count.append("3f")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3g)] == filterClash3g for k in
                          range(7 - len(filterClash3g))]
                if Filter[-1]:
                    count.append("3g")
                elif True in Filter and j == 0:
                    count.append("3g")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3h)] == filterClash3h for k in
                          range(7 - len(filterClash3h))]
                if Filter[-1]:
                    count.append("3h")
                elif True in Filter and j == 0:
                    count.append("3h")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3i)] == filterClash3i for k in
                          range(7 - len(filterClash3i))]
                if Filter[-1]:
                    count.append("3i")
                elif True in Filter and j == 0:
                    count.append("3i")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3j)] == filterClash3j for k in
                          range(7 - len(filterClash3j))]
                if Filter[-1]:
                    count.append("3j")
                elif True in Filter and j == 0:
                    count.append("3j")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3k)] == filterClash3k for k in
                          range(7 - len(filterClash3k))]
                if Filter[-1]:
                    count.append("3k")
                elif True in Filter and j == 0:
                    count.append("3k")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3l)] == filterClash3l for k in
                          range(7 - len(filterClash3l))]
                if Filter[-1]:
                    count.append("3l")
                elif True in Filter and j == 0:
                    count.append("3l")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3m)] == filterClash3m for k in
                          range(7 - len(filterClash3m))]
                if Filter[-1]:
                    count.append("3m")
                elif True in Filter and j == 0:
                    count.append("3m")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2a)] == filter2a for k in
                          range(7 - len(filter2a))]
                if Filter[-1]:
                    count.append("2a")
                elif True in Filter and j == 0:
                    count.append("2a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2b)] == filter2b for k in
                          range(7 - len(filter2b))]
                if Filter[-1]:
                    count.append("2b")
                elif True in Filter and j == 0:
                    count.append("2b")
                #     ********************************************************

                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filter5)] == filter5 for k in range(7 - len(filter5))]
                if Filter[-1]:
                    count.append("5")
                elif True in Filter and j == 0:
                    count.append("5")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filter4)] == filter4 for k in range(7 - len(filter4))]
                if Filter[-1]:
                    count.append("4")
                elif True in Filter and j == 0:
                    count.append("4")

                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash4a)] == filterClash4a for k in
                          range(7 - len(filterClash4a))]
                if Filter[-1]:
                    count.append("4a")
                elif True in Filter and j == 0:
                    count.append("4a")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash4b)] == filterClash4b for k in
                          range(7 - len(filterClash4b))]
                if Filter[-1]:
                    count.append("4b")
                elif True in Filter and j == 0:
                    count.append("4b")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash4c)] == filterClash4c for k in
                          range(7 - len(filterClash4c))]
                if Filter[-1]:
                    count.append("4c")
                elif True in Filter and j == 0:
                    count.append("4c")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash4d)] == filterClash4d for k in
                          range(7 - len(filterClash4d))]
                if Filter[-1]:
                    count.append("4d")
                elif True in Filter and j == 0:
                    count.append("4d")
                Filter = self.map[j:j + 7, i].tolist() == filterClash4e
                if Filter:
                    count.append("4e")

                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterLive3a)] == filterLive3a for k in
                          range(7 - len(filterLive3a))]
                if Filter[-1]:
                    count.append("3a")
                elif True in Filter and j == 0:
                    count.append("3a")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterLive3b)] == filterLive3b for k in
                          range(7 - len(filterLive3b))]
                if Filter[-1]:
                    count.append("3b")
                elif True in Filter and j == 0:
                    count.append("3b")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterLive3c)] == filterLive3c for k in
                          range(7 - len(filterLive3c))]
                if Filter[-1]:
                    count.append("3c")
                elif True in Filter and j == 0:
                    count.append("3c")

                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3d)] == filterClash3d for k in
                          range(7 - len(filterClash3d))]
                if Filter[-1]:
                    count.append("3d")
                elif True in Filter and j == 0:
                    count.append("3d")
                Filter = self.map[j:j + 7, i].tolist() == filterClash3e
                if Filter:
                    count.append("3e")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3f)] == filterClash3f for k in
                          range(7 - len(filterClash3f))]
                if Filter[-1]:
                    count.append("3f")
                elif True in Filter and j == 0:
                    count.append("3f")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3g)] == filterClash3g for k in
                          range(7 - len(filterClash3g))]
                if Filter[-1]:
                    count.append("3g")
                elif True in Filter and j == 0:
                    count.append("3g")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3h)] == filterClash3h for k in
                          range(7 - len(filterClash3h))]
                if Filter[-1]:
                    count.append("3h")
                elif True in Filter and j == 0:
                    count.append("3h")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3i)] == filterClash3i for k in
                          range(7 - len(filterClash3i))]
                if Filter[-1]:
                    count.append("3i")
                elif True in Filter and j == 0:
                    count.append("3i")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3j)] == filterClash3j for k in
                          range(7 - len(filterClash3j))]
                if Filter[-1]:
                    count.append("3j")
                elif True in Filter and j == 0:
                    count.append("3j")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3k)] == filterClash3k for k in
                          range(7 - len(filterClash3k))]
                if Filter[-1]:
                    count.append("3k")
                elif True in Filter and j == 0:
                    count.append("3k")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3l)] == filterClash3l for k in
                          range(7 - len(filterClash3l))]
                if Filter[-1]:
                    count.append("3l")
                elif True in Filter and j == 0:
                    count.append("3l")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3m)] == filterClash3m for k in
                          range(7 - len(filterClash3m))]
                if Filter[-1]:
                    count.append("3m")
                elif True in Filter and j == 0:
                    count.append("3m")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2a)] == filter2a for k in
                          range(7 - len(filter2a))]
                if Filter[-1]:
                    count.append("2a")
                elif True in Filter and j == 0:
                    count.append("2a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2b)] == filter2b for k in
                          range(7 - len(filter2b))]
                if Filter[-1]:
                    count.append("2b")
                elif True in Filter and j == 0:
                    count.append("2b")
        #         *****************************************************
        # print(time.time()-gu)
        # exit()

        for i in range(self.mapsize-7):
            for j in range(self.mapsize-7):
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filter5)] == filter5 for k in range(7 - len(filter5))]
                if True in Filter and j==0:
                    count.append("5")
                elif True in Filter and j == 0:
                    count.append("5")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filter4)] == filter4 for k in range(7 - len(filter4))]
                if True in Filter and j==0:
                    count.append("4")
                elif True in Filter and j == 0:
                    count.append("4")

                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash4a)] == filterClash4a for k in
                            range(7 - len(filterClash4a))]
                if True in Filter and j==0:
                    count.append("4a")
                elif True in Filter and j == 0:
                    count.append("4a")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash4b)] == filterClash4b for k in
                            range(7 - len(filterClash4b))]
                if True in Filter and j==0:
                    count.append("4b")
                elif True in Filter and j == 0:
                    count.append("4b")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash4c)] == filterClash4c for k in
                            range(7 - len(filterClash4c))]
                if True in Filter and j==0:
                    count.append("4c")
                elif True in Filter and j == 0:
                    count.append("4c")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash4d)] == filterClash4d for k in
                            range(7 - len(filterClash4d))]
                if True in Filter and j==0:
                    count.append("4d")
                elif True in Filter and j == 0:
                    count.append("4d")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash4e)] == filterClash4e for k in
                            range(7 - len(filterClash4e))]
                if True in Filter and j==0:
                    count.append("4e")
                elif True in Filter and j == 0:
                    count.append("4e")

                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterLive3a)] == filterLive3a for k in
                            range(7 - len(filterLive3a))]
                if True in Filter and j==0:
                    count.append("3a")
                elif True in Filter and j == 0:
                    count.append("3a")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterLive3b)] == filterLive3b for k in
                            range(7 - len(filterLive3b))]
                if True in Filter and j==0:
                    count.append("3b")
                elif True in Filter and j == 0:
                    count.append("3b")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterLive3c)] == filterLive3c for k in
                            range(7 - len(filterLive3c))]
                if True in Filter and j==0:
                    count.append("3c")
                elif True in Filter and j == 0:
                    count.append("3c")

                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3d)] == filterClash3d for k in
                            range(7 - len(filterClash3d))]
                if True in Filter and j==0:
                    count.append("3d")
                elif True in Filter and j == 0:
                    count.append("3d")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3e)] == filterClash3e for k in
                            range(7 - len(filterClash3e))]
                if True in Filter and j==0:
                    count.append("3e")
                elif Filter:
                    count.append("3e")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3f)] == filterClash3f for k in
                            range(7 - len(filterClash3f))]
                if True in Filter and j==0:
                    count.append("3f")
                elif True in Filter and j == 0:
                    count.append("3f")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3g)] == filterClash3g for k in
                            range(7 - len(filterClash3g))]
                if True in Filter and j==0:
                    count.append("3g")
                elif True in Filter and j == 0:
                    count.append("3g")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3h)] == filterClash3h for k in
                            range(7 - len(filterClash3h))]
                if True in Filter and j==0:
                    count.append("3h")
                elif True in Filter and j == 0:
                    count.append("3h")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3i)] == filterClash3i for k in
                            range(7 - len(filterClash3i))]
                if True in Filter and j==0:
                    count.append("3i")
                elif True in Filter and j == 0:
                    count.append("3i")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3j)] == filterClash3j for k in
                            range(7 - len(filterClash3j))]
                if True in Filter and j==0:
                    count.append("3j")
                elif True in Filter and j == 0:
                    count.append("3j")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3k)] == filterClash3k for k in
                            range(7 - len(filterClash3k))]
                if True in Filter and j==0:
                    count.append("3k")
                elif True in Filter and j == 0:
                    count.append("3k")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3l)] == filterClash3l for k in
                            range(7 - len(filterClash3l))]
                if True in Filter and j==0:
                    count.append("3l")
                elif True in Filter and j == 0:
                    count.append("3l")
                Filter = [self.map[range(j,j+7), range(i,i+7)].tolist()[k:k + len(filterClash3m)] == filterClash3m for k in
                            range(7 - len(filterClash3m))]
                if True in Filter and j==0:
                    count.append("3m")
                elif True in Filter and j == 0:
                    count.append("3m")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2a)] == filter2a for k in
                          range(7 - len(filter2a))]
                if Filter[-1]:
                    count.append("2a")
                elif True in Filter and j == 0:
                    count.append("2a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2b)] == filter2b for k in
                          range(7 - len(filter2b))]
                if Filter[-1]:
                    count.append("2b")
                elif True in Filter and j == 0:
                    count.append("2b")
                #     ******************************************

                Filter = [self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filter5)] == filter5 for k
                          in range(7 - len(filter5))]
                if Filter[-1]:
                    count.append("5")
                elif True in Filter and j == 0:
                    count.append("5")
                Filter = [self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filter4)] == filter4 for k
                          in range(7 - len(filter4))]
                if Filter[-1]:
                    count.append("4")
                elif True in Filter and j == 0:
                    count.append("4")

                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash4a)] == filterClash4a
                    for k in
                    range(7 - len(filterClash4a))]
                if Filter[-1]:
                    count.append("4a")
                elif True in Filter and j == 0:
                    count.append("4a")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash4b)] == filterClash4b
                    for k in
                    range(7 - len(filterClash4b))]
                if Filter[-1]:
                    count.append("4b")
                elif True in Filter and j == 0:
                    count.append("4b")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash4c)] == filterClash4c
                    for k in
                    range(7 - len(filterClash4c))]
                if Filter[-1]:
                    count.append("4c")
                elif True in Filter and j == 0:
                    count.append("4c")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash4d)] == filterClash4d
                    for k in
                    range(7 - len(filterClash4d))]
                if Filter[-1]:
                    count.append("4d")
                elif True in Filter and j == 0:
                    count.append("4d")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash4e)] == filterClash4e
                    for k in
                    range(7 - len(filterClash4e))]
                if Filter[-1]:
                    count.append("4e")
                elif True in Filter and j == 0:
                    count.append("4e")

                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterLive3a)] == filterLive3a for
                    k in
                    range(7 - len(filterLive3a))]
                if Filter[-1]:
                    count.append("3a")
                elif True in Filter and j == 0:
                    count.append("3a")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterLive3b)] == filterLive3b for
                    k in
                    range(7 - len(filterLive3b))]
                if Filter[-1]:
                    count.append("3b")
                elif True in Filter and j == 0:
                    count.append("3b")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterLive3c)] == filterLive3c for
                    k in
                    range(7 - len(filterLive3c))]
                if Filter[-1]:
                    count.append("3c")
                elif True in Filter and j == 0:
                    count.append("3c")

                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash3d)] == filterClash3d
                    for k in
                    range(7 - len(filterClash3d))]
                if Filter[-1]:
                    count.append("3d")
                elif True in Filter and j == 0:
                    count.append("3d")
                Filter = self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist() == filterClash3e
                if Filter:
                    count.append("3e")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash3f)] == filterClash3f
                    for k in
                    range(7 - len(filterClash3f))]
                if Filter[-1]:
                    count.append("3f")
                elif True in Filter and j == 0:
                    count.append("3f")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash3g)] == filterClash3g
                    for k in
                    range(7 - len(filterClash3g))]
                if Filter[-1]:
                    count.append("3g")
                elif True in Filter and j == 0:
                    count.append("3g")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash3h)] == filterClash3h
                    for k in
                    range(7 - len(filterClash3h))]
                if Filter[-1]:
                    count.append("3h")
                elif True in Filter and j == 0:
                    count.append("3h")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash3i)] == filterClash3i
                    for k in
                    range(7 - len(filterClash3i))]
                if Filter[-1]:
                    count.append("3i")
                elif True in Filter and j == 0:
                    count.append("3i")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash3j)] == filterClash3j
                    for k in
                    range(7 - len(filterClash3j))]
                if Filter[-1]:
                    count.append("3j")
                elif True in Filter and j == 0:
                    count.append("3j")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash3k)] == filterClash3k
                    for k in
                    range(7 - len(filterClash3k))]
                if Filter[-1]:
                    count.append("3k")
                elif True in Filter and j == 0:
                    count.append("3k")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash3l)] == filterClash3l
                    for k in
                    range(7 - len(filterClash3l))]
                if Filter[-1]:
                    count.append("3l")
                elif True in Filter and j == 0:
                    count.append("3l")
                Filter = [
                    self.map[range(j, j + 7), range(i + 6, i - 1, -1)].tolist()[k:k + len(filterClash3m)] == filterClash3m
                    for k in
                    range(7 - len(filterClash3m))]
                if Filter[-1]:
                    count.append("3m")
                elif True in Filter and j == 0:
                    count.append("3m")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2a)] == filter2a for k in
                          range(7 - len(filter2a))]
                if Filter[-1]:
                    count.append("2a")
                elif True in Filter and j == 0:
                    count.append("2a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2b)] == filter2b for k in
                          range(7 - len(filter2b))]
                if Filter[-1]:
                    count.append("2b")
                elif True in Filter and j == 0:
                    count.append("2b")
                #     ******************************

                # ------------------------------------------------
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter5)] == filter5 for k in range(7 - len(filter5))]
                if Filter[-1]:
                    count.append("5")
                elif True in Filter and j == 0:
                    count.append("5")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter4)] == filter4 for k in range(7 - len(filter4))]
                if Filter[-1]:
                    count.append("4")
                elif True in Filter and j == 0:
                    count.append("4")

                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4a)] == filterClash4a for k in
                          range(7 - len(filterClash4a))]
                if Filter[-1]:
                    count.append("4a")
                elif True in Filter and j == 0:
                    count.append("4a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4b)] == filterClash4b for k in
                          range(7 - len(filterClash4b))]
                if Filter[-1]:
                    count.append("4b")
                elif True in Filter and j == 0:
                    count.append("4b")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4c)] == filterClash4c for k in
                          range(7 - len(filterClash4c))]
                if Filter[-1]:
                    count.append("4c")
                elif True in Filter and j == 0:
                    count.append("4c")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4d)] == filterClash4d for k in
                          range(7 - len(filterClash4d))]
                if Filter[-1]:
                    count.append("4d")
                elif True in Filter and j == 0:
                    count.append("4d")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash4e)] == filterClash4e for k in
                          range(7 - len(filterClash4e))]
                if Filter[-1]:
                    count.append("4e")
                elif True in Filter and j == 0:
                    count.append("4e")

                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterLive3a)] == filterLive3a for k in
                          range(7 - len(filterLive3a))]
                if Filter[-1]:
                    count.append("3a")
                elif True in Filter and j == 0:
                    count.append("3a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterLive3b)] == filterLive3b for k in
                          range(7 - len(filterLive3b))]
                if Filter[-1]:
                    count.append("3b")
                elif True in Filter and j == 0:
                    count.append("3b")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterLive3c)] == filterLive3c for k in
                          range(7 - len(filterLive3c))]
                if Filter[-1]:
                    count.append("3c")
                elif True in Filter and j == 0:
                    count.append("3c")

                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3d)] == filterClash3d for k in
                          range(7 - len(filterClash3d))]
                if Filter[-1]:
                    count.append("3d")
                elif True in Filter and j == 0:
                    count.append("3d")
                Filter = self.map[i, j:j + 7].tolist() == filterClash3e
                if Filter:
                    count.append("3e")

                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3f)] == filterClash3f for k in
                          range(7 - len(filterClash3f))]
                if Filter[-1]:
                    count.append("3f")
                elif True in Filter and j == 0:
                    count.append("3f")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3g)] == filterClash3g for k in
                          range(7 - len(filterClash3g))]
                if Filter[-1]:
                    count.append("3g")
                elif True in Filter and j == 0:
                    count.append("3g")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3h)] == filterClash3h for k in
                          range(7 - len(filterClash3h))]
                if Filter[-1]:
                    count.append("3h")
                elif True in Filter and j == 0:
                    count.append("3h")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3i)] == filterClash3i for k in
                          range(7 - len(filterClash3i))]
                if Filter[-1]:
                    count.append("3i")
                elif True in Filter and j == 0:
                    count.append("3i")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3j)] == filterClash3j for k in
                          range(7 - len(filterClash3j))]
                if Filter[-1]:
                    count.append("3j")
                elif True in Filter and j == 0:
                    count.append("3j")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3k)] == filterClash3k for k in
                          range(7 - len(filterClash3k))]
                if Filter[-1]:
                    count.append("3k")
                elif True in Filter and j == 0:
                    count.append("3k")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3l)] == filterClash3l for k in
                          range(7 - len(filterClash3l))]
                if Filter[-1]:
                    count.append("3l")
                elif True in Filter and j == 0:
                    count.append("3l")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filterClash3m)] == filterClash3m for k in
                          range(7 - len(filterClash3m))]
                if Filter[-1]:
                    count.append("3m")
                elif True in Filter and j == 0:
                    count.append("3m")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2a)] == filter2a for k in
                          range(7 - len(filter2a))]
                if Filter[-1]:
                    count.append("2a")
                elif True in Filter and j == 0:
                    count.append("2a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2b)] == filter2b for k in
                          range(7 - len(filter2b))]
                if Filter[-1]:
                    count.append("2b")
                elif True in Filter and j == 0:
                    count.append("2b")
                #     ****************************************

                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filter5)] == filter5 for k in range(7 - len(filter5))]
                if Filter[-1]:
                    count.append("5")
                elif True in Filter and j == 0:
                    count.append("5")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filter4)] == filter4 for k in range(7 - len(filter4))]
                if Filter[-1]:
                    count.append("4")
                elif True in Filter and j == 0:
                    count.append("4")

                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash4a)] == filterClash4a for k in
                          range(7 - len(filterClash4a))]
                if Filter[-1]:
                    count.append("4a")
                elif True in Filter and j == 0:
                    count.append("4a")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash4b)] == filterClash4b for k in
                          range(7 - len(filterClash4b))]
                if Filter[-1]:
                    count.append("4b")
                elif True in Filter and j == 0:
                    count.append("4b")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash4c)] == filterClash4c for k in
                          range(7 - len(filterClash4c))]
                if Filter[-1]:
                    count.append("4c")
                elif True in Filter and j == 0:
                    count.append("4c")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash4d)] == filterClash4d for k in
                          range(7 - len(filterClash4d))]
                if Filter[-1]:
                    count.append("4d")
                elif True in Filter and j == 0:
                    count.append("4d")
                Filter = self.map[j:j + 7, i].tolist() == filterClash4e
                if Filter:
                    count.append("4e")

                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterLive3a)] == filterLive3a for k in
                          range(7 - len(filterLive3a))]
                if Filter[-1]:
                    count.append("3a")
                elif True in Filter and j == 0:
                    count.append("3a")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterLive3b)] == filterLive3b for k in
                          range(7 - len(filterLive3b))]
                if Filter[-1]:
                    count.append("3b")
                elif True in Filter and j == 0:
                    count.append("3b")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterLive3c)] == filterLive3c for k in
                          range(7 - len(filterLive3c))]
                if Filter[-1]:
                    count.append("3c")
                elif True in Filter and j == 0:
                    count.append("3c")

                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3d)] == filterClash3d for k in
                          range(7 - len(filterClash3d))]
                if Filter[-1]:
                    count.append("3d")
                elif True in Filter and j == 0:
                    count.append("3d")
                Filter = self.map[j:j + 7, i].tolist() == filterClash3e
                if Filter:
                    count.append("3e")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3f)] == filterClash3f for k in
                          range(7 - len(filterClash3f))]
                if Filter[-1]:
                    count.append("3f")
                elif True in Filter and j == 0:
                    count.append("3f")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3g)] == filterClash3g for k in
                          range(7 - len(filterClash3g))]
                if Filter[-1]:
                    count.append("3g")
                elif True in Filter and j == 0:
                    count.append("3g")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3h)] == filterClash3h for k in
                          range(7 - len(filterClash3h))]
                if Filter[-1]:
                    count.append("3h")
                elif True in Filter and j == 0:
                    count.append("3h")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3i)] == filterClash3i for k in
                          range(7 - len(filterClash3i))]
                if Filter[-1]:
                    count.append("3i")
                elif True in Filter and j == 0:
                    count.append("3i")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3j)] == filterClash3j for k in
                          range(7 - len(filterClash3j))]
                if Filter[-1]:
                    count.append("3j")
                elif True in Filter and j == 0:
                    count.append("3j")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3k)] == filterClash3k for k in
                          range(7 - len(filterClash3k))]
                if Filter[-1]:
                    count.append("3k")
                elif True in Filter and j == 0:
                    count.append("3k")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3l)] == filterClash3l for k in
                          range(7 - len(filterClash3l))]
                if Filter[-1]:
                    count.append("3l")
                elif True in Filter and j == 0:
                    count.append("3l")
                Filter = [self.map[j:j + 7, i].tolist()[k:k + len(filterClash3m)] == filterClash3m for k in
                          range(7 - len(filterClash3m))]
                if Filter[-1]:
                    count.append("3m")
                elif True in Filter and j == 0:
                    count.append("3m")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2a)] == filter2a for k in
                          range(7 - len(filter2a))]
                if Filter[-1]:
                    count.append("2a")
                elif True in Filter and j == 0:
                    count.append("2a")
                Filter = [self.map[i, j:j + 7].tolist()[k:k + len(filter2b)] == filter2b for k in
                          range(7 - len(filter2b))]
                if Filter[-1]:
                    count.append("2b")
                elif True in Filter and j == 0:
                    count.append("2b")

        self.mapsize-=2
        self.map=self.map[1:-1,1:-1]
        return count

    def f6(self,count,line):
        if line == filter4:
            count.append("4")
        if line == filterClash4a:
            count.append("4a")
        if line == filterClash4b:
            count.append("4b")
        if line == filterLive3b:
            count.append("3b")
        if line == filterLive3c:
            count.append("3c")
        if line == filterClash3d:
            count.append("3d")
        if line == filterClash3f:
            count.append("3f")
        if line == filterClash3g:
            count.append("3g")
        if line == filterClash3h:
            count.append("3h")
        if line == filterClash3i:
            count.append("3i")
        if line == filterClash3j:
            count.append("3j")

    def f5(self,count,line):
        if line == filter5:
            count.append("5")
        if line == filterClash4c:
            count.append("4c")
        if line == filterClash4d:
            count.append("4d")
        if line == filterClash4e:
            count.append("3e")
        if line == filterLive3a:
            count.append("3a")
        if line == filterClash3k:
            count.append("3k")
        if line == filterClash3l:
            count.append("3l")
        if line == filterClash3m:
            count.append("3m")
        if line == filter2b:
            count.append("2b")

    def getcount_(self):
        self.mapsize+=2
        map = np.ones((self.mapsize,self.mapsize))*2
        map[1:-1,1:-1]=self.map
        self.map=map
        count = []
        # gu=time.time()
        for i in range(self.mapsize):
            for j in range(self.mapsize):
                c=7
                if j<self.mapsize-c:
                    if i!=0 and i!=self.mapsize-1:
                        line = map[i,j:j+c].tolist()
                        if line==filterClash3e:
                            count.append("3e")
                        line = map[j:j+c,i].tolist()
                        if line==filterClash3e:
                            count.append("3e")
                    if i<self.mapsize-c:
                        line = map[range(i,i+c), range(j,j+c)].tolist()
                        if line==filterClash3e:
                            count.append("3e")
                        line = map[range(i, i + c), range(j + c-1, j - 1, -1)].tolist()
                        if line==filterClash3e:
                            count.append("3e")
                c=6
                if j<self.mapsize-c:
                    if i!=0 and i!=self.mapsize-1:
                        line = map[i,j:j+c].tolist()


                        self.f6(count,line)
                        line = map[j:j+c,i].tolist()
                        self.f6(count,line)
                    if i<self.mapsize-c:
                        line = map[range(i,i+c), range(j,j+c)].tolist()
                        self.f6(count, line)
                        line = map[range(i, i + c), range(j + c-1, j - 1, -1)].tolist()
                        self.f6(count, line)
                c=5
                if j<self.mapsize-c:
                    if i!=0 and i!=self.mapsize-1:
                        line = map[i,j:j+c].tolist()
                        self.f5(count,line)
                        line = map[j:j+c,i].tolist()
                        self.f5(count,line)
                    if i<self.mapsize-c:
                        line = map[range(i,i+c), range(j,j+c)].tolist()
                        self.f5(count, line)
                        line = map[range(i, i + c), range(j + c-1, j - 1, -1)].tolist()
                        self.f5(count, line)
                c=4
                if j<self.mapsize-c:
                    if i!=0 and i!=self.mapsize-1:
                        line = map[i,j:j+c].tolist()
                        if line==filter2a:
                            count.append("2a")
                        line = map[j:j+c,i].tolist()
                        if line==filter2a:
                            count.append("2a")
                    if i<self.mapsize-c:
                        line = map[range(i,i+c), range(j,j+c)].tolist()
                        if line == filter2a:
                            count.append("2a")
                        line = map[range(i, i + c), range(j + c-1, j - 1, -1)].tolist()
                        if line == filter2a:
                            count.append("2a")
        self.mapsize -= 2
        self.map = self.map[1:-1, 1:-1]
        return count

    def getScore(self,count,user):
        score = 0
        if '5' in count:
            score+=10000
        if '4' in count or np.sum(count=='4a')+np.sum(count=='4b')+np.sum(count=='4c')+np.sum(count=='4d')+np.sum(count=='4e')>1:
            score+=5000
        if np.sum(count=='4a')+np.sum(count=='4b')+np.sum(count=='4c')+np.sum(count=='4d')+np.sum(count=='4e')>=1 and np.sum(count=='3a')+np.sum(count=='3b')+np.sum(count=='3c')>=1:
            score+=2500
        if np.sum(count=='3a')+np.sum(count=='3b')+np.sum(count=='3c')>1:
            score+=2500

        for i in count:
            if i in ['4a', '4b', '4c', '4d', '4e', '3a', '3b', '3c']:
                score += 1000
            else:
                score += 50

        # else:
        #     if '5' in count:
        #         score+=10000
        #     if np.sum(count=='4a')+np.sum(count=='4b')+np.sum(count=='4c')+np.sum(count=='4d')+np.sum(count=='4e')>=1:
        #         score+=10000
        #     for i in count:
        #         if i in ['3a','3b','3c']:
        #             score+=100
        #         else:
        #             score+=50
        return score


map = np.zeros((15,15))
map[7,3:6]=2
print(map)
a = app(15)
def check(a,map):
    index = np.argwhere(map!=0)
    pos_score = a.pos_score
    for i in index:
        pos_score[i[0],i[1]]=-100000
    if len(index)>0:
        miny = np.min(index[:,0])
        maxy = np.max(index[:,0])
        minx = np.min(index[:,1])
        maxx = np.max(index[:,1])
        for i in np.argwhere(map[miny-1:maxy+2,minx-1:maxx+2]==0):
            map[miny-1+i[0],minx-1+i[1]]=1
            a.map = map
            d = a.getcount_()
            a1 = (map == 1)
            a2 = (map == 2)
            map[a1] = 2
            map[a2] = 1
            a.map = map
            e = a.getcount_()
            pos_score[miny-1+i[0],minx-1+i[1]]+=a.getScore(np.array(d), 0)-a.getScore(np.array(e),1)
            map[a1] = 1
            map[a2] = 2
            map[miny-1+i[0],minx-1+i[1]]=0

    else:
        pos_score = a.pos_score
    print(pos_score)
    print(np.argmax(pos_score)//15,np.argmax(pos_score)%15)
    return np.argmax(pos_score)//15,np.argmax(pos_score)%15

def getarea(map):
    index = np.argwhere(map!=0)
    if len(index)>0:
        miny = np.min(index[:,0])
        maxy = np.max(index[:,0])
        minx = np.min(index[:,1])
        maxx = np.max(index[:,1])
        miny = np.max([miny,0])
        maxy = np.min([maxy,map.shape[0]])
        minx = np.max([minx,0])
        maxx = np.min([maxx,map.shape[0]])
        return miny,maxy,minx,maxx

class point:
    def __init__(self,x,y,score):
        self.x = x
        self.y = y
        self.score = score
        self.point = None
        self.map = None

def minmax(a,map):
    xy0 = getarea(map)
    if xy0 == None:
        xy0 = (7-1,7+2,7-1,7+2)
    choosepoint=point(0,0,-1000000)
    for i in np.argwhere(map[xy0[0] - 1:xy0[1] + 2, xy0[2] - 1:xy0[3] + 2] == 0):
        map[xy0[0] - 1 +i[0], xy0[2] - 1 +i[1]] = 1
        xy1 = getarea(map)
        minscore = point(0,0,1000000)

        index = np.argwhere(map != 0)
        pos_score = a.pos_score.copy()
        for l in index:
            pos_score[l[0], l[1]] = -100000

        for j in np.argwhere(map[xy1[0] - 1:xy1[1] + 2, xy1[2] - 1:xy1[3] + 2] == 0):
            map[xy1[0] - 1 + j[0], xy1[2] - 1 + j[1]] = 2
            a.map = map
            d = a.getcount_()
            a1 = (map == 1)
            a2 = (map == 2)
            map[a1] = 2
            map[a2] = 1
            a.map = map
            e = a.getcount_()
            map[a1] = 1
            map[a2] = 2
            score = pos_score[xy1[0] - 1 + j[0], xy1[2] - 1 + j[1]] + a.getScore(np.array(d), 0) - a.getScore(
                np.array(e), 1)
            # minpoint.append(minscore)
            if minscore.score > score:
                minscore.score = score
                minscore.x = xy1[2] - 1 +j[1]
                minscore.y = xy1[0] - 1 +j[0]
                minscore.point = score
            map[xy1[0] - 1 +j[0], xy1[2] - 1 +j[1]] = 0
            if choosepoint.score >= score:
                break
        if choosepoint.score < minscore.score:
            choosepoint.score = minscore.score
            choosepoint.x = xy0[2] - 1 +i[1]
            choosepoint.y = xy0[0] - 1 +i[0]
            choosepoint.point = minscore
        map[xy0[0] - 1 +i[0], xy0[2] - 1 +i[1]] = 0
    return choosepoint

tt1=time.time()
cc =minmax(a,map)
print(time.time()-tt1)
print(cc.x,cc.y)
print(cc.point.x,cc.point.y)

# mm = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]])
# print(check(a,mm))
# a