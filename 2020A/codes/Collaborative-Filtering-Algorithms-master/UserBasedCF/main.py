import random
import math
class UserBasedCF:
    def __init__(self,datafile = None):
        self.datafile = datafile
        self.readData()
        self.splitData(3,47)
    def readData(self,datafile = None):
        """
        read the data from the data file which is a data set
        """
        self.datafile = datafile or self.datafile
        self.data = []
        for line in open(self.datafile):
            userid,itemid,record,mtime = line.split("\t")
            self.data.append((userid,itemid,int(record)))
    def splitData(self,k,seed,data = None,M = 8):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        self.testdata = {}
        self.traindata = {}
        data = data or self.data
        random.seed(seed)
        for user,item,record in self.data:
            if random.randint(0,M) == k:
                self.testdata.setdefault(user,{})
                #testdata[user]={}
                self.testdata[user][item] = record
            else:
                self.traindata.setdefault(user,{})
                self.traindata[user][item] = record
    def userSimilarity(self,train = None):
        """
        one method of getting user similarity matrix
        """
        train = train or self.traindata
        self.userSim = dict()
        for u in train.keys():
            for v in train.keys():
                if u == v:
                    continue
                self.userSim.setdefault(u,{})
                self.userSim[u][v] = len(set(train[u].keys()) & set(train[v].keys()))#�������ƶ�
                self.userSim[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)

    def userSimilarityBest(self,train = None):
        """
        the other method of getting user similarity which is better than above
        you can get the method on page 46
        In this experiment, we use this method
        """
        train = train or self.traindata
        self.userSimBest = dict()
        item_users = dict()
        #build inverse table for item_users
        for u,item in train.items():
            for i in item.keys():
                item_users.setdefault(i,set())
                item_users[i].add(u)
        #calculate co-rated items between users
        user_item_count = dict()
        count = dict()
        for item,users in item_users.items():
            for u in users:
                user_item_count.setdefault(u,0)
                user_item_count[u] += 1
                for v in users:
                    if u == v:
                        continue
                    count.setdefault(u,{})
                    count[u].setdefault(v,0)
                    count[u][v] += 1
        #calculate finial similarity matrix W
        for u,related_users in count.items():
            self.userSimBest.setdefault(u,dict())
            for v,cuv in related_users.items():
                self.userSimBest[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v] *1.0)
                
    def userSimilarityBest_IIF(self,train = None):
        """
        the other method of getting user similarity which is better than above
        you can get the method on page 46
        In this experiment, we use this method
        """
        train = train or self.traindata
        self.userSimBest = dict()
        item_users = dict()
        #build inverse table for item_users
        for u,item in train.items():
            for i in item.keys():
                item_users.setdefault(i,set())
                item_users[i].add(u)
        #calculate co-rated items between users
        user_item_count = dict()
        count = dict()
        for item,users in item_users.items():
            for u in users:
                user_item_count.setdefault(u,0)
                user_item_count[u] += 1
                for v in users:
                    if u == v:
                        continue
                    count.setdefault(u,{})
                    count[u].setdefault(v,0)
                    count[u][v] += 1/math.log(1+len(users))
        #calculate finial similarity matrix W
        for u,related_users in count.items():
            self.userSimBest.setdefault(u,dict())
            for v,cuv in related_users.items():
                self.userSimBest[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v] *1.0)

    def recommend(self,user,train = None,k = 8,nitem = 40):
            train = train or self.traindata
            rank = dict()
            interacted_items = train.get(user,{})
            for v,wuv in sorted(self.userSimBest[user].items(),key = lambda x:x[1],reverse = True)[0:k]:
                for i in train[v].keys():
                    if i in interacted_items:
                        continue
                    rank.setdefault(i,0)
                    rank[i] += wuv
            return dict(sorted(rank.items(),key = lambda x:x[1],reverse = True)[0:nitem])
    def recallAndPrecision(self,train = None,test = None,k = 8,nitem = 10):
        """
        get the racall and precision, the method you want to know is listed
        in the page 43
        """
        train = train or self.traindata
        test = test or self.testdata
        hit = 0
        recall = 0
        precision = 0
        for user in train.keys():
            tu = test.get(user,{})
            rank = self.recommend(user,train = train,k = k, nitem = nitem)
            for item,ratings in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += nitem
        return (hit / (recall * 1.0),hit / (precision * 1.0))

    def coverage(self,train = None,test = None,k = 8,nitem = 10):
        train = train or self.traindata
        test = test or self.testdata
        recommend_items = set()
        all_items = set()
        for user in train.keys():
            for item in train[user].keys():
                all_items.add(item)
            rank = self.recommend(user, train, k = k,nitem = nitem)
            for item,ratings in rank.items():
                recommend_items.add(item)
        return len(recommend_items) / (len(all_items) * 1.0)

    def popularity(self,train = None,test = None,k = 8,nitem = 10):
        """
        get the popularity
        the algorithm on page 44
        """
        train = train or self.traindata
        test = test or self.testdata
        item_popularity = dict()
        for user, items in train.items():
            for item in items.keys():
                item_popularity.setdefault(item,0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in train.keys():
            rank = self.recommend(user,train,k = k,nitem = nitem)
            for item,ratings in rank.items():
                ret += math.log(1+item_popularity[item])
                n += 1
        return ret / (n * 1.0)

def testRecommend():
    ubcf = UserBasedCF('data//u.data')
    ubcf.readData()
    ubcf.splitData(4,100)
    ubcf.userSimilarity()
    user = "345"
    rank = ubcf.recommend(user,k = 3)
    for i,rvi in rank.items():
        items = ubcf.testdata.get(user,{})
        record = items.get(i,0)
        print ("%5s:%.4f--%.4f"%(i,rvi,record))

def testUserBasedCF():
    cf = UserBasedCF('data//u.data')
    cf.userSimilarityBest()
    print ("%.13s%3s%20s%20s%20s%20s"%("             ",'K',"recall",'precision','coverage','popularity'))
    for k in [5,10,20,40,80,160]:
        recall,precision = cf.recallAndPrecision(k = k)
        coverage = cf.coverage(k = k)
        popularity = cf.popularity(k = k)
        print ("%.13s%3d%19.3f%%%19.3f%%%19.3f%%%20.3f" % ("UserCF       ",k,recall * 100,precision * 100,coverage * 100,popularity))

def testUserBasedCF_IIF():
    cf = UserBasedCF('data//u.data')
    cf.userSimilarityBest()
    print ("%.13s%3s%20s%20s%20s%20s"%("             ",'K',"recall",'precision','coverage','popularity'))
    for k in [80]:
        recall,precision = cf.recallAndPrecision(k = k)
        coverage = cf.coverage(k = k)
        popularity = cf.popularity(k = k)
        print ("%.13s%3d%19.3f%%%19.3f%%%19.3f%%%20.3f" % ("UserCF       ",k,recall * 100,precision * 100,coverage * 100,popularity))
    cf.userSimilarityBest_IIF()
    for k in [80]:
        recall,precision = cf.recallAndPrecision(k = k)
        coverage = cf.coverage(k = k)
        popularity = cf.popularity(k = k)
        print ("%.13s%3d%19.3f%%%19.3f%%%19.3f%%%20.3f" % ("UserCF_IIF   ",k,recall * 100,precision * 100,coverage * 100,popularity))

if __name__ == "__main__":
    x = int(input("Please enter an integer(1,UserCF;2,UserCF_IIF):"))
    if x == 1:
        testUserBasedCF()
    elif x == 2:
        testUserBasedCF_IIF()
