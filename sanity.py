import pickle as pkl
import sys
from tqdm import tqdm

def train_val_test(datatuples,splits,split_num=0,validation=0.5,rows=None):
    """
    Builds train/val/test indexes sets from tuple list and split list
    Validation set at 0.5 if n split is 5 gives an 80:10:10 split as usually used.
    """
    train,test = [],[]

    for idx,split in tqdm(enumerate(splits),total=len(splits),desc="Building train/test of split #{}".format(split_num)):
        if split == split_num:
            test.append(idx)
        else:
            train.append(idx)

    if len(test) <= 0:
            raise IndexError("Test set is empty - split {} probably doesn't exist".format(split_num))

    if rows and type(rows) is tuple:
        rows = {v:k for k,v in enumerate(rows)}
        print("Tuples rows are the following:")
        print(rows)

    if validation > 0:

        if 0 < validation < 1:
            val_len = int(validation * len(test))

        validation = test[-val_len:]
        test = test[:-val_len]

    else:
        validation = []

    idxs = (train,test,validation)
    iters = idxs

    return (datatuples,iters)

def means(data,train,test):
    '''
    Sanity check:
    data is a list of tuples [(user,item,text,rating)]
    train/test idx are given
    the goal is to predict the rating

    1) predict only the mean rating from train exemples
    2) predict the mean + mean user diff + mean item diff

    => Should match with HFTs ...

    '''
    print("intersect is empty? :")
    print(set(train).intersection(set(test)))

    all_r = [ int(data[idx][3]) for idx in train]
    mean = sum(all_r)/len(all_r)
    print(f"Train mean {mean}")

    mse_mean = [ (int(data[idx][3])-mean)**2 for idx in test]
    print(f"error -> only mean {sum(mse_mean)/len(mse_mean)}")

    mean_u_i = {}

    for idx in train:
        #computing diff means u & i
        (user,item,_,rating) = data[idx]
        mean_u_i.setdefault(user,[]).append(rating - mean)
        mean_u_i.setdefault(item,[]).append(rating - mean)

    mean_u_i = {k: sum(v)/len(v) for k,v in mean_u_i.items()}

    pred_func = lambda tuple: mean + mean_u_i.get(tuple[0],0) + mean_u_i.get(tuple[1],0) 

    mse_mean = [ (int(data[idx][3]) - pred_func(data[idx]))**2 for idx in test]
    print(f"error -> mean + udiff + idiff {sum(mse_mean)/len(mse_mean)}")    
    


datadict = pkl.load(open(str(sys.argv[-2]),"rb"))
split = int(sys.argv[-1])
print(split)
data_tl,(trainit,valit,testit) = train_val_test(datadict["data"],datadict["splits"],int(split),validation=0.5,rows=datadict["rows"])
means(data_tl,trainit,testit)
