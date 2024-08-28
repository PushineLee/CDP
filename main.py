import torch

from PIL import Image
import numpy as np
import sys
import random
import os
import math
import argparse
import matplotlib.pyplot as plt
from chr.histogram import Histogram
from chr.grey_boxes import HistogramAccumulator
from scipy.stats.mstats import mquantiles
from scipy import stats


parser = argparse.ArgumentParser("C3D_backbone")
parser.add_argument("--dataset",default="avec2013")
parser.add_argument("--method",default="CDP")
args = parser.parse_args()

def split_reg(val_scores,val_labels,test_score,test_label):
    PICP_90 = []
    MPIW_90 = []
    PICP_95 = []
    MPIW_95 = []

    cal_scores, val_scores = val_scores, test_score
    cal_labels, val_labels = val_labels, test_label
    n = cal_scores.shape[0]
     
    conformal_score = abs(cal_labels - cal_scores)
    intervals = np.quantile(conformal_score,[np.ceil((n+1)*0.9)/n,np.ceil((n+1)*0.95)/n])

    val_95_upper = val_scores + intervals[1]
    val_95_lower = val_scores - intervals[1]
    val_95_lower = np.clip(val_95_lower,a_min=0,a_max=64)
    val_90_upper = val_scores + intervals[0]
    val_90_lower = val_scores - intervals[0]
    val_90_lower = np.clip(val_90_lower,a_min=0,a_max=64)

        

    PICP_95.append(((val_95_upper>=val_labels) * (val_95_lower<=val_labels)).mean())
    MPIW_95.append((val_95_upper - val_95_lower).mean())

    PICP_90.append(((val_90_upper>=val_labels) * (val_90_lower<=val_labels)).mean())
    MPIW_90.append((val_90_upper - val_90_lower).mean())


    # idx = np.argsort(val_labels)
    # val_labels = np.sort(val_labels)

    plt.figure(figsize = (50,10))
    plt.plot(val_95_upper,label = "95 upper")
    plt.plot(val_labels,label = "label")
    plt.plot(val_scores,label = "pred")
    plt.plot(val_95_lower,label = "95 lower")
    plt.legend()
    plt.grid()
    plt.title("PICP_95:{:.4f}, MPIW_95:{:.4f}".format(PICP_95[0],MPIW_95[0]))
    plt.savefig("split_reg_95_14")
    plt.close()

    plt.figure(figsize = (30,10))
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.plot(val_90_upper,label = "$y_{hi}$")
    plt.plot(val_labels,label = "$y$")
    plt.plot(val_scores,label = "$\hat{y}$")
    plt.plot(val_90_lower,label = "$y_{lo}$")
    plt.legend(fontsize = 15)
    plt.grid()
    plt.xlabel("$i$",fontdict={'size':25})
    plt.ylabel("depression score",fontdict={'size':20})
    # plt.title("PICP_90:{:.4f}, MPIW_90:{:.4f}".format(PICP_90[0],MPIW_90[0]))
    plt.savefig("split_reg_90_14")
    plt.close()    
    print("PICP_95:{:.4f}, MPIW_95:{:.4f}, PICP_90:{:.4f}, MPIW_90:{:.4f}".format(PICP_95[0],MPIW_95[0], PICP_90[0], MPIW_90[0]))
    
    return val_labels, val_90_lower,val_90_upper        

def fair_split_reg(val_scores,val_labels):


    cal_scores = val_scores
    cal_labels = val_labels
    n = cal_scores.shape[0]
     
    conformal_score = abs(cal_labels - cal_scores)
    intervals = np.quantile(conformal_score,[np.ceil((n+1)*0.9)/n,np.ceil((n+1)*0.95)/n])

    return intervals        

def smallestSubWithSum(arr, alpha):
    nums, freq = np.histogram(arr,bins = range(arr.min().astype(int),arr.max().astype(int) + 1))
    n = len(nums)
    if n==0:
        return arr[0],arr[0]
    x = (alpha * (n + 1) / n) * nums.sum()
    x = np.clip(x,a_min=0,a_max=nums.sum())

    end_init = 0
    start_max = n
    # Initialize optimal solution
    start_best = 0
    end_best = n
    min_len = n + 1

    # Initialize starting index
    start = 0

    # Initialize current sum
    curr_sum = np.sum(nums[start:end_init])

    for end in range(end_init, n):
        curr_sum += nums[end]
        while (curr_sum >= x) and (start <= end) and (start <= start_max):
            if (end - start + 1 < min_len):
                min_len = end - start + 1
                start_best = start
                end_best = end

            curr_sum -= nums[start]
            start += 1

    return freq[start_best], freq[end_best + 1]

def acc_reg(val_scores,val_labels,test_score,test_label):
    PICP_90 = []
    MPIW_90 = []
    PICP_95 = []
    MPIW_95 = []
    score_range_min = min(val_scores.min(),test_score.min())
    score_range_max = max(val_scores.max(),test_score.max())

    cal_scores, val_scores = val_scores, test_score
    cal_labels, val_labels = val_labels, test_label

    # n = cal_scores.shape[0]
     

    score_edge = np.linspace(score_range_min,score_range_max,11)
     
    score_dict = {}

    for i in range(10):
        score_invals_lower = score_edge[i]
        score_invals_upper = score_edge[i+1]

        for j in range(len(cal_scores)):
            if  (cal_scores[j]>=score_invals_lower) and (cal_scores[j]<=score_invals_upper):
                if (score_edge[i] + score_edge[i+1])/2. not in score_dict:
                    score_dict[(score_edge[i] + score_edge[i+1])/2.] = [cal_labels[j]]
                else:
                    score_dict[(score_edge[i] + score_edge[i+1])/2.] += [cal_labels[j]]

        # print(score_dict)
    score_q_dict = {}
    print(score_dict)
    for i in score_dict.keys():
        l_95,h_95 = smallestSubWithSum(np.array(score_dict[i]),0.95)
        l_90,h_90 = smallestSubWithSum(np.array(score_dict[i]),0.90)
        score_q_dict[i] = [l_95,l_90,h_90,h_95]

        # print(count)
        # print(score_q_dict)
        # exit(0)
    val_95_upper = []
    val_95_lower = []
    val_90_upper = []
    val_90_lower = []
    for i in range(len(val_scores)):
        for j in range(10):
            if (val_scores[i]>=score_edge[j]) and (val_scores[i]<=score_edge[j+1]):
                val_95_upper.append(score_q_dict[(score_edge[j] + score_edge[j+1])/2.][-1])
                val_95_lower.append(score_q_dict[(score_edge[j] + score_edge[j+1])/2.][0])
                val_90_upper.append(score_q_dict[(score_edge[j] + score_edge[j+1])/2.][-2])
                val_90_lower.append(score_q_dict[(score_edge[j] + score_edge[j+1])/2.][1])

    val_95_upper = np.array(val_95_upper)
    val_95_lower = np.array(val_95_lower)
    val_90_upper = np.array(val_90_upper)
    val_90_lower = np.array(val_90_lower)
        # print(val_95_upper)
        # print(val_labels)


    PICP_95.append(((val_95_upper>=val_labels) * (val_95_lower<=val_labels)).mean())
    MPIW_95.append((val_95_upper - val_95_lower).mean())

    PICP_90.append(((val_90_upper>=val_labels) * (val_90_lower<=val_labels)).mean())
    MPIW_90.append((val_90_upper - val_90_lower).mean())


    idx = np.argsort(val_scores)
    val_scores = np.sort(val_scores)

    plt.figure(figsize = (70,10))
    plt.plot(val_95_upper[idx],label = "95 upper")
    plt.scatter(range(len(val_labels)),val_labels[idx],label = "label",c = "black")
    plt.plot(val_scores,label = "pred")
    plt.plot(val_95_lower[idx],label = "95 lower")
            # print(score_edge)
            # for values in score_edge:
            #     plt.axhline(values,c = "pink")
    plt.legend()
    plt.grid()
    plt.title("PICP_95:{:.4f}, MPIW_95:{:.4f}".format(PICP_95[0],MPIW_95[0]))
    plt.savefig("group_reg_95_inverse_new_14")
    plt.close()

    plt.figure(figsize = (70,10))
    plt.plot(val_90_upper[idx],label = "90 upper")
    plt.scatter(range(len(val_labels)),val_labels[idx],label = "label",c = "black")
    plt.plot(val_scores,label = "pred")
    plt.plot(val_90_lower[idx],label = "90 lower")
            # for values in score_edge:
            #     plt.axhline(values,c = "gold")
    plt.legend()
    plt.grid()
    plt.title("PICP_90:{:.4f}, MPIW_90:{:.4f}".format(PICP_90[0],MPIW_90[0]))
    plt.savefig("group_reg_90_inverse_new_14")
    plt.close()            
            # print(np.where(~(val_90_upper>=val_labels) * (val_90_lower<=val_labels)))
            # exit(0)
    print("PICP_95:{:.4f}, MPIW_95:{:.4f}, PICP_90:{:.4f}, MPIW_90:{:.4f}".format(np.array(PICP_95).mean(),np.array(MPIW_95).mean(),np.array(PICP_90).mean(),np.array(MPIW_90).mean()))

def QR(test_score,test_label):
    upper_95 = test_score[:,94]
    lower_5 = test_score[:,4]
    upper_975 = (test_score[:,97] + test_score[:,96])/2.
    lower_025 = (test_score[:,1] + test_score[:,2])/2.

    upper_99 = test_score[:,-1]
    lower_1 = test_score[:,0]


    PICP_95 = ((upper_975>=test_label) * (lower_025<=test_label)).mean()
    PICP_90 = ((upper_95>=test_label) * (lower_5<=test_label)).mean()
    PICP_98 = ((upper_99>=test_label) * (lower_1<=test_label)).mean()

    MPIW_95 = (upper_975 - lower_025).mean()
    MPIW_90 = (upper_95 - lower_5).mean()
    MPIW_98 = (upper_99 - lower_1).mean()

    plt.figure(figsize = (30,10))
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.plot(upper_95,label = "$y_{hi}$")
    plt.plot(test_label,label = "$y$")
    plt.plot(test_score[:,49],label = "$\hat{y}$")
    plt.plot(lower_5,label = "$y_{lo}$")
    plt.xlabel("$i$",fontdict={'size':25})
    plt.ylabel("depression score",fontdict={'size':20})
    plt.legend(fontsize = 15)
    plt.grid()
    # plt.title("PICP_90:{:.4f}, MPIW_90:{:.4f}".format(PICP_90[0],MPIW_90[0]))
    plt.savefig("QR_90_14")
    plt.close()   

    print("PICP_95:{:.4f}, MPIW_95:{:.4f}, PICP_90:{:.4f}, MPIW_90:{:.4f}".format(PICP_95,MPIW_95, PICP_90, MPIW_90))
    print("PICP_98:{:.4f}, MPIW_98:{:.4f}".format(PICP_98,MPIW_98))
    return test_label, lower_5, upper_95

def CQR(val_scores,val_labels,test_score,test_label):
    PICP_90 = []
    MPIW_90 = []
    PICP_95 = []
    MPIW_95 = []


    cal_scores, val_scores = val_scores, test_score
    cal_labels, val_labels = val_labels, test_label

    upper_95 = cal_scores[:,94]
    lower_5 = cal_scores[:,4]
    upper_975 = (cal_scores[:,97] + cal_scores[:,96])/2.
    lower_025 = (cal_scores[:,1] + cal_scores[:,2])/2.
    # print(np.concatenate(((lower_5 - cal_labels),(cal_labels - upper_95)),axis = 0))
    E_90 = np.amax(np.concatenate(((lower_5 - cal_labels)[None,:],(cal_labels - upper_95)[None,:]),axis = 0),axis = 0)
 
    E_95 = np.amax(np.concatenate(((lower_025 - cal_labels)[None,:],(cal_labels - upper_975)[None,:]),axis = 0),axis = 0)
    n = cal_scores.shape[0]

    intervals_90 = np.quantile(E_90,np.ceil((n+1)*0.9)/n)
    intervals_95 = np.quantile(E_95,np.ceil((n+1)*0.95)/n)

    val_95_upper = (val_scores[:,97] + val_scores[:,96])/2. + intervals_95
    val_95_lower = (val_scores[:,1] + val_scores[:,2])/2. - intervals_95
    # val_95_upper = val_scores[:,94]  + intervals_95
    # val_95_lower = val_scores[:,4] - intervals_95
    val_95_lower = np.clip(val_95_lower,a_min=0,a_max=64)
    val_90_upper = val_scores[:,94] + intervals_90
    val_90_lower = val_scores[:,4] - intervals_90
    val_90_lower = np.clip(val_90_lower,a_min=0,a_max=64)

        

    PICP_95.append(((val_95_upper>=val_labels) * (val_95_lower<=val_labels)).mean())
    MPIW_95.append((val_95_upper - val_95_lower).mean())

    PICP_90.append(((val_90_upper>=val_labels) * (val_90_lower<=val_labels)).mean())
    MPIW_90.append((val_90_upper - val_90_lower).mean())


    # idx = np.argsort(val_labels)
    # val_labels = np.sort(val_labels)

    plt.figure(figsize = (70,10))
    plt.plot(val_95_upper,label = "95 upper")
    plt.plot(val_labels,label = "label")
    plt.plot(val_scores[:,49],label = "pred")
    plt.plot(val_95_lower,label = "95 lower")
    plt.legend()
    plt.grid()
    plt.title("PICP_95:{:.4f}, MPIW_95:{:.4f}".format(PICP_95[0],MPIW_95[0]))
    plt.savefig("cqr_reg_95_14")
    plt.close()

    plt.figure(figsize = (30,10))
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.plot(val_90_upper,label = "$y_{hi}$")
    plt.plot(val_labels,label = "$y$")
    plt.plot(val_scores[:,49],label = "$\hat{y}$")
    plt.plot(val_90_lower,label = "$y_{lo}$")
    plt.xlabel("$i$",fontdict={'size':25})
    plt.ylabel("depression score",fontdict={'size':20})
    plt.legend(fontsize = 15)
    plt.grid()
    # plt.title("PICP_90:{:.4f}, MPIW_90:{:.4f}".format(PICP_90[0],MPIW_90[0]))
    plt.savefig("cqr_reg_90_14")
    plt.close()     

    print("PICP_95:{:.4f}, MPIW_95:{:.4f}, PICP_90:{:.4f}, MPIW_90:{:.4f}".format(np.array(PICP_95).mean(),np.array(MPIW_95).mean(),np.array(PICP_90).mean(),np.array(MPIW_90).mean()))
    return test_label, val_90_lower, val_90_upper

def CHR(val_scores,val_labels,test_score,test_label):
    # val_scores = np.clip(val_scores,a_min=0,a_max=64)
    # test_score = np.clip(test_score,a_min=0,a_max=64)
    grid_histogram = np.arange(0,64,0.1)
    hist = Histogram(grid_quantiles, grid_histogram)

    print(val_scores.shape,test_score.shape)
    # val_scores = np.concatenate((np.zeros((val_scores.shape[0],1)),val_scores),axis = 1)
    # test_score = np.concatenate((np.zeros((test_score.shape[0],1)),test_score),axis = 1)

    val_scores = np.concatenate((val_scores,64 * np.ones((val_scores.shape[0],5))),axis = 1)
    test_score = np.concatenate((test_score,64 * np.ones((test_score.shape[0],5))),axis = 1)

    histogram_val = hist.compute_histogram(val_scores, 0, 64, 0.1)

    # print(val_labels[100])
    for i in range(val_scores.shape[0]):
        plt.step(grid_histogram, histogram_val[i], where='pre', color='black')
        plt.axvline(val_labels[i])
        plt.savefig("demo/CHR_demo_single_" + str(i))
        plt.close()
    

    # Desired level
    alpha = 0.1
    # Initialize density accumulator (grey-box)
    accumulator = HistogramAccumulator(histogram_val, grid_histogram, alpha=0.1, delta_alpha=0.001)

    # Generate noise for randomization
    epsilon = np.random.uniform(low=0.0, high=1.0, size=val_scores.shape[0])
    # S, _ = accumulator.predict_intervals(alpha, epsilon=epsilon)
    

    # for i in range(val_scores.shape[0]):

    #     S_int = [np.arange(S[i][0],S[i][1]+1) for i in range(len(S))]
    #     idx = S_int[i]
    #     z = np.zeros(len(grid_histogram),)
    #     z[idx] = histogram_val[i,idx]
        
    #     plt.step(grid_histogram, histogram_val[i], where='pre', color='black')
    #     plt.fill_between(grid_histogram, z, step="pre", alpha=0.4, color='gray')
    #     plt.axvline(val_labels[i])
    #     plt.title(str((histogram_val[i][S[i][0]:S[i][1]]).sum()))
    #     plt.savefig("demo01/CHR_demo_single_" + str(i))
    #     plt.close()
    
    # S_int = [np.arange(S[100][0],S[100][1]+1) for i in range(len(S))]
    # plt.step(grid_histogram, histogram_val[100], where='pre', color='black')
    # idx = S_int[100]
    # z = np.zeros(len(grid_histogram),)
    # z[idx] = histogram_val[100,idx]
    # print(val_labels[100])
    # plt.fill_between(grid_histogram, z, step="pre", alpha=0.4, color='gray')
    # plt.savefig("CHR_demo_s")
    # Compute randomized sets
    scores = accumulator.calibrate_intervals(val_labels.astype(np.float32), epsilon=epsilon)
   
    level_adjusted = (1.0-alpha)*(1.0+1.0/float(val_scores.shape[0]))
    plt.hist(scores)
    plt.savefig("CHR_demo_score")

  
    # print(mquantiles(scores, prob=level_adjusted)[0])
    calibrated_alpha = np.round(1.0-mquantiles(scores, prob=level_adjusted)[0],4)

        
        # Print message
    print("Calibrated alpha (nominal level: {}): {:.3f}.".format(alpha, calibrated_alpha))

    histogram_test = hist.compute_histogram(test_score, 0,64, 0.1)
    test_accumulator = HistogramAccumulator(histogram_test, grid_histogram, alpha=0.1, delta_alpha=0.01)
    epsilon = np.random.uniform(low=0.0, high=1.0, size=test_score.shape[0])
    S, bands = test_accumulator.predict_intervals(calibrated_alpha, epsilon=epsilon)
    MPIW = (bands[:,1] - bands[:,0]).mean()
    PICP = ((test_label>=bands[:,0]).astype(float)*(test_label<=bands[:,1]).astype(float)).mean()
    
    print("PICP_90:{:.4f}, MPIW_90:{:.4f}".format(PICP,MPIW))

def new_acc_reg(val_scores,val_labels,test_score,test_label):
    PICP_90 = []
    MPIW_90 = []
    PICP_95 = []
    MPIW_95 = []

    nums = 15
    score_range_min = min(val_scores.min(),test_score.min())
    score_range_max = max(val_scores.max(),test_score.max())
    
    cal_scores, val_scores = val_scores, test_score
    cal_labels, val_labels = val_labels, test_label


    # cal_scores, val_scores = val_scores[1::2], val_scores[::2]
    # cal_labels, val_labels = val_labels[1::2], val_labels[::2]

    # n = cal_scores.shape[0]
    
    score_edge = np.linspace(score_range_min,score_range_max,nums)
    # score_edge = np.array([-0.5,1,5,30,42,50,64])
    # nums = len(score_edge)
     
    score_dict = {}

    for i in range(nums-1):
        score_invals_lower = score_edge[i]
        score_invals_upper = score_edge[i+1]

        for j in range(len(cal_scores)):
            if  (cal_scores[j]>=score_invals_lower) and (cal_scores[j]<=score_invals_upper):
                if (score_edge[i] + score_edge[i+1])/2 not in score_dict:
                    score_dict[(score_edge[i] + score_edge[i+1])/2.] = [cal_labels[j]]
                else:
                    score_dict[(score_edge[i] + score_edge[i+1])/2.] += [cal_labels[j]]

    
    score_q_dict = {}

    width = 0.
    for i in score_dict.keys():
        l_95,h_95 = smallestSubWithSum(np.array(score_dict[i]),0.95)
        l_90,h_90 = smallestSubWithSum(np.array(score_dict[i]),0.90)
        score_q_dict[i] = [l_95,l_90,h_90,h_95]
        width += (h_90 - l_90)
    width /= (nums-1)
    print(width)
        # print(count)
        # print(score_q_dict)
        # exit(0)
    print(score_q_dict)
    val_95_upper = []
    val_95_lower = []
    val_90_upper = []
    val_90_lower = []
    for i in range(len(val_scores)):
        for j in range(nums-1):
            if (val_scores[i]>=score_edge[j]) and (val_scores[i]<=score_edge[j+1]):
                val_95_upper.append(score_q_dict[(score_edge[j] + score_edge[j+1])/2.][-1])
                val_95_lower.append(score_q_dict[(score_edge[j] + score_edge[j+1])/2.][0])
                val_90_upper.append(score_q_dict[(score_edge[j] + score_edge[j+1])/2.][-2])
                val_90_lower.append(score_q_dict[(score_edge[j] + score_edge[j+1])/2.][1])

    val_95_upper = np.array(val_95_upper)
    val_95_lower = np.array(val_95_lower)
    val_90_upper = np.array(val_90_upper)
    val_90_lower = np.array(val_90_lower)
        # print(val_95_upper)
        # print(val_labels)


    PICP_95.append(((val_95_upper>=val_labels) * (val_95_lower<=val_labels)).mean())
    MPIW_95.append((val_95_upper - val_95_lower).mean())

    PICP_90.append(((val_90_upper>=val_labels) * (val_90_lower<=val_labels)).mean())
    MPIW_90.append((val_90_upper - val_90_lower).mean())


    idx = np.argsort(val_scores)
    val_scores = np.sort(val_scores)

    plt.figure(figsize = (70,10))
    plt.plot(val_95_upper,label = "95 upper")
    plt.scatter(range(len(val_labels)),val_labels,label = "label",c = "black")
    plt.plot(val_scores,label = "pred")
    plt.plot(val_95_lower,label = "95 lower")
            # print(score_edge)
            # for values in score_edge:
            #     plt.axhline(values,c = "pink")
    plt.legend()
    plt.grid()
    plt.title("PICP_95:{:.4f}, MPIW_95:{:.4f}".format(PICP_95[0],MPIW_95[0]))
    plt.savefig("group_reg_95_inverse_new_14")
    plt.close()

    plt.figure(figsize = (30,10))
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.plot(val_90_upper[idx],label = "$y_{hi}$",c = "#1f77b4")
    plt.scatter(range(len(val_labels)),val_labels[idx],label = "$y$",c = "black")
    # plt.plot(val_labels[idx],label = "$y$")s
    plt.plot(val_scores,label = "$\hat{y}$",c = "green")
    plt.plot(val_90_lower[idx],label = "$y_{lo}$",c = "red")
            # for values in score_edge:
            #     plt.axhline(values,c = "gold")
    plt.xlabel("$i$",fontdict={'size':25})
    plt.ylabel("depression score",fontdict={'size':20})
    plt.legend(fontsize = 15)
    plt.grid()
    # plt.title("PICP_90:{:.4f}, MPIW_90:{:.4f}".format(PICP_90[0],MPIW_90[0]))
    plt.savefig("group_reg_90_inverse_new_14")
    plt.close()            
            # print(np.where(~(val_90_upper>=val_labels) * (val_90_lower<=val_labels)))
            # exit(0)
    print("PICP_95:{:.4f}, MPIW_95:{:.4f}, PICP_90:{:.4f}, MPIW_90:{:.4f}".format(np.array(PICP_95).mean(),np.array(MPIW_95).mean(),np.array(PICP_90).mean(),np.array(MPIW_90).mean()))
    return val_labels, val_90_lower, val_90_upper

def calulate_SSC(test_label_list, y_lo, y_hi):
    test_label = np.array(test_label_list)

    idx_bin1 = ((test_label >=0) * (test_label <=13))
    idx_bin2 = ((test_label >=14) * (test_label <=19))
    idx_bin3 = ((test_label >=20) * (test_label <=28))
    idx_bin4 = test_label >=29

    PICP_bin1 = ((y_hi[idx_bin1]>=test_label[idx_bin1]) * (y_lo[idx_bin1]<=test_label[idx_bin1])).mean()
    PICP_bin2 = ((y_hi[idx_bin2]>=test_label[idx_bin2]) * (y_lo[idx_bin2]<=test_label[idx_bin2])).mean()
    PICP_bin3 = ((y_hi[idx_bin3]>=test_label[idx_bin3]) * (y_lo[idx_bin3]<=test_label[idx_bin3])).mean()
    PICP_bin4 = ((y_hi[idx_bin4]>=test_label[idx_bin4]) * (y_lo[idx_bin4]<=test_label[idx_bin4])).mean()

    return min([PICP_bin1,PICP_bin2,PICP_bin3,PICP_bin4])
     
if __name__ == '__main__':

    grid_quantiles = np.arange(0.01,1.05,0.01)
    nums = args.dataset[4:]
    val_outs,val_prediction,val_labels = np.load("AVEC_" + nums + "_QR_val_all.npz")["val_outs"],np.load("AVEC_" + nums + "_QR_val_all.npz")["val_prediction"],np.load("AVEC_" + nums + "_QR_val_all.npz")["val_labels"]
    test_outs,test_prediction,test_labels = np.load("AVEC_" + nums + "_QR_test_all.npz")["test_outs"],np.load("AVEC_" + nums + "_QR_test_all.npz")["test_prediction"],np.load("AVEC_" + nums + "_QR_test_all.npz")["test_labels"]
 

############################################################################################
    if args.method == "CDP":
        target,y_lo,y_hi = split_reg(val_prediction,val_labels,test_prediction,test_labels)
    elif args.method == "QR":
        target,y_lo,y_hi = QR(test_outs,test_labels)
    elif args.method == "CQR":
        target,y_lo,y_hi = CQR(val_outs,val_labels,test_outs,test_labels)
    else:
        target,y_lo,y_hi = new_acc_reg(val_prediction,val_labels,test_prediction,test_labels)
    ssc = calulate_SSC(target,y_lo,y_hi)
    print(ssc)
    
    # CHR(val_outs,val_labels,test_outs,test_labels)

