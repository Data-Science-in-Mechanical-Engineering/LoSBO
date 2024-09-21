import json
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.utils.evaluation_utils import (evaluate_all,
                                        evaluate_multiple_performance,
                                        evaluate_performance)
from src.utils.visualization_utils import initialize_plot, set_size

color_json = "src/utils/RWTHcolors.json"
with open(color_json) as json_file:
    c = json.load(json_file)

c_real_beta = c["orange100"]
c_losbo = c["blau100"]
c_ucb_losbo = c["violett100"]

def evaluate_experiment_2_3():
    '''
    function to evaluate experiment 2 and 3
    '''
    res_df1 = evaluate_all("results/20240201_results/experiment_2_safeopt/onb_function", "2")
    print("beta=2")
    print(res_df1.not_started.sum()/(1000000))
    print(res_df1["safety_violations"].max()/10000)
    print(res_df1["safety_violations"].sum()/1000000)

    res_df2 = evaluate_all("results/20240201_results/experiment_31_real_beta/onb_function", "31")
    print("B=2.5")
    print(res_df2.not_started.sum()/(1000000))
    print(res_df2["safety_violations"].max()/10000)
    print(res_df2["safety_violations"].sum()/1000000)

    res_df3 = evaluate_all("results/20240201_results/experiment_30_real_beta/onb_function", "30")
    print("B=10")
    print(res_df3.not_started.sum()/(1000000))
    print(res_df3["safety_violations"].max()/10000)
    print(res_df3["safety_violations"].sum()/1000000)

    res_df4 = evaluate_all("results/20240201_results/experiment_32_real_beta/onb_function", "32")
    print("B=20")
    print(res_df4.not_started.sum()/(1000000))
    print(res_df4["safety_violations"].max()/10000)
    print(res_df4["safety_violations"].sum()/1000000)
    

    res_df5 = evaluate_all("results/20240201_results/experiment_33_losbo/onb_function", "33")
    print("LosBO")
    print(res_df5.not_started.sum()/(1000000))
    print(res_df5["safety_violations"].max()/10000)
    print(res_df5["safety_violations"].sum()/1000000)
    #plt.plot(res_df5["safety_violations"]/10000)
    #plt.show()


    function_info = yaml.load(open("results/20240201_results/experiment_31_real_beta/onb_function_46.yaml"), Loader=yaml.FullLoader)
    fig, axs = plt.subplots()
    axs = evaluate_performance("results/20240201_results/experiment_31_real_beta/onb_function_46/experiment_results_31_0_10000.pkl", function_info, axs, label="B=2.5")
    axs = evaluate_performance("results/20240201_results/experiment_30_real_beta/onb_function_46/experiment_results_30_0_10000.pkl", function_info, axs,label="B=10")
    axs = evaluate_performance("results/20240201_results/experiment_32_real_beta/onb_function_46/experiment_results_32_0_10000.pkl", function_info, axs, label="B=20")
    axs = evaluate_performance("results/20240201_results/experiment_33_losbo/onb_function_46/experiment_results_33_0_10000.pkl", function_info, axs, label="LosBO")
    plt.legend(loc='lower right')
    plt.title("Performance of Real Beta SafeOpt with different RKHS Norm bounds")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Predicted Optimum")
    plt.show()

def evaluate_experiment_4():
    """
    function to evaluate experiment 4
    """
    experiments = ["2", "30", "31", "32", "33", "40", "41", "42", "43", "50", "51", "52", "53"]
    algr = ["safeopt", "real_beta", "real_beta", "real_beta", "losbo", "real_beta", "losbo", "real_beta", "losbo", "real_beta", "losbo", "real_beta", "losbo"]
    experiments = ["42", "43"]
    algr = ["real_beta","losbo"]
    legend = ["real_beta","losbo"]
    functions = ["pre_rkhs","pre_rkhs"]
    colors = [c_losbo, c_real_beta, "g", c_ucb_losbo, "c"]
    fig, axs = plt.subplots()
    for i, experiment in enumerate(experiments):
        res_df, axs = evaluate_multiple_performance(f"results/20240201_results/experiment_{experiment}_{algr[i]}/", experiment, functions[i] , axs, legend[i], color=colors[i])
        average_final_performance = res_df["final_performance"].mean()
        print(experiment)
        print(average_final_performance)
    plt.legend(loc='lower right')
    plt.show()
    return res_df

def test_safety_violations():
    '''
    test to evaltuate safety violations
    '''
    experiments = ["40", "41", "42", "43", "50", "51", "52", "53", "60", "61", "70", "71"]
    algorithms = ["real_beta", "losbo", "real_beta", "losbo", "real_beta", "losbo", "real_beta", "losbo","real_beta", "losbo","real_beta","losbo"]
    functions = ["onb_function","onb_function", "pre_rkhs_function", "pre_rkhs_function", "pre_rkhs_function","pre_rkhs_function", "pre_rkhs_function", "pre_rkhs_function", "pre_rkhs_function", "pre_rkhs_function", "pre_rkhs_se_function","pre_rkhs_se_function"]
    for i,experiment in enumerate(experiments):
        res_df1 = evaluate_all(f"results/20240201_results/experiment_{experiment}_{algorithms[i]}/{functions[i]}", experiment)
        print(experiment)
        print(algorithms[i])
        print("not started ", res_df1.not_started.sum()/(1000000))
        print("safety violations wc", res_df1["safety_violations"].max()/10000)
        print("safety violation" , res_df1["safety_violations"].sum()/1000000)

def plot_from_pickle(file, label, color, ax):
    '''
    Code to generate plot from pickle file
    
    Args:
        file: path to pickle file
        label: label for the plot
        color: color for the line in the plot
        ax: axis to plot on
    '''
    # load pickle
    with open(file, 'rb') as f:
        res = pickle.load(f)
    mean = res["mean"]
    q1 = res["q1"]
    q9 = res["q9"]
    mean_funs = res["mean_funs"]

    ax.plot(mean.index, mean, '-o', label=f'{label}', color=color)
    ax.fill_between(mean.index, q1, q9, alpha=0.1, label=f'{label}', color=color)
    ax.plot(mean_funs.index, mean_funs.iloc[:,1:100], alpha=0.2, linestyle="solid", color=color, linewidth=0.1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("norm. pred. optimum")
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 1.1])
    return ax

def generate_plot_experiment_5(): 
    '''
    code to generate plot for experiment 5
    '''
    params = initialize_plot('AAAI')  # specifies font size etc., adjust accordingly
    plt.rcParams.update(params)
    x, y = set_size(470,
                subplots=(3, 2),  # specify subplot layout for nice scaling
                fraction=1.)  # scale width/height
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(x * 2 , y * 3))

    #axs[0,0] = plot_from_pickle("results/20240201_results/experiment_2_safeopt/summary_results_2.pkl", "Safeopt", c_ucb_losbo, axs[0,0])
    axs[0,0] = plot_from_pickle("results/20240201_results/experiment_40_real_beta/summary_results_40.pkl", "RealBetaSafeOpt", c_real_beta, axs[0,0])
    axs[0,0] = plot_from_pickle("results/20240201_results/experiment_41_losbo/summary_results_41.pkl", "LoSBO", c_losbo, axs[0,0])
    axs[0,0].set_title("ONB SE Functions")
    axs[0,0].legend(loc='lower right')
    
    axs[0,1] = plot_from_pickle("results/20240201_results/experiment_42_real_beta/summary_results_42.pkl", "RealBetaSafeOpt", c_real_beta, axs[0,1])
    axs[0,1]= plot_from_pickle("results/20240201_results/experiment_43_losbo/summary_results_43.pkl", "LoSBO", c_losbo, axs[0,1])
    axs[0,1].set_title("Pre-RKHS Matern Functions $l_{GP}=l_c$")
    axs[0,1].legend(loc='lower right')
    
    axs[1,0] = plot_from_pickle("results/20240201_results/experiment_50_real_beta/summary_results_50.pkl", "RealBetaSafeOpt", c_real_beta, axs[1,0])
    axs[1,0]= plot_from_pickle("results/20240201_results/experiment_51_losbo/summary_results_51.pkl", "LoSBO", c_losbo, axs[1,0])
    axs[1,0].set_title("Pre-RKHS Matern Functions $l_{GP}=4 l_c$")
    axs[1,0].legend(loc='lower right')
    
    axs[1,1] = plot_from_pickle("results/20240201_results/experiment_52_real_beta/summary_results_52.pkl", "RealBetaSafeOpt", c_real_beta, axs[1,1])
    axs[1,1]= plot_from_pickle("results/20240201_results/experiment_53_losbo/summary_results_53.pkl", "LoSBO", c_losbo, axs[1,1])
    axs[1,1].set_title("Pre-RKHS Matern Functions $l_{GP}=0.2 l_c$")
    axs[1,1].legend(loc='lower right')
    
    axs[2,0] = plot_from_pickle("results/20240201_results/experiment_60_real_beta/summary_results_60.pkl", "RealBetaSafeOpt", c_real_beta, axs[2,0])
    axs[2,0]= plot_from_pickle("results/20240201_results/experiment_61_losbo/summary_results_61.pkl", "LoSBO", c_losbo, axs[2,0])
    axs[2,0].set_title("Misspecified Kernel $l_{GP}=l_c$")
    axs[2,0].legend(loc='lower right')
    
    axs[2,1] = plot_from_pickle("results/20240201_results/experiment_70_real_beta/summary_results_70.pkl", "RealBetaSafeOpt", c_real_beta, axs[2,1])
    axs[2,1]= plot_from_pickle("results/20240201_results/experiment_71_losbo/summary_results_71.pkl", "LoSBO", c_losbo, axs[2,1])
    axs[2,1].set_title("Pre-RKHS SE functions $l_{GP}=l_c$")
    axs[2,1].legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
def generate_figure_4():
    '''
    Code to generate figure 4
    '''
    params = initialize_plot('AAAI')  # specifies font size etc., adjust accordingly
    plt.rcParams.update(params)
    x, y = set_size(235,
                subplots=(3, 1),  # specify subplot layout for nice scaling
                fraction=1.) # scale width/height
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(x*3 , y*0.4))

    #axs[0,0] = plot_from_pickle("results/20240201_results/experiment_2_safeopt/summary_results_2.pkl", "Safeopt", c_ucb_losbo, axs[0,0])
    axs[0] = plot_from_pickle("results/20240201_results/experiment_40_real_beta/summary_results_40.pkl", "RealBetaSafeOpt", c_real_beta, axs[0])
    axs[0] = plot_from_pickle("results/20240201_results/experiment_41_losbo/summary_results_41.pkl", "LoSBO", c_losbo, axs[0])
    axs[0].set_title("ONB SE")
    axs[0].legend(loc='lower right')
    
    axs[1] = plot_from_pickle("results/20240201_results/experiment_70_real_beta/summary_results_70.pkl", "RealBetaSafeOpt", c_real_beta, axs[1])
    axs[1]= plot_from_pickle("results/20240201_results/experiment_71_losbo/summary_results_71.pkl", "LoSBO", c_losbo, axs[1])
    axs[1].set_title("Pre-RKHS SE")
    axs[1].legend(loc='lower right')
    
    axs[2] = plot_from_pickle("results/20240201_results/experiment_42_real_beta/summary_results_42.pkl", "RealBetaSafeOpt", c_real_beta, axs[2])
    axs[2] = plot_from_pickle("results/20240201_results/experiment_43_losbo/summary_results_43.pkl", "LoSBO", c_losbo, axs[2])
    axs[2].set_title("Pre-RKHS Matern")
    axs[2].legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
def generate_figure_5():
    '''
    code to generate figure 5
    '''
    params = initialize_plot('AAAI')  # specifies font size etc., adjust accordingly
    plt.rcParams.update(params)
    x, y = set_size(235,
                subplots=(3, 1),  # specify subplot layout for nice scaling
                fraction=1.)  # scale width/height
    fig, axs = plt.subplots(1,3, figsize=(x * 3, y * 0.4))
    axs[0] = plot_from_pickle("results/20240201_results/experiment_50_real_beta/summary_results_50.pkl", "RealBetaSafeOpt", c_real_beta, axs[0])
    axs[0]= plot_from_pickle("results/20240201_results/experiment_51_losbo/summary_results_51.pkl", "LoSBO", c_losbo, axs[0])
    axs[0].set_title("Pre-RKHS Matern $l_{GP}=4 l_c$")
    axs[0].legend(loc='lower right')
    
    axs[1] = plot_from_pickle("results/20240201_results/experiment_52_real_beta/summary_results_52.pkl", "RealBetaSafeOpt", c_real_beta, axs[1])
    axs[1] = plot_from_pickle("results/20240201_results/experiment_53_losbo/summary_results_53.pkl", "LoSBO", c_losbo, axs[1])
    axs[1].set_title("Pre-RKHS Matern $l_{GP}=0.2 l_c$")
    axs[1].legend(loc='lower right')
    axs[2] = plot_from_pickle("results/20240201_results/experiment_60_real_beta/summary_results_60.pkl", "RealBetaSafeOpt", c_real_beta, axs[2])
    axs[2] = plot_from_pickle("results/20240201_results/experiment_61_losbo/summary_results_61.pkl", "LoSBO", c_losbo, axs[2])
    axs[2].set_title("Misspecified Kernel $l_{GP}=l_c$")
    axs[2].legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def generate_figure_7():
    '''
    code to generate figure 7
    '''
    params = initialize_plot('AAAI')  # specifies font size etc., adjust accordingly
    plt.rcParams.update(params)
    x, y = set_size(470,
                subplots=(1, 1),  # specify subplot layout for nice scaling
                fraction=1.)  # scale width/height
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(x  , y*0.8))

    axs = plot_from_pickle("results/20240201_results/experiment_42_real_beta/summary_results_42_1000.pkl", "RealBeta", c_real_beta, axs)
    axs = plot_from_pickle("results/20240201_results/experiment_43_losbo/summary_results_43_1000.pkl", "LosBO", c_losbo, axs)
    axs = plot_from_pickle("results/ucblosbo/summary_results_ucblosbo.pkl", "LoS-GP-UCB", c_ucb_losbo, axs)
    axs.set_title("Pre-RKHS Matern $l_{GP}=l_c$")
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.show()

def plot_for_fig_8(ax, dir="results/experiments/results/gaussian_10D", title="", random=False):
    '''
    Code to generate plots in figure 8 from results
    
    Args:
        ax: axis to plot on
        dir: directory where the results are stored
        title: title of the plot
        random: if true plot results from random search
    '''
    list_dir = [x[0] for x in os.walk(dir)]
    
    regret = pd.DataFrame()
    regret_random = pd.DataFrame()
    safety_violation = 0
    for i, dir in enumerate(list_dir):
                if dir[-1]=="D":
                    continue
                seed = dir.split("_")[-1]
                if random:
                    with open(dir + "/config.yaml", 'r') as yaml_file:
                        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
                    results = pd.read_csv(dir + "/results_random.csv")
                    if results["Y"].min() < config["h"]:
                        #print(f"safety constraint violated in {seed}")
                        safety_violation +=1
                else:
                    results = pd.read_csv(dir + f"/seed_{seed}_results.csv")
                    # read config
                    with open(dir + "/config.yaml", 'r') as yaml_file:
                        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
                    # check if safety constraint is violated
                    if results["Y"].min() < config["h"]:
                        #print(f"safety constraint violated in {seed}")
                        safety_violation +=1
                regret[f"regret_{i}"] = (config["optimum"]-results["regret"])/(config["optimum"])

    # calculater average for each row in df
    calculated_regret = regret.mean(axis=1)

    var = regret.var(axis=1)

    if random:
         c = '#0098A1'
         label = "Random"
    else:
         c = '#612158'
         label = "LoS-GP-UCB"
    ax.set_title(title)
    ax.plot(regret.index, regret, alpha=0.05, color=c)
    ax.plot(calculated_regret, color=c,label=label)
    ax.fill_between(calculated_regret.index, calculated_regret-var, calculated_regret+var, color=c, alpha=0.3)
    return ax, safety_violation

def generate_figure_8():
    '''
    Code to generate figure 8
    '''
    params = initialize_plot('AAAI')  # specifies font size etc., adjust accordingly
    plt.rcParams.update(params)
    x, y = set_size(235,
                subplots=(3, 1),  # specify subplot layout for nice scaling
            fraction=1.)  # scale width/height
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(x *3 , y *0.4))
    ax[0], s=plot_for_fig_8(ax[0], "results/results_final/gaussian_10D", "Gaussian10D")
    ax[1], s=plot_for_fig_8(ax[1], "results/results_final/hartmann_6D", "Hartmann6D")
    ax[2], s=plot_for_fig_8(ax[2], "results/results_final/se_camel_2D", "Camelback2D")
    ax[0], s=plot_for_fig_8(ax[0], "results/experiments/results/gaussian_10D", "Gaussian10D", random=True)
    ax[1], s=plot_for_fig_8(ax[1], "results/experiments/results/hartmann_6D", "Hartmann6D", random=True)
    ax[2], s=plot_for_fig_8(ax[2], "results/experiments/results//se_camel_2D", "Camelback2D", random=True)
    ax[2].set_ylim([0, 1])
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')
    ax[0].set_ylabel("Pred Optimum")
    ax[1].set_ylabel("Pred Optimum")
    ax[2].set_ylabel("Pred_Optimum")
    ax[0].set_xlabel("Iterations")
    ax[1].set_xlabel("Iterations")
    ax[2].set_xlabel("Iterations")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    generate_figure_4()
    generate_figure_5()
    generate_figure_7()
    generate_figure_8()