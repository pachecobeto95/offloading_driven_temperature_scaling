import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, config, argparse


def plotBetaTradeOff(args, df, threshold, n_branches_edge, plotPath):


	acc, inf_time = -df.beta_acc.values, df.beta_inf_time.values
	print(acc.shape)
	plt.scatter(inf_time, acc)
	plt.xlabel("Inference Time (ms)", fontsize = args.fontsize)
	plt.ylabel("Accuracy at the Edge", fontsize = args.fontsize)
	plt.title("Number of Branches: %s, Threshold: %s"%(n_branches_edge, threshold), fontsize = args.fontsize)
	plt.tight_layout()
	plt.savefig(plotPath+".pdf")
	plt.savefig(plotPath+".jpg")


def main(args):

	resultPath = os.path.join(".", "beta_analysis_%s_%s_branches_%s.csv"%(args.model_name, args.n_branches, args.model_id))

	plotDir = os.path.join(".", "plots", "beta_analysis")

	if(not os.path.exists(plotDir)):
		os.makedirs(plotDir)

	threshold_list = [0.7]

	df = pd.read_csv(resultPath)

	for n_branches_edge in reversed(range(1, args.n_branches+1)):

		for threshold in threshold_list:
			df_inf_data = df[(df.threshold==threshold) & (df.n_branches_edge==n_branches_edge)]
			plotPath = os.path.join(plotDir, "beta_analysis_%s_branches_threshold_%s"%(n_branches_edge, threshold) )

			plotBetaTradeOff(args, df_inf_data, threshold, n_branches_edge, plotPath)




if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Plots the Cumulative Regret Versus Time Horizon for several contexts.')
	parser.add_argument('--model_id', type=int, help='Model Id.')
	parser.add_argument('--n_branches', type=int, help='Number of exit exits.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--fontsize', type=int, default=18, help='Font Size.')

	args = parser.parse_args()
	main(args)
