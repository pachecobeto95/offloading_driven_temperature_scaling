import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, argparse

def main(args):


	resultPath = os.path.join(".", "theoretical_beta_analysis_%s_%s_branches_%s_with_overhead_with_nano_with_test_set_pos_2_review_theo.csv"%(args.model_name, args.n_branches, args.model_id))
	resultPath1 = os.path.join(".", "theoretical_beta_analysis_%s_%s_branches_%s_with_overhead_with_nano_with_test_set_pos_1_review_theo.csv"%(args.model_name, args.n_branches, args.model_id))

	plotDir = os.path.join(".", "plots_pos_review2")

	if(not os.path.exists(plotDir)):
		os.makedirs(plotDir)

	threshold_list = [0.8]

	df = pd.read_csv(resultPath)
	df1 = pd.read_csv(resultPath1)

	#df_inf_data = df[(df.threshold==threshold) & (df.n_branches_edge==n_branches_edge) & (df.overhead==overhead)]
	print(df.threshold.unique())


if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Plots the Cumulative Regret Versus Time Horizon for several contexts.')
	parser.add_argument('--model_id', type=int, help='Model Id.')
	parser.add_argument('--n_branches', type=int, help='Number of exit exits.')
	parser.add_argument('--model_name', type=str, help='Model name.')
	parser.add_argument('--fontsize', type=int, default=18, help='Font Size.')
	parser.add_argument('--overhead', type=int, help='Overhead')

	#parser.add_argument('--mode', type=str, help='Theoretical or Experimental Data.')

	args = parser.parse_args()
	main(args)
