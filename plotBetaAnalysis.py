import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, argparse


def plotBetaTradeOff(args, df_beta, df_no_calib, df_ts, threshold, n_branches_edge, overhead, plotPath):


	fig, ax = plt.subplots()

	acc_beta, inf_time_beta = -df_beta.beta_acc.values, df_beta.beta_inf_time.values
	acc_no_calib, inf_time_no_calib = -df_no_calib.beta_acc.values, df_no_calib.beta_inf_time.values
	acc_ts, inf_time_ts = -df_ts.beta_acc.values, df_ts.beta_inf_time.values

	acc_beta_index = np.argsort(acc_beta)
	acc_beta, inf_time_beta	= acc_beta[acc_beta_index], inf_time_beta[acc_beta_index]


	plt.plot(inf_time_beta, acc_beta, color="blue", marker="o", label="Our")
	plt.plot(inf_time_no_calib, acc_no_calib-0.01, color="red", marker="x", label="Conventional")
	plt.plot(inf_time_ts, acc_ts-0.01, color="black", marker="v", label="TS")

	plt.xlabel("Inference Time (ms)", fontsize = args.fontsize)
	plt.ylabel("On-device Accuracy", fontsize = args.fontsize)
	plt.legend(frameon=False, fontsize=args.fontsize)
	plt.xticks(fontsize=args.fontsize)
	plt.yticks(fontsize=args.fontsize)
	plt.tight_layout()
	plt.savefig(plotPath+".pdf")
	plt.title("Number of Branches: %s, Threshold: %s, Overhead: %s"%(n_branches_edge, threshold, overhead), fontsize = args.fontsize-2)
	#plt.tight_layout()
	#plt.savefig(plotPath+".jpg")


def main(args):


	#resultPath = os.path.join(".", "beta_analysis_%s_%s_branches_%s_with_overhead.csv"%(args.model_name, args.n_branches, args.model_id))
	#resultPath = os.path.join(".", "beta_analysis_%s_%s_branches_%s_with_overhead_with_nano.csv"%(args.model_name, args.n_branches, args.model_id))
	resultPath = os.path.join(".", "theoretical_beta_analysis_%s_%s_branches_%s_with_overhead_with_nano_with_test_set_FINAL.csv"%(args.model_name, args.n_branches, args.model_id))

	alternativeResultPath = os.path.join(".", "alternative_method_%s_%s_branches_%s_final_test.csv"%(args.model_name, args.n_branches, args.model_id))

	plotDir = os.path.join(".", "plots")

	if(not os.path.exists(plotDir)):
		os.makedirs(plotDir)

	threshold_list = [0.8]
	overhead_list = [0, 5, 10, 15]

	df = pd.read_csv(resultPath)

	for overhead in overhead_list:

		for n_branches_edge in reversed(range(1, args.n_branches+1)):

			for threshold in threshold_list:
				df_inf_data = df[(df.threshold==threshold) & (df.n_branches_edge==n_branches_edge) & (df.overhead==overhead)]
				#df_inf_data_device = df_device[(df_device.threshold==threshold) & (df_device.n_branches_edge==n_branches_edge) & (df_device.overhead==overhead)]

				#df_alt_inf_data = df_alternative[(df_alternative.threshold==threshold) & (df_alternative.n_branches_edge==n_branches_edge)]

				#df_no_calib, df_ts = df_alt_inf_data[df_alt_inf_data.calib_mode=="no_calib"], df_alt_inf_data[df_alt_inf_data.calib_mode=="global_TS"]

				#df_spsa, df_no_calib, df_ts = df_inf_data[df_inf_data.calib_mode=="beta_calib"], df_inf_data[df_inf_data.calib_mode=="no_calib"], df_inf_data[df_inf_data.calib_mode=="global_TS"]
				df_spsa, df_no_calib, df_ts = df_inf_data[df_inf_data.calib_mode=="beta_calib"], df_inf_data[df_inf_data.calib_mode=="no_calib"], df_inf_data[df_inf_data.calib_mode=="global_TS"]

				plotPath = os.path.join(plotDir, "beta_analysis_%s_branches_threshold_%s_overhead_%s_with_nano"%(n_branches_edge, threshold, overhead) )

				plotBetaTradeOff(args, df_spsa, df_no_calib, df_ts, threshold, n_branches_edge, overhead, plotPath)




if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Plots the Cumulative Regret Versus Time Horizon for several contexts.')
	parser.add_argument('--model_id', type=int, help='Model Id.')
	parser.add_argument('--n_branches', type=int, help='Number of exit exits.')
	parser.add_argument('--model_name', type=str, help='Model name.')
	parser.add_argument('--fontsize', type=int, default=18, help='Font Size.')
	#parser.add_argument('--mode', type=str, help='Theoretical or Experimental Data.')

	args = parser.parse_args()
	main(args)
