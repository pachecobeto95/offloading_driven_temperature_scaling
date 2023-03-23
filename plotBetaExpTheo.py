import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, argparse


def plotBetaTradeOff(args, df_beta_exp, df_beta_theo, df_no_calib, df_ts, threshold, n_branches_edge, overhead, plotPath):


	fig, ax = plt.subplots()

	acc_beta_exp, inf_time_beta_exp = -df_beta_exp.beta_acc.values, df_beta_exp.beta_inf_time.values
	acc_beta_theo, inf_time_beta_theo = -df_beta_theo.beta_acc.values, df_beta_theo.beta_inf_time.values
	acc_no_calib, inf_time_no_calib = -df_no_calib.beta_acc.values, df_no_calib.beta_inf_time.values
	acc_ts, inf_time_ts = -df_ts.beta_acc.values, df_ts.beta_inf_time.values

	acc_beta_index_theo, acc_beta_index_exp = np.argsort(acc_beta_theo), np.argsort(acc_beta_exp)
	acc_beta_theo, inf_time_beta_theo = acc_beta_theo[acc_beta_index_theo], inf_time_beta_theo[acc_beta_index_theo]
	acc_beta_exp, inf_time_beta_exp = acc_beta_exp[acc_beta_index_exp], inf_time_beta_exp[acc_beta_index_exp]


	plt.plot(inf_time_beta_exp, acc_beta_exp, color="green", marker="^", label="Real EdOff-TS")
	plt.plot(inf_time_beta_theo, acc_beta_theo, color="blue", marker=".", label="EdOff-TS")
	plt.plot(inf_time_no_calib, acc_no_calib-0.01, color="red", marker="x", label="No-calibration")
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
	resultExpPath = os.path.join(".", "beta_analysis_%s_%s_branches_%s_with_overhead_with_nano_final.csv"%(args.model_name, args.n_branches, args.model_id))
	resultTheoPath = os.path.join(".", "theoretical_beta_analysis_%s_%s_branches_%s_with_overhead_with_nano_with_test_set_FINAL_FINAL.csv"%(args.model_name, args.n_branches, args.model_id))

	alternativeResultPath = os.path.join(".", "theoretical_beta_analysis_%s_%s_branches_%s_with_overhead_with_nano_with_test_set_FINAL.csv"%(args.model_name, args.n_branches, args.model_id))

	plotDir = os.path.join(".", "new_plots")

	if(not os.path.exists(plotDir)):
		os.makedirs(plotDir)

	threshold_list = [0.8]
	overhead_list = [0, 5, 10, 15]

	df_exp = pd.read_csv(resultExpPath)
	df_theo = pd.read_csv(resultTheoPath)
	df_alt = pd.read_csv(alternativeResultPath)

	for overhead in overhead_list:

		for n_branches_edge in reversed(range(1, args.n_branches+1)):

			for threshold in threshold_list:
				df_inf_data_exp = df_exp[(df_exp.threshold==threshold) & (df_exp.n_branches_edge==n_branches_edge) & (df_exp.overhead==overhead)]
				df_inf_data_theo = df_theo[(df_theo.threshold==threshold) & (df_theo.n_branches_edge==n_branches_edge) & (df_theo.overhead==overhead)]

				df_inf_data_alt = df_alt[(df_alt.threshold==threshold) & (df_alt.n_branches_edge==n_branches_edge) & (df_alt.overhead==overhead)]

				#df_no_calib, df_ts = df_alt_inf_data[df_alt_inf_data.calib_mode=="no_calib"], df_alt_inf_data[df_alt_inf_data.calib_mode=="global_TS"]

				df_no_calib, df_ts = df_inf_data_alt[df_inf_data_alt.calib_mode=="no_calib"], df_inf_data_alt[df_inf_data_alt.calib_mode=="global_TS"]
				df_spsa_theo = df_inf_data_theo[df_inf_data_theo.calib_mode=="beta_calib"]
				df_spsa_exp = df_inf_data_exp[df_inf_data_exp.calib_mode=="beta_calib"]

				plotPath = os.path.join(plotDir, "theo_exp_beta_analysis_%s_branches_threshold_%s_overhead_%s_with_nano"%(n_branches_edge, threshold, overhead) )

				plotBetaTradeOff(args, df_spsa_exp, df_spsa_theo, df_no_calib, df_ts, threshold, n_branches_edge, overhead, plotPath)




if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Plots the Cumulative Regret Versus Time Horizon for several contexts.')
	parser.add_argument('--model_id', type=int, help='Model Id.')
	parser.add_argument('--n_branches', type=int, help='Number of exit exits.')
	parser.add_argument('--model_name', type=str, help='Model name.')
	parser.add_argument('--fontsize', type=int, default=18, help='Font Size.')
	#parser.add_argument('--mode', type=str, help='Theoretical or Experimental Data.')

	args = parser.parse_args()
	main(args)
