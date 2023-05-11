import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, argparse

def plotBetaTradeOff(args, df_spsa, df_spsa1, df_no_calib, df_ts, threshold, n_branches, overhead, plotPath):
	acc_beta, inf_time_beta = df_spsa.beta_acc.values, df_spsa.beta_inf_time.values

	acc_beta, inf_time_beta = df_spsa.beta_acc.values, df_spsa.beta_inf_time.values
	acc_beta1, inf_time_beta1 = df_spsa1[df_spsa1.beta_acc>0.6].beta_acc.values, df_spsa1[df_spsa1.beta_acc>0.6].beta_inf_time.values

	acc_beta, inf_time_beta = np.concatenate((acc_beta, acc_beta1)), np.concatenate((inf_time_beta, inf_time_beta1))
	
	acc_beta_index = np.argsort(inf_time_beta)
	acc_beta, inf_time_beta	= acc_beta[acc_beta_index], inf_time_beta[acc_beta_index]
	acc_beta, inf_time_beta	= sorted(acc_beta), sorted(inf_time_beta)

	acc_no_calib, inf_time_no_calib = df_no_calib.beta_acc.values, df_no_calib.beta_inf_time.values
	acc_ts, inf_time_ts = df_ts.beta_acc.values, df_ts.beta_inf_time.values

	plt.plot(inf_time_beta, acc_beta, color="blue", marker="o", linestyle="solid", label=r"AdaTS($\beta$), $\gamma$=%s"%(threshold))
	plt.plot(inf_time_no_calib, acc_no_calib, marker="x", markersize=8, color= "red", label=r"No Calib, $\gamma$=%s"%(threshold))
	plt.plot(inf_time_ts, acc_ts , marker="*", markersize=8, color="black", label=r"TS, $\gamma$=%s"%(threshold))


	#plt.ylim(0.5, 1.01)
	plt.xlabel("Inference Time (ms)", fontsize=args.fontsize)
	plt.ylabel("On-device Accuracy", fontsize=args.fontsize)
	#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	plt.xticks(fontsize=args.fontsize)
	plt.yticks(fontsize=args.fontsize)
	plt.tight_layout()

	plt.savefig(plotPath+".pdf")


def main(args):


	resultPath = os.path.join(".", "theoretical_beta_analysis_%s_%s_branches_%s_with_overhead_with_nano_with_test_set_pos_2_review_theo.csv"%(args.model_name, args.n_branches, args.model_id))
	resultPath1 = os.path.join(".", "theoretical_beta_analysis_%s_%s_branches_%s_with_overhead_with_nano_with_test_set_pos_1_review_theo.csv"%(args.model_name, args.n_branches, args.model_id))

	plotDir = os.path.join(".", "plots_pos_review2")

	if(not os.path.exists(plotDir)):
		os.makedirs(plotDir)

	threshold = 0.8

	df = pd.read_csv(resultPath)
	df1 = pd.read_csv(resultPath1)

	df_inf_data = df[df.overhead==args.overhead]
	df_inf_data1 = df1[df1.overhead==args.overhead]

	df_spsa, df_no_calib, df_ts = df_inf_data[df_inf_data.calib_mode=="beta_calib"], df_inf_data[df_inf_data.calib_mode=="no_calib"], df_inf_data[df_inf_data.calib_mode=="global_TS"]
	#df_spsa1 = df_inf_data1[df_inf_data1.calib_mode=="beta_calib"]
	df_spsa1, df_no_calib1, df_ts1 = df_inf_data1[df_inf_data1.calib_mode=="beta_calib"], df_inf_data1[df_inf_data1.calib_mode=="no_calib"], df_inf_data1[df_inf_data1.calib_mode=="global_TS"]

	plotPath = os.path.join(plotDir, "beta_analysis_%s_branches_threshold_%s_overhead_%s_with_nano"%(args.n_branches, threshold, args.overhead) )

	plotBetaTradeOff(args, df_spsa, df_spsa1, df_no_calib, df_ts, threshold, args.n_branches, args.overhead, plotPath)


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
