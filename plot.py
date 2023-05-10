#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
lfsize = 18
labelsize = 24
labelsize_s,labelsize_b = 24,32
linewidth = 4
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
# colors = ['#D00C0E','#E09C1A','#08A720','#86A8E7','#9D5FFB','#D65780']
labels = ['ELVC','H.264','H.265','DVC','RLVC']
markers = ['o','P','s','D','>','^','<','v','*']
hatches = ['/' ,'\\','--','x', '+', 'O','-','o','.','*']
linestyles = ['solid','dotted','dashed','dashdot', (0, (3, 5, 1, 5, 1, 5))]
from collections import OrderedDict
linestyle_dict = OrderedDict(
    [('solid',               (0, ())),
     ('densely dotted',      (0, (1, 1))),
     ('dotted',              (0, (1, 5))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
linestyles = []
for i, (name, linestyle) in enumerate(linestyle_dict.items()):
    if i >= 9:break
    linestyles += [linestyle]

def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,lfsize=labelsize_b-8,legloc='best',
				xticks=None,yticks=None,xticklabel=None,ncol=None, yerr=None,markers=markers,
				use_arrow=False,arrow_coord=(0.1,43),ratio=None,bbox_to_anchor=(1.1,1.2),use_doublearrow=False,
				linestyles=None,use_text_arrow=False,fps_double_arrow=False,linewidth=None,markersize=None,
				bandlike=False,band_colors=None,annot_bpp_per_video=False,annot_psnr_per_video=False,arrow_rotation=-45,
				annot_loss=False,zoomin=False):
	def plot_axe(axe,xx,yy):
		if yerr is None:
			if linestyles is not None:
				axe.plot(xx, yy, color = color[i], marker = markers[i], 
					linestyle = linestyles[i], 
					label = label[i], 
					linewidth=linewidth, markersize=markersize)
			else:
				axe.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], 
					linewidth=linewidth, markersize=markersize)
		elif bandlike:
			if linestyles is not None:
				axe.plot(xx, yy, color = color[i], marker = markers[i], 
					linestyle = linestyles[i], 
					label = label[i], 
					linewidth=linewidth, markersize=markersize)
			else:
				axe.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], 
					linewidth=linewidth, markersize=markersize)
			axe.fill_between(xx, yy - yerr[i], yy + yerr[i], color=band_colors[i], alpha=0.3)
		else:
			axe.errorbar(xx, yy, yerr=yerr[i], color = color[i], 
				marker = markers[i], label = label[i], 
				linewidth=linewidth, markersize=markersize)
	if linewidth is None:
		linewidth = 2
	if markersize is None:
		markersize = 8
	fig, ax = plt.subplots()
	# ax.grid(zorder=0)
	plt.grid(True, which='both', axis='both', linestyle='--')
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		plot_axe(ax,xx,yy)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	if xticks is not None:
		if xticklabel is None:
			plt.xticks(xticks,fontsize=lfsize)
		else:
			plt.xticks(xticks,xticklabel,fontsize=lfsize)
	ax.tick_params(axis='both', which='major', labelsize=lbsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lbsize)
	if annot_loss:
		xx,yy = XX[0],YY[0]
		reduction = int(np.round((-yy[-1] + yy[0])/yy[0]*100))
		ax.text(xx[-1]-500, yy[-1]+0.2, f"\u2193{reduction}%", ha="center", va="center", size=lbsize, color=color[0])
		ax.text(XX[1][-1]-800, YY[1][-1]-0.4, "Divergent", ha="center", va="center", size=lbsize, color=color[1])
	if annot_bpp_per_video:
		offset = [(4,0),(4,0),(0,0.00125),(-1,-0.00125),(4,0),(-1,0.00125),(4,0)]
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			reduction = int(np.round((-yy[-1] + yy[0])/yy[0]*100))
			ax.text(xx[-1]+offset[i][0], yy[-1]+offset[i][1], f"\u2193{reduction}%", ha="center", va="center", size=lbsize, color=color[i])
	if annot_psnr_per_video:
		offset = [(0,0.5),(3,-0.5),(3,0),(0,-0.5),(3,0),(0,0.5),(0,-0.5)]
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			inc = yy[-1] - yy[0]
			sign = '+' if inc>0 else ''
			ax.text(xx[-1]+offset[i][0], yy[-1]+offset[i][1], f"{sign}{inc:.1f}", ha="center", va="center", size=lbsize, color=color[i])
	if use_arrow:
		ax.text(
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=arrow_rotation if arrow_rotation!=180 else 0, size=lbsize,
		    bbox=dict(boxstyle="larrow,pad=0.3" if arrow_rotation!=180 else "rarrow,pad=0.3", fc="white", ec="black", lw=2))
	if use_doublearrow:
		plt.axhline(y = YY[0,0], color = color[0], linestyle = '--')
		ax.annotate(text='', xy=(2,YY[0,0]), xytext=(2,YY[0,1]), arrowprops=dict(arrowstyle='<->',lw=2, color = color[0]))
		ax.text(
		    2.5, 25, "76% less time", ha="center", va="center", rotation='vertical', size=lfsize, color = color[0])
		plt.axhline(y = YY[2,0], color = color[2], linestyle = '--')
		ax.annotate(text='', xy=(6,YY[2,0]), xytext=(6,YY[2,5]), arrowprops=dict(arrowstyle='<->',lw=2, color = color[2]))
		ax.text(
		    6.5, 23, "87% less time", ha="center", va="center", rotation='vertical', size=lfsize,color = color[2])
	if fps_double_arrow:
		for i in range(3):
			ax.annotate(text='', xy=(31+i*0.5,YY[3*i,0]), xytext=(31+i*0.5,YY[0+3*i,-1]), arrowprops=dict(arrowstyle='<->',lw=2, color = color[i*3]))
			ax.text(
			    32+i*0.5, (YY[3*i,-1]+YY[i*3,0])/2+i*0.5, f"{YY[3*i,-1]/YY[3*i,0]:.1f}X", ha="center", va="center", rotation='vertical', size=lfsize, color = color[i*3])
	if use_text_arrow:
		ax.annotate('Better speed and\ncoding efficiency trade-off', xy=(XX[2][-1]+1, YY[2,-1]+20),  xycoords='data',
            xytext=(0.25, 0.4), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->',lw=2),size=lbsize,
            # horizontalalignment='right', verticalalignment='top'
            )

	if ncol!=0:
		if ncol is None:
			plt.legend(loc=legloc,fontsize = lfsize)
		else:
			plt.legend(loc=legloc,fontsize = lfsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
	
	if ratio is not None:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	# plt.xlim((0.8,3.2))
	# plt.ylim((-40,90))
	if zoomin:
		# create a second axes object for the inset plot
		axins = ax.inset_axes([0, 1, 0.8, 0.4])

		# plot the zoomed-in data on the inset plot
		for i in [0,1,4,5]:
			xx,yy = XX[i],YY[i]
			plot_axe(axins,xx,yy)
		# axins.set_xlim([-2, 2])
		# axins.set_ylim([-0.5, 0.5])

		# xleft, xright = axins.get_xlim()
		# ybottom, ytop = axins.get_ylim()
		# axins.set_aspect(abs((xright-xleft)/(ybottom-ytop))*0.5)

		# set the location of the inset plot
		axins.set_xticklabels('')
		axins.set_yticklabels('')
		ax.indicate_inset_zoom(axins)
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')
	plt.close()


def plot_RD_tradeoff(methods = ['cheng2020-attn','cheng2020-anchor','mbt2018','mbt2018-mean','cheng2020-attn+QR','cheng2020-anchor+QR','mbt2018+QR','mbt2018-mean+QR']):
	bbox_to_anchor = (.53,.59)

	# SPSNRs = [[32.61846537423134, 35.16243499135971, 36.386825913906094, 38.54153810071945, 40.405451371908185, 41.75280112934112, 42.9015796880722, 44.17670942020416], 
	# 		[32.25718354964256, 34.66197091174126, 36.158443320512774, 38.2797465865612, 40.04921350574494, 41.39807619094848, 42.41535145330428, 43.79857496213913],]
	# Sbpps =  [[0.010167921769284998, 0.033496813424336254, 0.049102608978225, 0.11271780705989062, 0.19504989973211062, 0.278838702796725, 0.37827365014607994, 0.537286800275145], 
	# 		[0.012500745584761878, 0.03349855022811375, 0.05198707885325626, 0.11722852922270624, 0.21010299721905001, 0.3225022190327757, 0.420959674954065, 0.593977216174845],]
	SPSNRs = [[28.46,29.79,31.35,33.42],
				[28.60, 29.94, 31.31, 33.44],
				[28.11,29.63,31.38,33.17],
				[27.70, 29.35, 31.12, 32.93],
				[28.61,29.88,31.41,33.53],
				[28.67, 30.05, 31.40, 33.45],
				[28.17,29.74,31.39,33.13],
				[27.73, 29.44, 31.16, 32.96]]
	Sbpps = [[0.112,0.170,0.265,0.422],
	[0.116, 0.180, 0.267, 0.412],
	[0.11,0.195,0.285,0.430],
	[0.124, 0.198, 0.307, 0.460],
	[0.112,0.170,0.265,0.422],
	[0.116, 0.180, 0.267, 0.412],
	[0.11,0.195,0.285,0.430],
	[0.124, 0.198, 0.307, 0.460]
	]
	SPSNRs = np.array(SPSNRs)
	print((SPSNRs[4:,:]-SPSNRs[:4,:]).mean(axis=1))
	print((SPSNRs[4:,:]-SPSNRs[:4,:]).std(axis=1))
	print((SPSNRs[4:,:]-SPSNRs[:4,:]).max(axis=1))
	Sbpps = np.array(Sbpps)
	# selected = [1,5]
	# SPSNRs = SPSNRs[selected]
	# Sbpps = Sbpps[selected]
	# methods = np.array(methods)
	# methods = methods[selected]
	# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
	colors = ["#1f77b4", "#8c564b", "#ff7f0e", "#e377c2", "#2ca02c", "#d62728", "#7f7f7f", "#9467bd", "#bcbd22"]

	line_plot(Sbpps,SPSNRs,methods,colors,
			f'rdtradeoff.eps',
			'BPP','PSNR (dB)',lbsize=24,lfsize=13,linewidth=1,yticks=range(29,34),
			ncol=1,markersize=2,bbox_to_anchor=bbox_to_anchor,)


plot_RD_tradeoff()