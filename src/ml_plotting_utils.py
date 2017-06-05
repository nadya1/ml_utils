__author__ = 'nadyaK'
__date__ = '05/12/2017'

import matplotlib.pyplot as plt
import numpy as np

def make_coefficient_plot(table,positive_words,negative_words,l2_penalty_list,output_file):
	plt.rcParams['figure.figsize'] = 10,6

	cmap_positive = plt.get_cmap('Reds')
	cmap_negative = plt.get_cmap('Blues')

	xx = l2_penalty_list
	plt.plot(xx,[0.] * len(xx),'--',lw=1,color='k')

	table_positive_words = table.filter_by(column_name='word',values=positive_words)
	table_negative_words = table.filter_by(column_name='word',values=negative_words)
	del table_positive_words['word']
	del table_negative_words['word']

	for i in xrange(len(positive_words)):
		color = cmap_positive(0.8 * ((i + 1) / (len(positive_words) * 1.2) + 0.15))
		plt.plot(xx,table_positive_words[i:i + 1].to_numpy().flatten(),'-',label=positive_words[i],linewidth=4.0,
			color=color)

	for i in xrange(len(negative_words)):
		color = cmap_negative(0.8 * ((i + 1) / (len(negative_words) * 1.2) + 0.15))
		plt.plot(xx,table_negative_words[i:i + 1].to_numpy().flatten(),'-',label=negative_words[i],linewidth=4.0,
			color=color)

	plt.legend(loc='best',ncol=3,prop={'size':16},columnspacing=0.5)
	plt.axis([1,1e5,-1,2])
	plt.title('Coefficient path for 5 (positive & negative) words')
	plt.xlabel('L2 penalty ($\lambda$)')
	plt.ylabel('Coefficient value')
	plt.xscale('log')
	plt.rcParams.update({'font.size':18})
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()

def make_classsification_accuracy_plot(train_accuracy,validation_accuracy,output_file):
	plt.rcParams['figure.figsize'] = 10,6
	sorted_list = sorted(train_accuracy.items(),key=lambda x:x[0])
	plt.plot([p[0] for p in sorted_list],[p[1] for p in sorted_list],'bo-',linewidth=4,label='Training accuracy')
	sorted_list = sorted(validation_accuracy.items(),key=lambda x:x[0])
	plt.plot([p[0] for p in sorted_list],[p[1] for p in sorted_list],'ro-',linewidth=4,label='Validation accuracy')
	plt.xscale('symlog')
	plt.axis([0,1e3,0.78,0.786])
	plt.legend(loc='lower left')
	plt.title('Classification Accuracy vs L2 penalty')
	plt.xlabel('L2 penalty ($\lambda$)')
	plt.ylabel('Classification Accuracy')
	plt.rcParams.update({'font.size':18})
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()

def make_figure(dim,title,xlabel,ylabel,legend):
	plt.rcParams['figure.figsize'] = dim
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if legend is not None:
		plt.legend(loc=legend,prop={'size':15})
	plt.rcParams.update({'font.size':16})
	plt.tight_layout()

def make_train_error_vs_val_error_trees_plot(training_errors, validation_errors, output_file):
	plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')
	plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')

	make_figure(dim=(10,5), title='Error vs number of trees',
				xlabel='Number of trees',
				ylabel='Classification error',
				legend='best')
	plt.savefig(output_file)
	plt.close()

def make_performance_of_adaboost_plot(error_all, output_file):
	plt.rcParams['figure.figsize'] = 7,5
	plt.plot(range(1,31),error_all,'-',linewidth=4.0,label='Training error')
	plt.title('Performance of Adaboost ensemble')
	plt.xlabel('# of iterations')
	plt.ylabel('Classification error')
	plt.legend(loc='best',prop={'size':15})

	plt.rcParams.update({'font.size':16})
	plt.savefig(output_file)
	plt.close()

def make_performance_of_adaboost_plot_error(error_all, test_error_all,output_file):
	plt.rcParams['figure.figsize'] = 7,5
	plt.plot(range(1,31),error_all,'-',linewidth=4.0,label='Training error')
	plt.plot(range(1,31),test_error_all,'-',linewidth=4.0,label='Test error')

	plt.title('Performance of Adaboost ensemble')
	plt.xlabel('# of iterations')
	plt.ylabel('Classification error')
	plt.rcParams.update({'font.size':16})
	plt.legend(loc='best',prop={'size':15})
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()

def plot_pr_curve(precision, recall, title, output_file):
	plt.rcParams['figure.figsize'] = 7, 5
	plt.locator_params(axis = 'x', nbins = 5)
	plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
	plt.title(title)
	plt.xlabel('Precision')
	plt.ylabel('Recall')
	plt.rcParams.update({'font.size': 16})
	plt.savefig(output_file)
	plt.close()

def make_plot_log_likelihood(log_likelihood_all,len_data,batch_size,
							smoothing_window=1,label='',output_file=''):
	plt.rcParams.update({'figure.figsize':(9,5)})
	log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all),np.ones((smoothing_window,) ) /smoothing_window, mode='valid')
	plt.plot(np.array(range( smoothing_window -1, len(log_likelihood_all)) ) *float(batch_size ) /len_data,
		log_likelihood_all_ma, linewidth=4.0, label=label)
	plt.rcParams.update({'font.size': 16})
	plt.tight_layout()
	plt.xlabel('# of passes over data')
	plt.ylabel('Average log likelihood per data point')
	plt.legend(loc='lower right', prop={'size':14})
	#plt.xlim(0,2)
	if output_file != '':
		plt.savefig(output_file)
		plt.close()



