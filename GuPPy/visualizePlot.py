import os
import glob
import param
import re
import math
import numpy as np 
import pandas as pd
from random import randint
import holoviews as hv
from holoviews import opts 
from bokeh.io import export_svgs, export_png
from holoviews.plotting.util import process_cmap
from holoviews.operation.datashader import datashade
import datashader as ds
import matplotlib.pyplot as plt
from preprocess import get_all_stores_for_combining_data
import panel as pn 
pn.extension()


# read h5 file as a dataframe
def read_Df(filepath, event, name):
	if name:
		op = os.path.join(filepath, event+'_{}.h5'.format(name))
	else:
		op = os.path.join(filepath, event+'.h5')
	df = pd.read_hdf(op, key='df', mode='r')

	return df

# make a new directory for saving plots
def make_dir(filepath):
	op = os.path.join(filepath, 'saved_plots')
	if not os.path.exists(op):
		os.mkdir(op)

	return op

# remove unnecessary column names
def remove_cols(cols):
	regex = re.compile('bin_err_*')
	remove_cols = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
	remove_cols = remove_cols + ['err', 'timestamps']
	cols = [i for i in cols if i not in remove_cols]

	return cols

#def look_psth_bins(event, name):

# helper function to create plots
def helper_plots(filepath, event, name, inputParameters):

	basename = os.path.basename(filepath)
	visualize_zscore_or_dff = inputParameters['visualize_zscore_or_dff']

	# note when there are no behavior event TTLs
	if len(event)==0:
		print("\033[1m"+"There are no behavior event TTLs present to visualize.".format(event)+"\033[0m")
		return 0

	
	if os.path.exists(os.path.join(filepath, 'cross_correlation_output')):
		event_corr, frames = [], []
		if visualize_zscore_or_dff=='z_score':
			corr_fp = glob.glob(os.path.join(filepath, 'cross_correlation_output', '*_z_score_*'))
		elif visualize_zscore_or_dff=='dff':
			corr_fp = glob.glob(os.path.join(filepath, 'cross_correlation_output', '*_dff_*'))
		for i in range(len(corr_fp)):
			filename = os.path.basename(corr_fp[i]).split('.')[0]
			event_corr.append(filename)
			df = pd.read_hdf(corr_fp[i], key='df', mode='r')
			frames.append(df)
		if len(frames)>0:
			df_corr = pd.concat(frames, keys=event_corr, axis=1)
		else:
			event_corr = []
			df_corr = []
	else:
		event_corr = []
		df_corr = None


	# combine all the event PSTH so that it can be viewed together
	if name:
		event_name, name = event, name
		new_event, frames, bins = [], [], {}
		for i in range(len(event_name)):
			
			for j in range(len(name)):
				new_event.append(event_name[i]+'_'+name[j].split('_')[-1])
				new_name = name[j]
				temp_df = read_Df(filepath, new_event[-1], new_name)
				cols = list(temp_df.columns)
				regex = re.compile('bin_[(]')
				bins[new_event[-1]] = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
				#bins.append(keep_cols)
				frames.append(temp_df)

		df = pd.concat(frames, keys=new_event, axis=1)
	else:
		new_event = list(np.unique(np.array(event)))
		frames, bins = [], {}
		for i in range(len(new_event)):
			temp_df = read_Df(filepath, new_event[i], '')
			cols = list(temp_df.columns)
			regex = re.compile('bin_[(]')
			bins[new_event[i]] = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
			frames.append(temp_df)

		df = pd.concat(frames, keys=new_event, axis=1)

	if isinstance(df_corr, pd.DataFrame):
		new_event.extend(event_corr)
		df = pd.concat([df,df_corr],axis=1,sort=False).reset_index()
	
	columns_dict = dict()
	for i in range(len(new_event)):
		df_1 = df[new_event[i]]
		columns = list(df_1.columns)
		columns.append('All')
		columns_dict[new_event[i]] = columns

	
	# create a class to make GUI and plot different graphs
	class Viewer(param.Parameterized):

		#class_event = new_event

		# make options array for different selectors
		multiple_plots_options = []
		heatmap_options = new_event
		bins_keys = list(bins.keys())
		if len(bins_keys)>0:
			bins_new = bins
			for i in range(len(bins_keys)):
				arr = bins[bins_keys[i]]
				if len(arr)>0:
					#heatmap_options.append('{}_bin'.format(bins_keys[i]))
					for j in arr:
						multiple_plots_options.append('{}_{}'.format(bins_keys[i], j))

			multiple_plots_options = new_event + multiple_plots_options
		else:
			multiple_plots_options = new_event
		
		# create different options and selectors 
		event_selector = param.ObjectSelector(default=new_event[0], objects=new_event)
		event_selector_heatmap = param.ObjectSelector(default=heatmap_options[0], objects=heatmap_options)
		columns = columns_dict
		df_new = df


		colormaps = plt.colormaps()
		new_colormaps = ['plasma', 'plasma_r', 'magma', 'magma_r', 'inferno', 'inferno_r', 'viridis', 'viridis_r']
		set_a = set(colormaps)
		set_b = set(new_colormaps)
		colormaps = new_colormaps + list(set_a.difference(set_b))

		x_min = float(inputParameters['nSecPrev'])-20
		x_max = float(inputParameters['nSecPost'])+20
		selector_for_multipe_events_plot = param.ListSelector(default=[multiple_plots_options[0]], objects=multiple_plots_options)
		x = param.ObjectSelector(default=columns[new_event[0]][-4], objects=[columns[new_event[0]][-4]])
		y = param.ObjectSelector(default=remove_cols(columns[new_event[0]])[-2], objects=remove_cols(columns[new_event[0]]))

		trial_no = range(1, len(remove_cols(columns[heatmap_options[0]])[:-2])+1)
		trial_ts = ["{} - {}".format(i,j) for i,j in zip(trial_no, remove_cols(columns[heatmap_options[0]])[:-2])] + ['All']
		heatmap_y = param.ListSelector(default=[trial_ts[-1]], objects=trial_ts)
		psth_y = param.ListSelector(objects=trial_ts[:-1])
		select_trials_checkbox = param.ListSelector(default=['just trials'], objects=['mean', 'just trials'])
		Y_Label = param.ObjectSelector(default='y', objects=['y','z-score', '\u0394F/F'])     
		save_options = param.ObjectSelector(default='None' , objects=['None', 'save_png_format', 'save_svg_format', 'save_both_format'])
		save_options_heatmap = param.ObjectSelector(default='None' , objects=['None', 'save_png_format', 'save_svg_format', 'save_both_format'])
		color_map = param.ObjectSelector(default='plasma' , objects=colormaps)
		height_heatmap = param.ObjectSelector(default=600, objects=list(np.arange(0,5100,100))[1:])
		width_heatmap = param.ObjectSelector(default=1000, objects=list(np.arange(0,5100,100))[1:])
		Height_Plot = param.ObjectSelector(default=300, objects=list(np.arange(0,5100,100))[1:])
		Width_Plot = param.ObjectSelector(default=1000, objects=list(np.arange(0,5100,100))[1:])
		save_hm = param.Action(lambda x: x.param.trigger('save_hm'), label='Save')
		save_psth = param.Action(lambda x: x.param.trigger('save_psth'), label='Save')
		X_Limit = param.Range(default=(-5, 10), bounds=(x_min,x_max))
		Y_Limit = param.Range(bounds=(-50, 50.0))

		#C_Limit = param.Range(bounds=(-20,20.0))
		
		results_hm = dict()
		results_psth = dict()


		# function to save heatmaps when save button on heatmap tab is clicked
		@param.depends('save_hm', watch=True)
		def save_hm_plots(self):
			plot = self.results_hm['plot']
			op = self.results_hm['op']
			save_opts = self.save_options_heatmap
			print(save_opts)
			if save_opts=='save_svg_format':
				p = hv.render(plot, backend='bokeh')
				p.output_backend = 'svg'
				export_svgs(p, filename=op+'.svg')
			elif save_opts=='save_png_format':
				p = hv.render(plot, backend='bokeh')
				export_png(p, filename=op+'.png')
			elif save_opts=='save_both_format':
				p = hv.render(plot, backend='bokeh')
				p.output_backend = 'svg'
				export_svgs(p, filename=op+'.svg')
				p_png = hv.render(plot, backend='bokeh')
				export_png(p_png, filename=op+'.png')
			else:
				return 0


		# function to save PSTH plots when save button on PSTH tab is clicked
		@param.depends('save_psth', watch=True)
		def save_psth_plot(self):
			plot, op = [], []
			plot.append(self.results_psth['plot_combine'])
			op.append(self.results_psth['op_combine'])
			plot.append(self.results_psth['plot'])
			op.append(self.results_psth['op'])
			for i in range(len(plot)):
				temp_plot, temp_op = plot[i], op[i]
				save_opts = self.save_options
				if save_opts=='save_svg_format':
					p = hv.render(temp_plot, backend='bokeh')
					p.output_backend = 'svg'
					export_svgs(p, filename=temp_op+'.svg')
				elif save_opts=='save_png_format':
					p = hv.render(temp_plot, backend='bokeh')
					export_png(p, filename=temp_op+'.png')
				elif save_opts=='save_both_format':
					p = hv.render(temp_plot, backend='bokeh')
					p.output_backend = 'svg'
					export_svgs(p, filename=temp_op+'.svg')
					p_png = hv.render(temp_plot, backend='bokeh')
					export_png(p_png, filename=temp_op+'.png')
				else:
					return 0

		# function to change Y values based on event selection
		@param.depends('event_selector', watch=True)
		def _update_x_y(self):
			x_value = self.columns[self.event_selector]
			y_value = self.columns[self.event_selector]
			self.param['x'].objects = [x_value[-4]]
			self.param['y'].objects = remove_cols(y_value)
			self.x = x_value[-4]
			self.y = self.param['y'].objects[-2]

		@param.depends('event_selector_heatmap', watch=True)
		def _update_df(self):
			cols = self.columns[self.event_selector_heatmap]
			trial_no = range(1, len(remove_cols(cols)[:-2])+1)
			trial_ts = ["{} - {}".format(i,j) for i,j in zip(trial_no, remove_cols(cols)[:-2])] + ['All'] 
			self.param['heatmap_y'].objects = trial_ts
			self.heatmap_y = [trial_ts[-1]]

		@param.depends('event_selector', watch=True)
		def _update_psth_y(self):
			cols = self.columns[self.event_selector]
			trial_no = range(1, len(remove_cols(cols)[:-2])+1)
			trial_ts = ["{} - {}".format(i,j) for i,j in zip(trial_no, remove_cols(cols)[:-2])]
			self.param['psth_y'].objects = trial_ts
			self.psth_y = [trial_ts[0]]

	    # function to plot multiple PSTHs into one plot
		@param.depends('selector_for_multipe_events_plot', 'Y_Label', 'save_options', 'X_Limit', 'Y_Limit', 'Height_Plot', 'Width_Plot')
		def update_selector(self):
			data_curve, cols_curve, data_spread, cols_spread = [], [], [], []
			arr = self.selector_for_multipe_events_plot
			df1 = self.df_new
			for i in range(len(arr)):
				if 'bin' in arr[i]:
					split = arr[i].rsplit('_',2)
					df_name = split[0]  #'{}_{}'.format(split[0], split[1])
					col_name_mean = '{}_{}'.format(split[-2], split[-1])
					col_name_err = '{}_err_{}'.format(split[-2], split[-1])
					data_curve.append(df1[df_name][col_name_mean])
					cols_curve.append(arr[i])
					data_spread.append(df1[df_name][col_name_err])
					cols_spread.append(arr[i])
				else:
					data_curve.append(df1[arr[i]]['mean'])
					cols_curve.append(arr[i]+'_'+'mean')
					data_spread.append(df1[arr[i]]['err'])
					cols_spread.append(arr[i]+'_'+'mean')

			

			if len(arr)>0:
				if self.Y_Limit==None:
					self.Y_Limit = (np.nanmin(np.asarray(data_curve))-0.5, np.nanmax(np.asarray(data_curve))+0.5)

				if 'bin' in arr[i]:
					split = arr[i].rsplit('_', 2)
					df_name = split[0]
					data_curve.append(df1[df_name]['timestamps'])
					cols_curve.append('timestamps')
					data_spread.append(df1[df_name]['timestamps'])
					cols_spread.append('timestamps')
				else:
					data_curve.append(df1[arr[i]]['timestamps'])
					cols_curve.append('timestamps')
					data_spread.append(df1[arr[i]]['timestamps'])
					cols_spread.append('timestamps')
				df_curve = pd.concat(data_curve, axis=1)
				df_spread = pd.concat(data_spread, axis=1)
				df_curve.columns = cols_curve
				df_spread.columns = cols_spread

				ts = df_curve['timestamps']
				index = np.arange(0,ts.shape[0], 3)
				df_curve = df_curve.loc[index, :]
				df_spread = df_spread.loc[index, :]
				overlay = hv.NdOverlay({c:hv.Curve((df_curve['timestamps'], df_curve[c]), kdims=['Time (s)']).opts(width=int(self.Width_Plot), height=int(self.Height_Plot), xlim=self.X_Limit, ylim=self.Y_Limit) for c in cols_curve[:-1]})
				spread = hv.NdOverlay({d:hv.Spread((df_spread['timestamps'], df_curve[d], df_spread[d], df_spread[d]), vdims=['y', 'yerrpos', 'yerrneg']).opts(line_width=0, fill_alpha=0.3) for d in cols_spread[:-1]})
				plot_combine = ((overlay * spread).opts(opts.NdOverlay(xlabel='Time (s)', ylabel=self.Y_Label))).opts(shared_axes=False)
				#plot_err = new_df.hvplot.area(x='timestamps', y=[], y2=[])
				save_opts = self.save_options
				op = make_dir(filepath)
				op_filename = os.path.join(op, str(arr)+'_mean')

				self.results_psth['plot_combine'] = plot_combine
				self.results_psth['op_combine'] = op_filename
				#self.save_plots(plot_combine, save_opts, op_filename)
				return plot_combine


		# function to plot mean PSTH, single trial in PSTH and all the trials of PSTH with mean
		@param.depends('event_selector', 'x', 'y', 'Y_Label', 'save_options', 'Y_Limit', 'X_Limit', 'Height_Plot', 'Width_Plot')
		def contPlot(self):
			df1 = self.df_new[self.event_selector]
			#height = self.Heigth_Plot
			#width = self.Width_Plot
			#print(height, width)
			if self.y == 'All':
				if self.Y_Limit==None:
					self.Y_Limit = (np.nanmin(np.asarray(df1))-0.5, np.nanmax(np.asarray(df1))-0.5)


				options = self.param['y'].objects 
				regex = re.compile('bin_[(]')
				remove_bin_trials = [options[i] for i in range(len(options)) if not regex.match(options[i])]

				ndoverlay = hv.NdOverlay({c:hv.Curve((df1[self.x], df1[c])) for c in remove_bin_trials[:-2]})
				img1 = datashade(ndoverlay, normalization='linear', aggregator=ds.count())
				x_points = df1[self.x]
				y_points = df1['mean']
				img2 = hv.Curve((x_points, y_points))
				img = (img1*img2).opts(opts.Curve(width=int(self.Width_Plot), height=int(self.Height_Plot), line_width=4, color='black', xlim=self.X_Limit, ylim=self.Y_Limit, xlabel='Time (s)', ylabel=self.Y_Label))

				save_opts = self.save_options

				op = make_dir(filepath)
				op_filename = os.path.join(op, self.event_selector+'_'+self.y)
				self.results_psth['plot'] = img
				self.results_psth['op'] = op_filename
				#self.save_plots(img, save_opts, op_filename)

				return img

			elif self.y == 'mean' or 'bin' in self.y:

				xpoints = df1[self.x]
				ypoints = df1[self.y]
				if self.y == 'mean':
					err = df1['err']
				else:
					split = self.y.split('_')
					err = df1['{}_err_{}'.format(split[0],split[1])]

				index = np.arange(0, xpoints.shape[0], 3)

				if self.Y_Limit==None:
					self.Y_Limit = (np.nanmin(ypoints)-0.5, np.nanmax(ypoints)+0.5)

				ropts_curve = dict(width=int(self.Width_Plot), height=int(self.Height_Plot), xlim=self.X_Limit, ylim=self.Y_Limit, color='blue', xlabel='Time (s)', ylabel=self.Y_Label)
				ropts_spread = dict(width=int(self.Width_Plot), height=int(self.Height_Plot), fill_alpha=0.3, fill_color='blue', line_width=0)

				plot_curve = hv.Curve((xpoints[index], ypoints[index]))  #.opts(**ropts_curve)
				plot_spread = hv.Spread((xpoints[index], ypoints[index], err[index], err[index]))  #.opts(**ropts_spread) #vdims=['y', 'yerrpos', 'yerrneg']
				plot = (plot_curve * plot_spread).opts({'Curve': ropts_curve, 
										   'Spread': ropts_spread})

				save_opts = self.save_options
				op = make_dir(filepath)
				op_filename = os.path.join(op, self.event_selector+'_'+self.y)
				self.results_psth['plot'] = plot
				self.results_psth['op'] = op_filename
				#self.save_plots(plot, save_opts, op_filename)

				return plot

			else:
				xpoints = df1[self.x]
				ypoints = df1[self.y]
				if self.Y_Limit==None:
					self.Y_Limit = (np.nanmin(ypoints)-0.5, np.nanmax(ypoints)+0.5)

				ropts_curve = dict(width=int(self.Width_Plot), height=int(self.Height_Plot), xlim=self.X_Limit, ylim=self.Y_Limit, color='blue', xlabel='Time (s)', ylabel=self.Y_Label)
				plot = hv.Curve((xpoints, ypoints)).opts({'Curve': ropts_curve})

				save_opts = self.save_options
				op = make_dir(filepath)
				op_filename = os.path.join(op, self.event_selector+'_'+self.y)
				self.results_psth['plot'] = plot
				self.results_psth['op'] = op_filename
				#self.save_plots(plot, save_opts, op_filename)

				return plot

		# function to plot specific PSTH trials
		@param.depends('event_selector', 'x', 'psth_y', 'select_trials_checkbox', 'Y_Label', 'save_options', 'Y_Limit', 'X_Limit', 'Height_Plot', 'Width_Plot')
		def plot_specific_trials(self):
			df_psth = self.df_new[self.event_selector]
			#if self.Y_Limit==None:
			#	self.Y_Limit = (np.nanmin(ypoints)-0.5, np.nanmax(ypoints)+0.5)

			if self.psth_y==None:
				return None
			else:
				selected_trials = [s.split(' - ')[1] for s in list(self.psth_y)]

			index = np.arange(0, df_psth['timestamps'].shape[0], 3)

			if self.select_trials_checkbox==['just trials']:
				overlay = hv.NdOverlay({c:hv.Curve((df_psth['timestamps'][index], df_psth[c][index]), kdims=['Time (s)']) for c in selected_trials})
				ropts = dict(width=int(self.Width_Plot), height=int(self.Height_Plot), xlim=self.X_Limit, ylim=self.Y_Limit, xlabel='Time (s)', ylabel=self.Y_Label)
				return overlay.opts(**ropts)
			elif self.select_trials_checkbox==['mean']:
				arr = np.asarray(df_psth[selected_trials])
				mean = np.nanmean(arr, axis=1)
				err = np.nanstd(arr, axis=1)/math.sqrt(arr.shape[1])
				ropts_curve = dict(width=int(self.Width_Plot), height=int(self.Height_Plot), xlim=self.X_Limit, ylim=self.Y_Limit, color='blue', xlabel='Time (s)', ylabel=self.Y_Label)
				ropts_spread = dict(width=int(self.Width_Plot), height=int(self.Height_Plot), fill_alpha=0.3, fill_color='blue', line_width=0)
				plot_curve = hv.Curve((df_psth['timestamps'][index], mean[index]))  
				plot_spread = hv.Spread((df_psth['timestamps'][index], mean[index], err[index], err[index]))
				plot = (plot_curve * plot_spread).opts({'Curve': ropts_curve, 
										   'Spread': ropts_spread})
				return plot
			elif self.select_trials_checkbox==['mean', 'just trials']:
				overlay = hv.NdOverlay({c:hv.Curve((df_psth['timestamps'][index], df_psth[c][index]), kdims=['Time (s)']) for c in selected_trials})
				ropts_overlay = dict(width=int(self.Width_Plot), height=int(self.Height_Plot), xlim=self.X_Limit, ylim=self.Y_Limit, xlabel='Time (s)', ylabel=self.Y_Label)				

				arr = np.asarray(df_psth[selected_trials])
				mean = np.nanmean(arr, axis=1)
				err = np.nanstd(arr, axis=1)/math.sqrt(arr.shape[1])
				ropts_curve = dict(width=int(self.Width_Plot), height=int(self.Height_Plot), xlim=self.X_Limit, ylim=self.Y_Limit, color='black', xlabel='Time (s)', ylabel=self.Y_Label)
				ropts_spread = dict(width=int(self.Width_Plot), height=int(self.Height_Plot), fill_alpha=0.3, fill_color='black', line_width=0)
				plot_curve = hv.Curve((df_psth['timestamps'][index], mean[index]))  
				plot_spread = hv.Spread((df_psth['timestamps'][index], mean[index], err[index], err[index]))

				plot = (plot_curve*plot_spread).opts({'Curve': ropts_curve, 
										   			  'Spread': ropts_spread})
				return overlay.opts(**ropts_overlay)*plot


		# function to show heatmaps for each event
		@param.depends('event_selector_heatmap', 'color_map', 'height_heatmap', 'width_heatmap', 'heatmap_y')
		def heatmap(self):
			height = self.height_heatmap
			width = self.width_heatmap
			df_hm = self.df_new[self.event_selector_heatmap]
			cols = list(df_hm.columns)
			regex = re.compile('bin_err_*')
			drop_cols = [cols[i] for i in range(len(cols)) if regex.match(cols[i])]
			drop_cols = ['err', 'mean'] + drop_cols
			df_hm = df_hm.drop(drop_cols, axis=1)
			cols = list(df_hm.columns)
			bin_cols = [cols[i] for i in range(len(cols)) if re.compile('bin_*').match(cols[i])]
			time = np.asarray(df_hm['timestamps'])
			event_ts_for_each_event = np.arange(1,len(df_hm.columns[:-1])+1)
			yticks = list(event_ts_for_each_event)
			z_score = np.asarray(df_hm[df_hm.columns[:-1]]).T

			if self.heatmap_y[0]=='All':
				indices = np.arange(z_score.shape[0]-len(bin_cols))
				z_score = z_score[indices,:]
				event_ts_for_each_event = np.arange(1,z_score.shape[0]+1)
				yticks = list(event_ts_for_each_event)
			else:
				remove_all = list(set(self.heatmap_y)-set(['All']))
				indices = sorted([int(s.split('-')[0])-1 for s in remove_all])
				z_score = z_score[indices,:]
				event_ts_for_each_event = np.arange(1,z_score.shape[0]+1)
				yticks = list(event_ts_for_each_event)

			clim = (np.nanmin(z_score), np.nanmax(z_score))
			font_size = {'labels': 16, 'yticks': 6}
			
			if event_ts_for_each_event.shape[0]==1:
				dummy_image = hv.QuadMesh((time, event_ts_for_each_event, z_score)).opts(colorbar=True, clim=clim)
				image = ((dummy_image).opts(opts.QuadMesh(width=int(width), height=int(height), cmap=process_cmap(self.color_map, provider="matplotlib"), colorbar=True, ylabel='Trials', xlabel='Time (s)', fontsize=font_size, yticks=yticks))).opts(shared_axes=False)

				save_opts = self.save_options_heatmap
				op = make_dir(filepath)
				op_filename = os.path.join(op, self.event_selector_heatmap+'_'+'heatmap')
				self.results_hm['plot'] = image
				self.results_hm['op'] = op_filename
				#self.save_plots(image, save_opts, op_filename)
				return image
			else:
				ropts = dict(width=int(width), height=int(height), ylabel='Trials', xlabel='Time (s)', fontsize=font_size, yticks=yticks, invert_yaxis=True)
				dummy_image = hv.QuadMesh((time[0:100], event_ts_for_each_event, z_score[:,0:100])).opts(colorbar=True, cmap=process_cmap(self.color_map, provider="matplotlib"), clim=clim)
				actual_image = hv.QuadMesh((time, event_ts_for_each_event, z_score))
				
				dynspread_img = datashade(actual_image, cmap=process_cmap(self.color_map, provider="matplotlib")).opts(**ropts)   #clims=self.C_Limit, cnorm='log'
				image = ((dummy_image * dynspread_img).opts(opts.QuadMesh(width=int(width), height=int(height)))).opts(shared_axes=False) 
				
				save_opts = self.save_options_heatmap
				op = make_dir(filepath)
				op_filename = os.path.join(op, self.event_selector_heatmap+'_'+'heatmap')
				self.results_hm['plot'] = image
				self.results_hm['op'] = op_filename
				
				return image


	view = Viewer()
	print('view')
	psth_checkbox = pn.Param(view.param.select_trials_checkbox, widgets={
		'select_trials_checkbox': {'type': pn.widgets.CheckBoxGroup, 'inline': True, 
									'name': 'Select mean and/or just trials'}})
	parameters = pn.Param(view.param.selector_for_multipe_events_plot, widgets={
	    'selector_for_multipe_events_plot': {'type': pn.widgets.CrossSelector, 'width':550, 'align':'start'}})
	heatmap_y_parameters = pn.Param(view.param.heatmap_y, widgets={
		'heatmap_y': {'type':pn.widgets.MultiSelect, 'name':'Trial # - Timestamps', 'width':200, 'size':30}})
	psth_y_parameters = pn.Param(view.param.psth_y, widgets={
		'psth_y': {'type':pn.widgets.MultiSelect, 'name':'Trial # - Timestamps', 'width':200, 'size':15, 'align':'start'}})

	options = pn.Column(view.param.event_selector, 
		                pn.Row(view.param.x, view.param.y, width=400), 
						pn.Row(view.param.X_Limit, view.param.Y_Limit, width=400),
						pn.Row(view.param.Width_Plot, view.param.Height_Plot, view.param.Y_Label, view.param.save_options, width=400), 
					    view.param.save_psth)
	
	options_selectors = pn.Row(options, parameters, width=1050)

	line_tab = pn.Column('## '+basename, 
		 				 pn.Row(options_selectors, pn.Column(psth_checkbox, psth_y_parameters), width=1200), 
	                	 view.contPlot, 
	                	 view.update_selector, 
	                	 view.plot_specific_trials)

	hm_tab = pn.Column('## '+basename, pn.Row(view.param.event_selector_heatmap, view.param.color_map, 
											view.param.save_options_heatmap, 
											view.param.width_heatmap, view.param.height_heatmap,
											view.param.save_hm, width=1000), 
											pn.Row(view.heatmap, heatmap_y_parameters)) #
	print('app')

	template = pn.template.MaterialTemplate(title='Visualization GUI')
	
	number = randint(5000,5200)
	
	app = pn.Tabs(('PSTH', line_tab), 
				   ('Heat Map', hm_tab))
	
	template.main.append(app)
	
	template.show(port=number)



# function to combine all the output folders together and preprocess them to use them in helper_plots function
def createPlots(filepath, event, inputParameters):
	average = inputParameters['visualizeAverageResults']
	visualize_zscore_or_dff = inputParameters['visualize_zscore_or_dff']

	if average==True:
		path = []
		for i in range(len(event)):
			if visualize_zscore_or_dff=='z_score':
				path.append(glob.glob(os.path.join(filepath, event[i]+'*_z_score_*')))
			elif visualize_zscore_or_dff=='dff':
				path.append(glob.glob(os.path.join(filepath, event[i]+'*_dff_*')))

		path = np.concatenate(path)
	else:
		if visualize_zscore_or_dff=='z_score':
			path = glob.glob(os.path.join(filepath, 'z_score_*'))
		elif visualize_zscore_or_dff=='dff':
			path = glob.glob(os.path.join(filepath, 'dff_*'))

	name_arr = []
	event_arr = []
	
	indx = []
	for i in range(len(event)):
		if 'control' in event[i].lower() or 'signal' in event[i].lower():
			indx.append(i)

	event = np.delete(event, indx)
	
	for i in range(len(path)):
		name = (os.path.basename(path[i])).split('.')
		name = name[0]
		name_arr.append(name)


	if average==True:
		print('average')
		helper_plots(filepath, name_arr, '', inputParameters)
	else:
		helper_plots(filepath, event, name_arr, inputParameters)



def visualizeResults(inputParameters):

	
	inputParameters = inputParameters


	average = inputParameters['visualizeAverageResults']
	print(average)

	folderNames = inputParameters['folderNames']
	folderNamesForAvg = inputParameters['folderNamesForAvg']
	combine_data = inputParameters['combine_data']

	if average==True and len(folderNamesForAvg)>0:
		#folderNames = folderNamesForAvg
		filepath_avg = os.path.join(inputParameters['abspath'], 'average')
		#filepath = os.path.join(inputParameters['abspath'], folderNames[0])
		storesListPath = []
		for i in range(len(folderNamesForAvg)):
			filepath = folderNamesForAvg[i]
			storesListPath.append(glob.glob(os.path.join(filepath, '*_output_*')))
		storesListPath = np.concatenate(storesListPath)
		storesList = np.asarray([[],[]])
		for i in range(storesListPath.shape[0]):
			storesList = np.concatenate((storesList, np.genfromtxt(os.path.join(storesListPath[i], 'storesList.csv'), dtype='str', delimiter=',')), axis=1)
		storesList = np.unique(storesList, axis=1)
		
		createPlots(filepath_avg, np.unique(storesList[1,:]), inputParameters)

	else:
		if combine_data==True:
			storesListPath = []
			for i in range(len(folderNames)):
				filepath = folderNames[i]
				storesListPath.append(glob.glob(os.path.join(filepath, '*_output_*')))
			storesListPath = list(np.concatenate(storesListPath).flatten())
			op = get_all_stores_for_combining_data(storesListPath)
			for i in range(len(op)):
				storesList = np.asarray([[],[]])
				for j in range(len(op[i])):
					storesList = np.concatenate((storesList, np.genfromtxt(os.path.join(op[i][j], 'storesList.csv'), dtype='str', delimiter=',')), axis=1)
				storesList = np.unique(storesList, axis=1)
				filepath = op[i][0]
				createPlots(filepath, storesList[1,:], inputParameters)
		else:
			for i in range(len(folderNames)):
				
				filepath = folderNames[i]
				storesListPath = glob.glob(os.path.join(filepath, '*_output_*'))
				print(storesListPath)
				for j in range(len(storesListPath)):
					filepath = storesListPath[j]
					storesList = np.genfromtxt(os.path.join(filepath, 'storesList.csv'), dtype='str', delimiter=',')
					
					createPlots(filepath, storesList[1,:], inputParameters)


#print(sys.argv[1:])
#visualizeResults(sys.argv[1:][0])
