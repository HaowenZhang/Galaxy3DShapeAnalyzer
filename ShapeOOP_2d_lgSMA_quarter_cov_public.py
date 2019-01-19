# This is a object-oriented (poorly-written, though) code that decomposes the ***projected*** b/a axis ratio-lgSMA distribution
# to figure out what fractions the prolate galaxies populated in the early universe observed by CANDELS.
# Author: Haowen Zhang
# Date of last modification: Jan. 18, 2019

import mpi4py.MPI as MPI
from schwimmbad import MPIPool
import numpy as np
from numpy import sin, cos, arccos
import scipy
import scipy.io as sio
from scipy.special import iv, gammaln
from scipy.stats import rice, norm, lognorm, truncnorm, multivariate_normal, binned_statistic_2d
from itertools import product
import time
import os
import sys
import pandas as pd
import h5py
# import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
import corner
from emcee import EnsembleSampler
# from emcee.utils import MPIPool



def axis_ratio_2d(semi_axes, theta, phi):
	# the function to calculate the projected axis ratio
	# given the intrinsic 3D semi axes of an ellipsoid and
	# the viewing directions.
	# parameters:
	# 	semi_axes: array_like of shape (3,)
	# 		the three semi axes of the ellipsoid, in the
	# 		order of [a, b, c], satisfying that a >= b >= c.
	# 		in this program we can simply assume the eq. of 
	# 		the ellipsoid is x**2/a**2 + y**2/b**2 + z**2/c**2 = 1.
	# 	theta: float
	# 		the polar angle of the viewing direction vector in
	# 		the aforementioned coordinate frame
	# 	phi: float
	# 		the azimuthal angle of the viewing direction vector in
	# 		the same frame.
	# returns:
	# 	a: float
	# 		projected semi-major axis. 
	# 	b/a: float
	# 		project b/a axis ratio
	a, b, c = semi_axes
	n = [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
	A = sin(phi)**2 / a**2 + cos(phi)**2 / b**2
	B = (cos(phi)**2 / a**2 + sin(phi)**2 / b**2) * cos(theta)**2 + sin(theta)**2 / c**2
	C = (cos(phi)**2 / a**2 + sin(phi)**2 / b**2) * sin(theta)**2 + cos(theta)**2 / c**2
	D = 2 * sin(phi) * cos(phi) * cos(theta) * (1 / a**2 - 1 / b**2)
	E = -2 * sin(phi) * cos(phi) * sin(theta) * (1 / a**2 - 1 / b**2)
	F = 2 * sin(theta) * cos(theta) * (1 / c**2 - cos(phi)**2 / a**2 - sin(phi)**2 / b**2)
	k1 = n[1] * cos(phi) / b**2 - n[0] * sin(phi) / a**2
	k2 = n[2] * sin(theta) / c**2 - cos(theta) * (n[0] * cos(phi) / a**2 + n[1] * sin(phi) / b**2)
	k3 = n[2] * cos(theta) / c**2 + sin(theta) * (n[0] * cos(phi) / a**2 + n[1] * sin(phi) / b**2)
	p = - k1 / k3
	q = - k2 / k3
	AA = - (A + C * p**2 + E * p)
	BB = - (2 * C * p * q + D + E * q + F * p)
	CC = - (B + C * q**2 + F * q)
	aa = a
	bb = b
	a = (-2 / (AA + CC + ((AA - CC)**2 + BB**2)**0.5))**0.5
	b = (-2 / (AA + CC - ((AA - CC)**2 + BB**2)**0.5))**0.5
	return a, b / a

# This declaration assumes the variables: ETa_grid, ba_lgSMA_bins, bin_obs, num_obs
# are declared and initialized elsewhere in the code and are visible to this function, which is done 
# in the main function as an example.
def lnprob(p):
# def lnprob(p, ETa_grid, ba_lgSMA_bins, bin_obs, num_obs):
	# The log-probability function used in ShapeAnalysis.MCMC() function.
	# parameters:
	# 	p: array_like
	# 		the parameter set on which the probability is to be evaluated. 
	# 	ETa_grid: array_like
	# 		grid of (E, T, a) values, with the ba-lgSMA histogram calculated for each grid value.
	# 	ba_lgSMA_bins: array_like, 3-dimensional
	# 		list of the ba-lgSMA histograms for each single (E, T, a) parameter set. 
	# 	bin_obs: array_like
	# 		*linearized/serialized* observed ba-lgSMA bins.
	# 	num_obs: int
	# 		the total number of observed galaxies in the sample.
	# returns:
	# 	lnprob: float
	# 		log-probability value for the input parameter set p.
	E, T, a, covEE, covTT, covaa, covEa, = p 
	# print p
	# some hard limits on the parameters
	if not (0 <= E <= 1 and 0 <= T <= 1 and a < 1.5 and 0 < covEE < 1 and 0 < covTT < 1 and 0 < covaa < 1):
		return -np.inf
	# The covariance matrix for (E, T, (log10)a)
	cov = np.array([[covEE, 0, covEa],\
			 [0, covTT, 0],\
			 [covEa, 0, covaa]])
	try:
		# evaluate the probability distribution over the ETa_grid
		prob_grid = multivariate_normal.pdf(ETa_grid, mean=[E,T,a], cov=cov)
		prob_grid = prob_grid / np.sum(prob_grid)
	except ValueError:
		# This basically happens when the covariance matrix isn't positive semi-definite.
		return -np.inf
	
	if np.isnan(prob_grid).any():
		# When giving us NaN probability values, return a ln(prob) of -inf.
		print 'E,T,a,eE,eT,ea: ', p
		return -np.inf

	# The model distribution is simply the weighted sum of the observed b/a-lgSMA distributions 
	# generated from different intrinsic shape parameter sets (E, T, a), normalized by the number
	# of observed galaxies
	bin_model = (np.tensordot(prob_grid, ba_lgSMA_bins, axes=1) * num_obs)
	# Calculate the likelihood based on Poisson statistics
	likelihood = np.multiply(bin_obs, np.log(bin_model)) - bin_model - gammaln(bin_obs + 1)
	# if in a certain b/a-lgSMA bin, there are no galaxies observed, then the likelihood should be 
	# -bin_model, i.e. the negative of the number of model galaxies in this bin.
	likelihood[np.where(bin_obs == 0)] = -bin_model[np.where(bin_obs == 0)]
	# For bins with infinite likelihood, we assign a floor of -100 as a penalty.
	likelihood[np.where(np.isinf(likelihood))] = -100
	# Assuming independence between different bins, the total ln(likelihood) is simply the sum 
	# over all the b/a-lgSMA bins.
	lnlike = np.sum(likelihood)
	if np.isnan(likelihood).any():
		return -np.inf
	return lnlike

# Note: The meaning of symbol "a" in this code is sometimes confusing. So to clarify:
# All the "a" in the occurence of "(E, T, a)" refers to ***the log10 of*** the longest main
# axis of the galaxy; while when it comes to ***projected*** b/a, "a" refers to the 
# half-light semi-major axis measured by GALFIT (for real galaxies) or calculated from
# the ellipsoid projection algorithm (i.e. axis_ratio_2d()) (for model galaxies); "a" in the
# occurences of "intrinsic c/a" or "intrinsic b/a" refer to the intrinsic longest main axis,
# not its log10.
class ShapeAnalysis(object):
	def __init__(self, E_range=[0,1], T_range=[0,1], a_range=[0,10],\
				       E_step=0.01, T_step=0.01, lgSMA_step=0.1, ba_step=0.05, a_step=0.5):
		# initialization function. 
		# parameters:
		# 	E_range: array_like, optional
		# 		lower and upper limit of the ellipticity (E = 1 - c/a)
		# 		to be considered *both* when calculating the model 
		# 		histogram and grid searching. 
		# 		default: [0,1]
		# 	T_range: array_like, optional
		# 		lower and upper limit of the triaxiality (T = (a**2 - b**2) / (a**2 - c**2))
		# 		to be considered *both* when calculating the model 
		# 		histogram and grid searching. 
		# 		default: [0,1]
		# 	a_range: array_like, optional
		# 		lower and upper limit of the intrinsic semi-major axis in ***kpc*** (itself, ***not log10***)
		# 		of the ellipsoid to be considered *both* when calculating the model 
		# 		histogram and grid searching. 
		# 		default: [0,15]
		# 	E_step: float, optional
		# 		the step of E used *only* when calculating the model
		# 		histogram.
		# 		default: 0.01
		# 	T_step: float, optional
		# 		the step of T used *only* when calculating the model
		# 		histogram.
		# 		default: 0.01
		# 	a_step: float, optional
		# 		the step of *log10(a)* used *only* when calculating the model
		# 		histogram.
		# 		default: 0.1
		# 	lgSMA_step: float, optional
		# 		the step of lgSMA used *only* when calculating the model
		# 		histogram.
		# 		default: 0.1
		# 	ba_step: float, optional
		# 		the bin width of the b/a value in *all* the histograms
		# 		of b/a
		# 		default: 0.01
		self.E_range = E_range
		self.T_range = T_range
		self.a_range = a_range
		self.E_step = E_step
		self.T_step = T_step
		self.a_step = a_step
		self.lgSMA_step = lgSMA_step
		self.ba_step = ba_step
		self.ndim = 7

		# The grid 
		# self.ba_obs_grid = np.linspace(ba_step / 2.0, 1 - ba_step / 2.0, int(1 / ba_step))
		# self.ba_exp_grid = self.ba_obs_grid[:]
		# self.lgSMA_obs_grid = np.linspace(-2 + lgSMA_step / 2.0, 2 - lgSMA_step / 2, round(4 / lgSMA_step))
		# self.lgSMA_exp_grid = self.lgSMA_obs_grid[:]



	def GenerateModelHist(self, view_num=100000, save_name='./ETa_lgSMA.mat', oblate_only=False):
		# the function to generate the model ***projected*** b/a-lgSMA histograms
		# on a series of (E, T, a) grid points. The method is straightforward,
		# i.e. view the ellipsoid with (E, T, a) in a bunch of random directions
		# and bin the yielded b/a, and correct the intrinsic distribution to
		# the observed one, taking the asymmetric error into account. For 
		# detailed discussions on this asymmetric error, see Chang et al. (2013),
		# doi:10.1088/0004-637X/773/2/149.
		# 
		# parameters:
		# 	view_num: int, optional
		# 		the number of random directions used for each combination of (E, T, a). 
		# 		default: 100000
		# 	save_name: str, optional
		# 		the directory to save the generated histograms.
		# 		default: './ETa_lgSMA.mat'
		# oblate_only: Boolean, optional
		# 		Whether we only use the oblate (i.e. disky) galaxies in the modeling of
		# 		the observed ***projected*** b/a-lgSMA distributions.
		E_grid = np.linspace(self.E_range[0], self.E_range[-1] - self.E_step, \
					int((self.E_range[1] - self.E_range[0]) / self.E_step))
		# print E_grid.shape
		T_grid = np.linspace(self.T_range[0], self.T_range[-1] - self.T_step, 
					int((self.T_range[1] - self.T_range[0]) / self.T_step))
		# print T_grid.shape
		a_grid = np.linspace(np.log10(self.a_step), np.log10(self.a_range[-1]), 
					int((self.a_range[1] - self.a_range[0]) / self.a_step))

		self.E_grid = E_grid
		self.T_grid = T_grid
		self.a_grid = a_grid

		# Serialize the 3D grid in (E, T, a) param space.
		self.ETa_grid = np.array([cc for cc in product(E_grid, T_grid, a_grid)])

		# If the model ***projected*** b/a-lgSMA distributions are already calculated and
		# saved, just read it.
		if os.path.exists(save_name):
			try:
				hist = sio.loadmat(save_name)
				self.ba_lgSMA_bins = hist['ba_lgSMA_bins']
				self.ETa_grid = hist['grid_pts']
				self.grid_pts = hist['grid_pts']
			# if the file is too large for sio to read, use h5py instead.
			except NotImplementedError:
				hist = h5py.File(save_name, 'r')
				self.ba_lgSMA_bins = hist['ba_lgSMA_bins'].value.T
				# renormalize the model ***projected*** b/a-lgSMA distributions again to ensure correct
				# answers. The index 12345 is arbitrarily picked, since all b/a-lgSMA distributions should
				# have the same normalization (no matter what that value is).
				self.ba_lgSMA_bins /= np.sum(self.ba_lgSMA_bins[12345])
				# Note the needed transposition.
				self.ETa_grid = hist['grid_pts'].value.T
			
			# Calculate the ***intrinsic*** c/a and b/a of galaxy shape with parameters (E, T, a),
			# according to their definitions.
			self.ca_set = 1 - self.ETa_grid[:,0]
			self.ba_set = ((1 - self.ETa_grid[:,1]) * 1 + self.ETa_grid[:,1] * self.ca_set**2)**0.5
			self.ind_type = {}
			# Pick out different shapes according to the definition used by van der Wel et al. (2014) and
			# Zhang et al. (2019).
			self.ind_type['prolate'] = np.where(((1 - self.ba_set)**2 + self.ca_set**2 > 0.16) & (self.ba_set  < 1 - self.ca_set))
			self.ind_type['oblate'] = np.where(((1 - self.ba_set)**2 + self.ca_set**2 <= 0.16))
			self.ind_type['spheroidal'] = np.where(((1 - self.ba_set)**2 + self.ca_set**2 > 0.16) & (self.ba_set  >= 1 - self.ca_set))
			if oblate_only:
				# If we only use the oblate galaxies, just discard ptolate and oblate ones.
				oblate_inds = np.where((1 - self.ba_set)**2 + self.ca_set**2 <= 0.16)
				self.ba_lgSMA_bins = self.ba_lgSMA_bins[oblate_inds]
				self.ETa_grid = self.ETa_grid[oblate_inds]
				self.grid_pts = self.grid_pts[oblate_inds]
				# print 'the number of oblate elements: ', len(oblate_inds[0])
			# The calculation of dust extinction for galaxies is deprecated in the current version,
			# so please just ignore this.
			try:
				self.Av_bins = hist['AV_bins']
				if oblate_only:
					self.Av_bins = self.Av_bins[oblate_inds]
			except KeyError:
				print 'no dust maps in precalculated data.'
				pass
			return
		
		# If, instead, there's no pre-calculated ***projected*** b/a-lgSMA distributions for different
		# intrinsic shapes of galaxies, we have to calculate this using the following codes.
		grid_pts = [] # The list that is to contain all the (E, T, a) grid points whose corresponding ***projected*** b/a-lgSMA distributions are calculated.
		ba_lgSMA_bins = [] # The list that is to contain all the ***projected*** b/a-lgSMA distributions.
		
		# initialize the MPI communicator, in order to scatter the calculation to different sub-processes,
		# which enhances the speed.
		comm = MPI.COMM_WORLD
		comm_rank = comm.Get_rank()
		comm_size = comm.Get_size()

		# naive application of MPI: only scatter different E values to sub-processes.
		sendbuf = None
		if not comm_rank:
			sendbuf = E_grid
		local_E_grid = np.empty(len(E_grid) / comm_size, dtype=np.float64)
		comm.Scatter(sendbuf, local_E_grid, root=0)
		# print 'local_E_grid: ', local_E_grid
		

		local_ba_lgSMA_bins = [] # The list that is to contain all the ***projected*** b/a-lgSMA distributions calculated in the sub-process.
		local_grid_pts = [] # The list that is to contain all the (E, T, a) calculated in the sub-process.
		for i in range(len(local_E_grid)):
			# if not comm_rank:
			# print 'start the %d-th outer loop.' %i
			E = local_E_grid[i]
			for j in range(len(T_grid)):
				if not comm_rank:
					print j
				T = T_grid[j]
				for k in range(len(a_grid)):
					a = 10**a_grid[k]
					c = a * (1 - E)
					b = ((1 - T) * a**2 + T * c**2)**0.5

					# Note that to generate random viewing angles uniformly in 4pi solid space, 
					# we need to generate cos(theta) that is uniformly distributed in [-1, 1].
					coss = np.random.uniform(-1, 1, size=view_num)
					theta = np.arccos(coss)
					phi = np.random.uniform(0, 2 * np.pi, size=view_num)

					# Calculate the ***projected*** semi-major axis and b/a axis ratio based on the
					# intrinsic main axis and viewing angles.
					SMA, ba = axis_ratio_2d([a, b, c], theta, phi)


					if b/a >= c/b:
						sma_face = a
					else:
						sma_face = b
					ba_face = np.max([b/a, c/b])

					lgSMA_obs = []
					ba_obs = []
					# randomly sampling, taking observation errors into account
					for l in range(view_num):
						# The distribution used here is called Rice distribution, which is implemented in SciPy.
						# For more discussions on this issue, see Chang et al. (2013), doi:10.1088/0004-637X/773/2/149.
						# Here we choose 0.04 as a typical uncertainty for projected b/a.
						ba_tmp = 1 - rice.rvs((1 - ba[l]) / (0.04 * ba[l])) * 0.04 * ba[l]
						# We assume that the ***projected*** semi-minor axis of the image will not be 
						# affected by the error in b/a, which can be wrong. Based on this assumption, 
						# the error in b/a would affect the value of semi-major axis in the following way:
						lgsma_tmp = np.log10(SMA[l] * ba[l] / ba_tmp)

						# # The correction of systematic trend, deprecated
						# lgsma_tmp = lgsma_tmp - 0.1237 * (ba_tmp - ba_face) + 0.008408
						# sigma_sma = 0.07 / np.log(10)
						# lgsma_tmp = np.random.normal(lgsma_tmp, sigma_sma)
						# lgsma_tmp = np.random.normal(np.log10(SMA[l]), sigma_sma)
						
						# Append the randomized b/a and lgSMA to the list for binning.
						ba_obs.append(ba_tmp)
						lgSMA_obs.append(lgsma_tmp)

					# Bin the data.
					ba_lgSMA_bin = np.histogram2d(ba_obs, lgSMA_obs, range=[[0,1],[-1, 1]], bins=[int(1 / self.ba_step), round(2/ self.lgSMA_step)], normed=True)[0]
					# ba_lgSMA_bin = np.histogram2d(ba, np.log10(SMA), range=[[0,1],[-1, 1]], bins=[int(1 / self.ba_step), round(2 / self.lgSMA_step)], normed=True)[0]
					
					# Normalization. A special case is that the galaxy is observed in none of the bins.
					if np.sum(ba_lgSMA_bin) == 0:
						# ba_lgSMA_bin[:,:] = 0
						pass
					else:
						ba_lgSMA_bin = ba_lgSMA_bin / np.sum(ba_lgSMA_bin)

					# We further linearize the ***projected*** b/a-lgSMA 2D histograms, 
					# for the convenience of manipulations with tensor products.
					local_ba_lgSMA_bins.append(ba_lgSMA_bin.flatten())
					local_grid_pts.append([E, T, np.log10(a)])

					# Bug checking.
					# print ba_lgSMA_bin[0, 0], np.isnan(ba_lgSMA_bin).any()
					if (np.isnan(ba_lgSMA_bin).any() and (not np.isnan(ba_lgSMA_bin).all())):
						print 'E, T, a: ', E, T, a
						print 'SMA: ', SMA
						print 'ba: ', ba
						# plt.imshow(ba_lgSMA_bin)
						# plt.scatter(np.arange(len(ba_lgSMA_bin.flatten())), ba_lgSMA_bin)
						# plt.show()
						# plt.close()

		# Gather the calculated data back to the root process and write to the file.
		local_ba_lgSMA_bins = np.array(local_ba_lgSMA_bins)
		local_grid_pts = np.array(local_grid_pts)
		ba_lgSMA_bins = None
		grid_pts = None			
		if not comm_rank:
			grid_pts = np.empty([len(E_grid) * len(T_grid) * len(a_grid), 3], dtype=np.float64)
			ba_lgSMA_bins = np.empty((len(E_grid) * len(T_grid) * len(a_grid), int(round(1 / self.ba_step * 2 / self.lgSMA_step))), dtype=np.float64)
		comm.Gather(local_ba_lgSMA_bins, ba_lgSMA_bins, root=0)
		comm.Gather(local_grid_pts, grid_pts, root=0)
		print 'finished gathering.'
		if not comm_rank:
			save_dict = {'grid_pts': grid_pts, 'ba_lgSMA_bins': ba_lgSMA_bins}
			sio.savemat(save_name, save_dict)
		

	def ReadData(self, ba_lgSMA_bins=None, cat_path='./data_5fields.mat', mass_range=[9,9.5], z_range=[1.0,1.5]):
		# the function to read the CANDELS catalog data and bin the b/a values of 
		# the galaxies that lie within the z and mass bin needed. 
		# parameters:
		# 	ba_lgSMA_bins: array_like, optional
		# 		precalculated ba_lgSMA bins. If provided, the program doesn't read the data in .mat file
		# 		but instead use this as observed data. Note this binned data is *linearized/serialized*, i.e. 
		# 		one-dimensional.
		# 	cat_path: str, optional
		# 		the path of the catalog data. 
		# 		default: './data_5fields.mat'
		# 	mass_range: array_like, optional
		# 		specific lgM* range. 
		# 		default: [9.0, 9.5]
		# 	z_range: array_like, optional
		# 		specific z range. 
		# 		default: [1, 1.5]

		if ba_lgSMA_bins is not None:
			# self.bin_obs is the observed ***projected*** b/a-lgSMA histograms, serialized into one-
			# dimensional.
			self.bin_obs = ba_lgSMA_bins
			# self.num_obs is the total number of galaxies in the specified mass and redshift range.
			self.num_obs = np.sum(ba_lgSMA_bins)
			return

		data = sio.loadmat(cat_path)
		mass = data['mass'][0]
		z = data['redshift'][0]
		ba = data['ba'][0]
		# delsma = data['delsma'][0]
		sma = data['sma'][0]
		# delssfr = data['delssfr'][0]
		
		# dba and dsma are the uncertainties in ***projected*** b/a and lgSMA provided by the
		# input catalog. They will be used to do bootstrap analysis to determine the uncertainties
		# of best fitting parameters.
		try:
			dba = data['dba'][0]
			dsma = data['dsma'][0]
		except KeyError:
			print 'The input data don\'t have uncertainties, unable to resample!'
			pass

		# Again, any calculation of dust extinction in galaxies is deprecated.
		# try:
		# 	Av = data['av'][0]
		# except KeyError:
		# 	print 'The input data have no Av values.'
		# 	pass
		
		# pick out the galaxies in the specified mass and redshift range
		ind = np.where((mass > mass_range[0]) & (mass < mass_range[-1]) & \
			 		   (z > z_range[0]) & (z < z_range[1]))
		# print ind
		mass = mass[ind]
		z = z[ind]
		ba = ba[ind]
		# delsma = delsma[ind]
		sma = sma[ind]

		# # visualize the ***projected*** b/a-lgSMA distribution, if needed.
		# plt.scatter(sma, ba, s=3)
		# plt.xlim((-0.25,1))
		# plt.ylim((0,1))
		# plt.title('corrected', fontsize=15)
		# plt.xlabel('log a [kpc]', fontsize=15)
		# plt.ylabel('b/a', fontsize=15)
		# plt.savefig('ba_sma_scatter.eps')
		# plt.close()

		# fig = plt.figure(figsize=(8, 6))
		# ax = fig.add_subplot(111)
		# ind = np.where((sma > 0.1) & (sma < 0.3))
		# ax.hist(ba[ind], range=(0, 1), bins=20, histtype='step', color='blue', normed=True, label='0.1 < logSMA [kpc] < 0.3')
		# ind = np.where((sma > 0.5) & (sma < 0.7))
		# ax.hist(ba[ind], range=(0, 1), bins=20, histtype='step', color='orange', normed=True, label='0.5 < logSMA [kpc] < 0.7')
		# ind = np.where((sma > 0.6) & (sma < 0.8))
		# ax.hist(ba, range=(0, 1), bins=20, histtype='step', color='red', normed=True, label='all SMA')
		# plt.legend()
		# plt.ylabel('normalized frequency')
		# plt.xlabel(r'$b/a$')
		# plt.savefig('0.5z1.0_9.5m10.0_ba_hist.eps')
		# plt.close()

		# self.bin_obs = np.histogram2d(ba, sma, bins=[int(1 / self.ba_step), round(2 / self.lgSMA_step)], range=[[0,1],[-1, 1]])[0]
		# plt.imshow(np.flip(self.bin_obs, 0), extent=[-1, 1, 0, 1], aspect='auto')
		# plt.show()
		# plt.close()

		# store the information of picked galaxies in the class object itself
		self.mass = mass
		self.z = z
		self.sma = sma
		self.ba = ba
		
		
		self.num_obs = len(ba)
		print 'number of observed pts: ', self.num_obs
		try:
			
			# self.Av = Av[ind]
			if 'dba' in data.keys():
				self.dba = dba[ind]
				self.dsma = dsma[ind]
				ind = np.where((self.dba < 1) & (self.dsma < 1))
				self.mass = self.mass[ind]
				self.sma = self.sma[ind]
				self.z = self.z[ind]
				self.ba = self.ba[ind]
				# self.Av = self.Av[ind]
				self.dsma = self.dsma[ind]
				self.dba = self.dba[ind]
		
		except NameError:
			pass
		

		# Calculate the observed ***projected*** b/a-lgSMA distributions based on
		# the observed b/a and lgSMA values.
		self.bin_obs = np.histogram2d(ba, sma, bins=[int(1 / self.ba_step), round(2 / self.lgSMA_step)], range=[[0,1],[-1, 1]])[0]
		# print np.max(self.bin_obs)
		# plt.imshow(np.flip(self.bin_obs, 0), extent=[-1, 1, 0, 1], aspect='auto')
		# plt.show()
		# plt.close()
		# Linearize the observed bin to facilitate the lnprob caculation.
		self.bin_obs = self.bin_obs.flatten()

		# sma/ba_real are the values that actually come from the input catalog, 
		# we save them in these two arrays because self.sma/ba will be changed
		# by the process of resampling, which is designed to facilitate the determination
		# of fitting parameters' uncertainties via bootstrap.
		self.sma_real = sma[:]
		self.ba_real = ba[:]
		self.bin_obs_real = self.bin_obs[:]
		# print self.bin_obs



	def resample(self):
		# The function that resamples the projected b/a and lgSMA values of the observed galaxies.
		# The probability distribution assumed for lgSMA and b/a are Gaussian and Rice, respectively,
		# with the central values being the ones in the catalog, and the dispersion being the 1-sigma
		# error given by the catalog.
		# ***Note*** that self.sma/ba will get overwritten by resampled values. But the real values are
		# stored in self.sma/ba_real in ReadData().
		try:
			sma_new = self.sma_real[:]
			ba_new = self.ba_real[:]
			for i in range(len(sma_new)):
				sma_new[i] = norm.rvs(loc=sma_new[i], scale=self.dsma[i])
				ba_new[i] = 1 - rice.rvs((1 - ba_new[i]) / self.dba[i]) * self.dba[i]
			self.sma = sma_new
			self.ba = ba_new
			self.bin_obs = np.histogram2d(self.ba, self.sma, bins=[int(1 / self.ba_step), round(2 / self.lgSMA_step)], range=[[0,1],[-1, 1]])[0].flatten()
		except AttributeError:
			print 'no uncertainty data, unable to resample!'

	def BootStrap(self, nBS=100, log='./BootStrap_bfp_log'):
		# The function to carry out bootstrap, and write the ensemble of best fitting parameters
		# to the log file.
		# Parameters:
		# 		nBS: int, optional
		# 			The number of bootstrap, default: 100
		# 		log: str, optional
		# 			The file name of the log file that will contain the output best fitting parameters.
		# 			Default: './BootStrap_bfp_log'
		BS_log = open(log, 'w')
		BS_log.close()
		for i in range(nBS):
			# Everytime we should resample again.
			self.resample()
			global bin_obs
			bin_obs = self.bin_obs
			global num_obs
			num_obs = self.num_obs

			# And then carry out MCMC.
			bfp = self.MCMC(nburn=5, nMCMC=5, threads=8)

			# write best fitting parameters.
			BS_log = open(log, 'a+')
			BS_log.write(' '.join(map(str, bfp)) + '\n')
			BS_log.close()

	def MCMC(self, nwalkers=50, nburn=200, nMCMC=1000, use_MPI=False, chain_file='chain.dat', fig_name='./MCMC_corner.png', plot_corner=False, **kwargs):
		# The function to carry out MCMC. 
		# parameters:
		# 	nwalkers: int, optional
		# 		the number of walkers in MCMC, which must be even. 
		# 		default: 50
		# 	nburn: int, optional
		# 		the number of burn-in steps in MCMC.
		# 		default: 200
		# 	nMCMC: int, optional
		# 		the number of final MCMC steps in MCMC.
		# 		default: 1000
		# 	use_MPI: Boolean, optional
		# 		whether to use MPI. 
		# 		default: False
		# returns:
		# 	p_best: array_like
		# 		best fitting parameter set.
		
		# Initialize the walkers with a set of initial points, p0.
		E0 = np.random.normal(0.5, 0.3, size=nwalkers)
		T0 = np.random.normal(0.5, 0.3, size=nwalkers)
		a0 = np.random.normal(0, 0.7, size=nwalkers)
		covEE = truncnorm.rvs(0, 1, loc=0.3, scale=0.1, size=nwalkers)
		covTT = truncnorm.rvs(0 ,1, loc=0.3, scale=0.1, size=nwalkers)
		covaa = truncnorm.rvs(0, 1, loc=0.1, scale=0.1, size=nwalkers)
		covEa = truncnorm.rvs(0, 1, loc=0.1, scale=0.1, size=nwalkers)
		p0 = [[E0[i], T0[i], a0[i], covEE[i], covTT[i], covaa[i], covEa[i]] for i in range(nwalkers)]
		print 'start MCMC.'


		if not use_MPI:
			sampler = EnsembleSampler(nwalkers, self.ndim, lnprob, **kwargs)	
			# sampler = EnsembleSampler(nwalkers, self.ndim, lnprob, \
			# 		  args=(self.E_grid, self.T_grid, self.a_grid, self.ba_lgSMA_bins, self.bin_obs, self.num_obs), **kwargs)	

		# When using MPI, we differentiate between different processes.
		else:
			pool = MPIPool()
			if not pool.is_master():
				pool.wait()
				sys.exit(0)
			# sampler = EnsembleSampler(nwalkers, self.ndim, lnprob, \
			# 		  args=(self.E_grid, self.T_grid, self.a_grid, self.ba_lgSMA_bins, self.bin_obs, self.num_obs), pool=pool, **kwargs)
			sampler = EnsembleSampler(nwalkers, self.ndim, lnprob,  pool=pool, **kwargs)

		# burn-in phase
		pos, prob, state = sampler.run_mcmc(p0, nburn, chain_file=chain_file)
		sampler.reset()

		# MCMC phase
		sampler.run_mcmc(pos, nMCMC, chain_file=chain_file)

		if use_MPI:
			pool.close()
		
		# If we want to make classic corner plots...
		if plot_corner:
			samples = sampler.chain[:, nMCMC / 2:, :].reshape((-1, self.ndim))
			fig = corner.corner(samples, labels=['E', 'T', 'a', 'covEE', 'covTT', 'covaa', 'covEa'])
			fig.savefig(fig_name)

		# Get the best fitting parameters. We take the median parameter value for the ensemble
		# of steps with log-probabilities within the largest 30% among the whole ensemble as the
		# best parameters.
		samples = sampler.flatchain
		lnp = sampler.flatlnprobability
		crit_lnp = np.percentile(lnp, 70)
		good = np.where(lnp > crit_lnp)
		p_best = [np.median(samples[good, i]) for i in range(self.ndim)]

		return np.array(p_best)

	def prob_map(self, p, mass_range, z_range):
		# The function to make the map showing the probability of a galaxy's 
		# being prolate, oblate, or spheroidal, given its mass, redshift, projected
		# b/a and lgSMA. This is the function that generates the Figs. 14 and 15 of
		# Zhang et al. (2019) (arXiv: 1805.12331)
		# Parameters:
		# 		p: array_like
		# 			The parameter based on which the probability distributions are to
		# 			be generated.
		# 		mass_range: list of float, shape: (2,)
		# 			The range of the stellar masses whose probability map is to be gene-
		# 			rated. ***Note*** that the content of the map is determined by the
		# 			input model parameters, p,  ***only***. I make mass_range and z_range
		# 			as arguments only to facilitate creating file names.
		# 		z_range: list of float, shape: (2,)
		# 			The range of the redshifts whose probability map is to be gene-
		# 			rated. ***Note*** that the content of the map is determined by the
		# 			input model parameters, p,  ***only***. I make mass_range and z_range
		# 			as arguments only to facilitate creating file names.
		
		E, T, a, covEE, covTT, covaa, covEa = p 

		# normalize the observed ***projected*** b/a-lgSMA distribution to ensure correctness.
		self.ba_lgSMA_bins = self.ba_lgSMA_bins / np.sum(ba_lgSMA_bins[0])
		
		# parameters that are not physical
		if not (0 <= E <= 1 and 0 <= T <= 1 and a < 1.5):
			return -np.inf
		
		# The way to generate the probability map is to simply make a random realization
		# of the modeled distribution with the same number of galaxies as real observations, 
		# and use the modeled frequency maps of different shapes to approximate their probability
		# maps.
		
		# Calculate the probability for each set of (E, T, a) 
		cov = np.array([[covEE, 0, covEa],\
				 [0, covTT, 0],\
				 [covEa, 0, covaa]])
		prob_grid = multivariate_normal.pdf(self.ETa_grid, mean=[E,T,a], cov=cov)
		prob_grid = prob_grid / np.sum(prob_grid)
		if np.isnan(prob_grid).any():
			print 'E,T,a,eE,eT,ea: ', p
			return -np.inf

		# And make a weighted sum of the ***projected*** b/a-lgSMA distributions generated
		# from each set of (E, T, a) with their probability values being weights.
		bin_model = (np.tensordot(prob_grid, self.ba_lgSMA_bins, axes=1) * self.num_obs)
		# bin_model[np.where(bin_model < 1e-4)] = 0
		
		# The sum of b/a-lgSMA distributions generated by different intrinsic galaxy shapes
		bin_model_prolate = (np.tensordot(prob_grid[self.ind_type['prolate']], self.ba_lgSMA_bins[self.ind_type['prolate']], axes=1) * self.num_obs)
		bin_model_oblate = (np.tensordot(prob_grid[self.ind_type['oblate']], self.ba_lgSMA_bins[self.ind_type['oblate']], axes=1) * self.num_obs)
		bin_model_spheroidal = (np.tensordot(prob_grid[self.ind_type['spheroidal']], self.ba_lgSMA_bins[self.ind_type['spheroidal']], axes=1) * self.num_obs)
		
		# convert the linearized b/a-lgSMA distributions back to two-dimensional arrays to
		# facilitate plot making.
		model = bin_model.reshape(20, 20)
		model_prolate = bin_model_prolate.reshape(20,20)
		model_oblate = bin_model_oblate.reshape(20,20)
		model_spheroidal = bin_model_spheroidal.reshape(20,20)

		# We truncate the range of b/a-lgSMA distributions since there are barely anything
		# in the truncated part.
		model, model_prolate, model_oblate, model_spheroidal =\
		model[:,5:], model_prolate[:,5:], model_oblate[:,5:], model_spheroidal[:,5:]
		obs = self.bin_obs.reshape(20, 20)
		obs = obs[:,5:]

		# the fractions of different shapes in each b/a-lgSMA bin are simply the number
		# of galaxies of those shapes divided by the total galaxy number in that bin.
		frac_prolate = model_prolate / model
		frac_oblate = model_oblate / model
		frac_spheroidal = model_spheroidal / model

		# Start making plots
		fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
		
		# We only show the modeled fractions where there is at least one observed galaxy.
		frac_prolate[np.where(obs == 0)] = np.NaN
		frac_oblate[np.where(obs == 0)] = np.NaN
		frac_spheroidal[np.where(obs == 0)] = np.NaN
		cmap = plt.get_cmap('coolwarm')
		cmap.set_bad(color='white')

		fsize=7
		titlesize = 8
		cbticksize = 5

		im = ax[0].imshow(np.flip(frac_prolate, 0), aspect=2.0, extent=( -0.5, 1, 0.05, 1.0), cmap=cmap, vmin=0, vmax=1)
		ax[0].set_title('(a) elongated probability')
		cbar = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
		cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=cbticksize)
		cbar.set_label(r'probability', fontsize=fsize)
		ax[0].set_xlabel('log a [kpc]', fontsize=fsize)
		ax[0].set_ylabel('b/a', fontsize=fsize)

		im = ax[1].imshow(np.flip(frac_oblate, 0), aspect=2.0, extent=( -0.5, 1, 0.05, 1.0), cmap=cmap, vmin=0, vmax=1)
		ax[1].set_title('(b) disky probability')
		cbar = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
		cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=cbticksize)
		cbar.set_label(r'probability', fontsize=fsize)
		ax[1].set_xlabel('log a [kpc]', fontsize=fsize)
		ax[1].set_ylabel('b/a', fontsize=fsize)

		im = ax[2].imshow(np.flip(frac_spheroidal, 0), aspect=2.0, extent=( -0.5, 1, 0.05, 1.0), cmap=cmap, vmin=0, vmax=1)
		ax[2].set_title('(c) spheroidal probability')
		cbar = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
		cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=cbticksize)
		cbar.set_label(r'probability', fontsize=fsize)
		ax[2].set_xlabel('log a [kpc]', fontsize=fsize)
		ax[2].set_ylabel('b/a', fontsize=fsize)
		plt.subplots_adjust(left=0.1, wspace=0.45)
		plt.savefig('%.1fz%.1f_%.1fm%.1f_prob_ETa.eps' % (z_range[0], z_range[-1], mass_range[0], mass_range[-1]))
		plt.close()

	def compare_plot(self, p, mass_range=None, z_range=None):
		# The function to draw the distribution generated by a certain parameter set
		# along with the observed distribution, as a comparison. 
		# parameters:
		# 		p: array_like
		# 			The parameter based on which the model distributions are to
		# 			be generated.
		# 		mass_range: list of float, shape: (2,)
		# 			The range of the stellar masses whose probability map is to be gene-
		# 			rated. ***Note*** that the content of the map is determined by the
		# 			input model parameters, p,  ***only***. I make mass_range and z_range
		# 			as arguments only to facilitate creating file names.
		# 		z_range: list of float, shape: (2,)
		# 			The range of the redshifts whose probability map is to be gene-
		# 			rated. ***Note*** that the content of the map is determined by the
		# 			input model parameters, p,  ***only***. I make mass_range and z_range
		# 			as arguments only to facilitate creating file names.

		# The codes here are actually very similar to the ones in prob_map(), so please
		# refer to the comments therein in case of any questions.
		E, T, a, covEE, covTT, covaa, covEa = p 
		self.ba_lgSMA_bins = self.ba_lgSMA_bins / np.sum(ba_lgSMA_bins[0])
		if not (0 <= E <= 1 and 0 <= T <= 1 and a < 1.5):
			return -np.inf
		cov = np.array([[covEE, 0, covEa],\
				 [0, covTT, 0],\
				 [covEa, 0, covaa]])
		prob_grid = multivariate_normal.pdf(self.ETa_grid, mean=[E,T,a], cov=cov)
		prob_grid = prob_grid / np.sum(prob_grid)
		if np.isnan(prob_grid).any():
			print 'E,T,a,eE,eT,ea: ', p
			return -np.inf
		
		# The only difference with prob_map() is that we are getting the distributions of
		# ***numbers*** of galaxies with different shapes in each b/a-lgSMA bin, so we don't
		# have to divide the numbers by the total galaxy number in that bin.
		bin_model = (np.tensordot(prob_grid, self.ba_lgSMA_bins, axes=1) * self.num_obs)
		# bin_model[np.where(bin_model < 1e-4)] = 0
		bin_model_prolate = (np.tensordot(prob_grid[self.ind_type['prolate']], self.ba_lgSMA_bins[self.ind_type['prolate']], axes=1) * self.num_obs)
		bin_model_oblate = (np.tensordot(prob_grid[self.ind_type['oblate']], self.ba_lgSMA_bins[self.ind_type['oblate']], axes=1) * self.num_obs)
		bin_model_spheroidal = (np.tensordot(prob_grid[self.ind_type['spheroidal']], self.ba_lgSMA_bins[self.ind_type['spheroidal']], axes=1) * self.num_obs)
		model = bin_model.reshape(20, 20)
		model_prolate = bin_model_prolate.reshape(20,20)
		model_oblate = bin_model_oblate.reshape(20,20)
		model_spheroidal = bin_model_spheroidal.reshape(20,20)
		model, model_prolate, model_oblate, model_spheroidal =\
		model[:,5:], model_prolate[:,5:], model_oblate[:,5:], model_spheroidal[:,5:]
		obs = self.bin_obs.reshape(20, 20)
		obs = obs[:,5:]

		# sio.savemat('./%.1fz%.1f_%.1fm%.1f_modeling_results.mat'  % (z_range[0], z_range[1], mass_range[0], mass_range[1]),\
		# 		{'obs': obs, 'model': model, 'model_prolate': model_prolate,\
		# 		'model_spheroidal': model_spheroidal, 'model_oblate': model_oblate})

		# Write the model results (total and those with different intrinsic shapes) into .dat files
		file_name = './%.1fz%.1f_%.1fm%.1f_model.dat' % (z_range[0], z_range[1], mass_range[0], mass_range[1])
		f = open(file_name, 'w')
		for i in range(20):
			for j in range(15):
				f.write(str(model[i, j]) + ' ')
			f.write('\n')

		file_name = './%.1fz%.1f_%.1fm%.1f_model_prolate.dat' % (z_range[0], z_range[1], mass_range[0], mass_range[1])
		f = open(file_name, 'w')
		for i in range(20):
			for j in range(15):
				f.write(str(model_prolate[i, j]) + ' ')
			f.write('\n')

		file_name = './%.1fz%.1f_%.1fm%.1f_model_oblate.dat' % (z_range[0], z_range[1], mass_range[0], mass_range[1])
		f = open(file_name, 'w')
		for i in range(20):
			for j in range(15):
				f.write(str(model_oblate[i, j]) + ' ')
			f.write('\n')

		file_name = './%.1fz%.1f_%.1fm%.1f_model_spheroidal.dat' % (z_range[0], z_range[1], mass_range[0], mass_range[1])
		f = open(file_name, 'w')
		for i in range(20):
			for j in range(15):
				f.write(str(model_spheroidal[i, j]) + ' ')
			f.write('\n')

		file_name = './%.1fz%.1f_%.1fm%.1f_obs.dat' % (z_range[0], z_range[1], mass_range[0], mass_range[1])
		f = open(file_name, 'w')
		for i in range(20):
			for j in range(15):
				f.write(str(obs[i, j]) + ' ')
			f.write('\n')
		
		# Start making plots.
		fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
		
		fsize= 10
		titlesize = 10
		cbticksize = 7
		vmin = np.min([obs, model])
		vmax = np.max([obs, model])
		print np.max(obs)
		im = ax[0,0].imshow(np.flip(obs, 0), aspect='auto', extent=( -0.5, 1, 0.05, 1.0), cmap=plt.get_cmap('coolwarm'), vmin=vmin, vmax=vmax)
		ax[0,0].set_title('(a) observed distribution', fontsize=titlesize)
		cbar = fig.colorbar(im, ax=ax[0,0])
		cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=cbticksize)
		cbar.set_label('number', fontsize=fsize)
		ax[0,0].set_xlabel('log a [kpc]', fontsize=fsize)
		ax[0,0].set_ylabel('b/a', fontsize=fsize)

		im = ax[0,1].imshow(np.flip(model, 0), aspect='auto', extent=( -0.5, 1, 0.05, 1.0), cmap=plt.get_cmap('coolwarm'), vmin=vmin, vmax=vmax)
		ax[0,1].set_title('(b) model distribution', fontsize=titlesize)
		cbar = fig.colorbar(im, ax=ax[0,1])
		cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=cbticksize)
		cbar.set_label('number', fontsize=fsize)
		ax[0,1].set_xlabel('log a [kpc]', fontsize=fsize)
		ax[0,1].set_ylabel('b/a', fontsize=fsize)

		im = ax[0,2].imshow(np.flip(obs, 0) - np.flip(model, 0), aspect='auto', extent=( -0.5, 1, 0.05, 1.0), cmap=plt.get_cmap('coolwarm'))
		ax[0,2].set_title('(c) residual', fontsize=titlesize)
		cbar = fig.colorbar(im, ax=ax[0,2])
		cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=cbticksize)
		cbar.set_label(r'$\Delta$ number', fontsize=fsize)
		ax[0,2].set_xlabel('log a [kpc]', fontsize=fsize)
		ax[0,2].set_ylabel('b/a', fontsize=fsize)

		# vmin, vmax = np.min([model_prolate, model_oblate, model_spheroidal]), np.max([model_prolate, model_oblate, model_spheroidal])

		im = ax[1,0].imshow(np.flip(model_prolate, 0), aspect='auto', extent=( -0.5, 1, 0.05, 1.0), cmap=plt.get_cmap('coolwarm'))
		ax[1,0].set_title('(d) elongated galaxies', fontsize=titlesize)
		cbar = fig.colorbar(im, ax=ax[1,0])
		cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=cbticksize)
		cbar.set_label(r'number', fontsize=fsize)
		ax[1,0].set_xlabel('log a [kpc]', fontsize=fsize)
		ax[1,0].set_ylabel('b/a', fontsize=fsize)

		im = ax[1,1].imshow(np.flip(model_oblate, 0), aspect='auto', extent=( -0.5, 1, 0.05, 1.0), cmap=plt.get_cmap('coolwarm'))
		ax[1,1].set_title('(e) disky galaxies', fontsize=titlesize)
		cbar = fig.colorbar(im, ax=ax[1,1])
		cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=cbticksize)
		cbar.set_label(r'number', fontsize=fsize)
		ax[1,1].set_xlabel('log a [kpc]', fontsize=fsize)
		ax[1,1].set_ylabel('b/a', fontsize=fsize)

		im = ax[1,2].imshow(np.flip(model_spheroidal, 0), aspect='auto', extent=( -0.5, 1, 0.05, 1.0), cmap=plt.get_cmap('coolwarm'))
		ax[1,2].set_title('(f) spheroidal galaxies', fontsize=titlesize)
		cbar = fig.colorbar(im, ax=ax[1,2])
		cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=cbticksize)
		cbar.set_label(r'number', fontsize=fsize)
		ax[1,2].set_xlabel('log a [kpc]', fontsize=fsize)
		ax[1,2].set_ylabel('b/a', fontsize=fsize)
		plt.subplots_adjust(left=0.05, hspace=0.35, right=0.95)

		plt.savefig('%.1fz%.1f_%.1fm%.1f_ETa.eps' % (z_range[0], z_range[-1], mass_range[0], mass_range[-1]))
		plt.close()

	def shape_frac(self, p_input, z_range=None, mass_range=None):
		# The function to calculate the fractions of galaxies with different shapes
		# in the whole galaxy population within certain redshift and mass range, 
		# and to make the plot showing the ***intrinsic*** c/a-b/a distribution.
		# parameters:
		# 		p_input: array_like
		# 			The parameter based on which the model distributions are to
		# 			be generated.
		# 		mass_range: list of float, shape: (2,)
		# 			The range of the stellar masses whose probability map is to be gene-
		# 			rated. ***Note*** that the content of the map is determined by the
		# 			input model parameters, p,  ***only***. I make mass_range and z_range
		# 			as arguments only to facilitate creating file names.
		# 		z_range: list of float, shape: (2,)
		# 			The range of the redshifts whose probability map is to be gene-
		# 			rated. ***Note*** that the content of the map is determined by the
		# 			input model parameters, p,  ***only***. I make mass_range and z_range
		# 			as arguments only to facilitate creating file names.
		
		# We generate a large ensemble of (E, T, a) parameter sets according to
		# the probability distribution described by the input parameters, and 
		# simply count how many of them fit in different shape definitions used
		# by van der Wel et al. (2014) and Zhang et al. (2019).
		E, T, a, covEE, covTT, covaa, covEa = p_input
		cov = np.array([[covEE, 0, covEa],\
				[0, covTT, 0],\
				[covEa, 0, covaa]])
		sample = multivariate_normal.rvs(mean=[E,T,a], cov=cov, size=1000000)
		E_set = sample[:,0]
		T_set = sample[:,1]
		a_set = sample[:,2]
		good = np.where((E_set >= 0) & (E_set < 1) & (T_set >= 0) & (T_set <= 1))
		E_set = E_set[good]
		T_set = T_set[good]

		# To classify them according to intrinsic shapes, we need to convert E and T
		# to ***intrinsic*** (c/a, b/a).
		ca_set = 1 - E_set
		ba_set = (1 - T_set * (1 - ca_set**2))**0.5
		is_disk = (ba_set - 1)**2 + (ca_set)**2 <= 0.4**2
		disk_num = len(np.where(is_disk)[0])
		elon_num = len(np.where((~is_disk) & (ca_set < 1 - ba_set))[0])
		sphe_num = len(np.where((~is_disk) & (ca_set > 1 - ba_set))[0])
		tot_num = disk_num + elon_num + sphe_num
		self.shape_fracs = {}
		self.shape_fracs['prolate'] = elon_num * 1.0 / tot_num
		self.shape_fracs['oblate'] = disk_num * 1.0 / tot_num
		self.shape_fracs['spheroidal'] = sphe_num * 1.0 / tot_num

		# print out the fractions of different shapes
		print 'disk fraction: ', disk_num * 1.0 / tot_num
		print 'prolate fraction: ', elon_num * 1.0 / tot_num
		print 'spheroidal fraction: ', sphe_num * 1.0 / tot_num

		fig, ax = plt.subplots(figsize=(7,7))

		# Start making plots showing the ***intrinsic*** c/a-b/a distribution.
		mock, xedges, yedges = np.histogram2d(ca_set, ba_set, bins=(20,20), range=[[0,1],[0,1]])
		cmap = plt.get_cmap('coolwarm')
		
		# Drawing boundaries between different shape definitions.
		ax.imshow(np.flip(mock, 0), extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], aspect='auto', cmap=cmap)
		ax.plot([0,1],[0,1], color='yellow')
		ax.plot([0.5, 1 - 0.4 * 0.5**0.5], [0.5, 0.4 * 0.5**0.5], color='yellow')
		x = np.linspace(0.6, 1.0, 1000)
		y = (0.4**2 - (1 - x)**2)**0.5
		ax.plot(x, y, color='yellow')

		ax.set_xlabel('b/a')
		ax.set_ylabel('c/a')
		ax.set_title('%.1f < z < %.1f, %.1f < logM < %.1f' % (z_range[0], z_range[1], mass_range[0], mass_range[1]))
		plt.savefig('baca_%.1fz%.1f_%.1fm%.1f.eps' % (z_range[0], z_range[1], mass_range[0], mass_range[1]))
		# plt.show()
		plt.close()

		return self.shape_fracs

	def MCMC_data_process(self, chain_file):
		# The function to get the best fitting parameters from the file containing
		# all the MCMC steps after burn-in.
		# Parameters:
		# 		chain_file: str
		# 			The file name of the file containing the MCMC steps.
		chain = pd.read_csv(chain_file, names=['walker', 'E1', 'T1', 'a1', 'covEE1', 'covTT1', 'covaa1', 'covEa1', 'lnprob'],\
								usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64, header=None, \
								delim_whitespace=True)

		# We take the median parameter values of the ensemble of MCMC steps
		# with (log-)probabilities within the largest 30% among all the steps as
		# the best fitting parameters.
		lnprob_crit = np.percentile(chain['lnprob'], 70)
		good = np.where(chain['lnprob'] > lnprob_crit)
		p_best = []
		for k in chain.keys():
			if k == 'walker' or k == 'lnprob':
				continue
			med = np.median(np.array(chain[k])[good])
			# print 'median of parameter %s is %f.' % (k, med)
			p_best.append(med)
			# plt.hist(np.array(chain[k])[good])
			# plt.show()
			# plt.close()
		self.p_best = p_best
		# print p_best
		return p_best

if __name__ == '__main__':
	# comm = MPI.COMM_WORLD
	# comm_rank = comm.Get_rank()
	# comm_size = comm.Get_size()
	
	# initialize the object
	shape = ShapeAnalysis()
	shape.GenerateModelHist(save_name='../ETa_lgSMA_correlate_fine.mat')
	
	redshift_range = [2.0, 2.5]
	mass_range = [9.0, 9.5]

	# Here the cor_type denotes whether the observed b/a-lgSMA distribution has been corrected
	# in the way described in the Appendix of Zhang et al. (2019).
	cor_type = 'no_cor'
	chain_file = './%.1fz%.1f_%.1fm%.1f_1pop_half_cov_%s.dat' % (redshift_range[0], redshift_range[1],\
								      mass_range[0], mass_range[1], cor_type)
	shape.ReadData(cat_path='../data_5fields_uncertainty_125+160.mat', mass_range=mass_range, z_range=redshift_range)
		
	# declare the global variables that are to be used in the calculation of log-probability
	# in MCMC.
	E_grid = shape.E_grid
	T_grid = shape.T_grid
	a_grid = shape.a_grid
	ETa_grid = shape.ETa_grid
	ba_lgSMA_bins = shape.ba_lgSMA_bins
	ndim = shape.ndim
	num_obs = shape.num_obs
	bin_obs = shape.bin_obs
	# bin_obs = bin_obs.reshape(20,20)

	# Carry out MCMC. 
	p_best = shape.MCMC(chain_file=chain_file)	

	# Read the file containing the MCMC steps and get the best fitting parameters.
	p_best = shape.MCMC_data_process(chain_file)	

	# Making plots.
	shape.compare_plot(p_best, mass_range=mass_range, z_range=z_range)
	shape.prob_map(p_best, mass_range=mass_range, z_range=z_range)
	shape.shape_frac(p_best, mass_range=mass_range, z_range=z_range)







